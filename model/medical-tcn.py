import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from ComputeROC import compute_roc, compute_auprc
from src.efficient_kan import KAN
# from src.efficient_kan.kan02inter import KAN
from tool import dream_read_label
import scipy.io as sio
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def compute_roc_1219(gc_label, gc_predict, draw):
    gc_label = gc_label.flatten().astype(float)
    gc_predict = gc_predict.flatten().astype(float)
    if draw:
        score = draw_roc_curve(gc_label, gc_predict / gc_predict.max())
        aupr = metrics.average_precision_score(gc_label, gc_predict / gc_predict.max())
    else:
        score = metrics.roc_auc_score(gc_label, gc_predict / gc_predict.max())
        aupr = metrics.average_precision_score(gc_label, gc_predict / gc_predict.max())
    return score, aupr


def draw_roc_curve(label, predict):
    FPR, TPR, P = metrics.roc_curve(label, predict)
    plt.plot(FPR, TPR, 'b*-', label='roc')
    plt.plot([0, 1], [0, 1], 'r--', label="45°")
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    AUC_score = metrics.auc(FPR, TPR)
    return AUC_score
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.net(x)
        if self.downsample:
            x = self.downsample(x)
        return out[:, :, :x.size(2)] + x

class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, levels=10, kernel_size=3):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(TCNBlock(in_ch, hidden_dim, kernel_size, dilation=dilation_size))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.network(x)
        out = out.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        return self.linear(out)

def compute_gradient_gc_smooth_universal_v3_1220_103v1_san(
    model,
    input_seq_local,
    create_graph=True,
    lag_agg_local='mean',
    freq_denoise=True,
    cutoff_ratio=0.2,
    lambda_gc=1.0,
    tau=0.1,
    use_first_order_prior=True,
    tau_prior=0.05,
    beta_prior=5.0
):
    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 前向 ----
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0


    for j in range(P):
        s_j = outs[:, j].sum()

        if create_graph:
            retain = True
        else:
            retain = (j < P - 1)

        grads = torch.autograd.grad(
            s_j,
            inp,
            create_graph=create_graph,
            retain_graph=retain
        )[0]  # (1, T, P)

        grads_abs = grads.abs()

        # ---- 平滑 ----
        if freq_denoise:
            g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)
            n_freq = g_fft.shape[-1]
            cutoff = max(1, int(n_freq * cutoff_ratio))
            mask = torch.zeros_like(g_fft)
            mask[..., :cutoff] = 1.0
            g_fft = g_fft * mask
            grads_abs = torch.fft.irfft(
                g_fft,
                n=grads_abs.shape[1],
                dim=-1
            ).transpose(1, 2).abs()
        else:
            pass
            print('no use fre')


        # ---- lag 聚合 ----
        g = grads_abs.squeeze(0)  # (T, P)

        if lag_agg_local == 'mean':
            gc_row = g.mean(dim=0)

        elif lag_agg_local == 'max':
            gc_row = g.max(dim=0)[0]

        elif lag_agg_local == 'rms':
            gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

        elif lag_agg_local == 'lp':
            p = 2.0
            gc_row = (g.pow(p).mean(dim=0) + 1e-12).pow(1.0 / p)

        elif lag_agg_local == 'softmax':
            w = torch.softmax(g / tau, dim=0)
            gc_row = (w * g).sum(dim=0)

        elif lag_agg_local == 'quantile':
            gc_row = g.quantile(0.9, dim=0)

        else:
            raise ValueError(f"Unknown lag_agg_local: {lag_agg_local}")


        if use_first_order_prior:
            gc_row_1st = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

            # soft prior，不反传
            gc_prior = torch.sigmoid(
                (gc_row_1st.detach() - tau_prior) * beta_prior
            )

            gc_l1_loss = gc_l1_loss + (gc_prior * gc_row.abs()).sum()
        else:
            gc_l1_loss = gc_l1_loss + gc_row.abs().sum()
            print('no use prior')

        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc

def load_causaltime_seq(data_path, device, scaler_type='minmax', n_max=None):
    """
    加载 CausalTime 数据，使其格式与 DREAM4 兼容：
        input_seq:  (1, T-1, P)
        target_seq: (T-1, P)
        reversed_input_seq:  (1, T-1, P)
        reversed_target_seq: (T-1, P)
        mask: (P, P)
        X_raw: (T, P)  标准化后的原始序列

    参数：
        data_path: 包含 data.npy 和 graph.npy 的文件夹
        device:    torch device
        scaler_type: 标准化方式
        n_max:     可选，限制变量数（P）
    """

    # ------------------------------
    # 1. 读取 data.npy (T, P)
    # ------------------------------
    data_file = os.path.join(data_path, "gen_data.npy")
    print(f"Loading data from {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"data.npy not found in {data_path}")

    X = np.load(data_file)  # (T, P)
    if X.ndim == 3:
        # 假设形状是 (N_samples, T_per_sample, P) = (480, 40, 40)
        N_samples, T_per_sample, P_original = X.shape
        T_total = N_samples * T_per_sample
        print(f"Loaded 3D data with shape {X.shape}. Flattening to 2D shape ({T_total}, {P_original}).")

        # 将前两个维度 (480 * 40 = 19200) 合并为总时间步 T_total
        X = X.reshape(T_total, P_original)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif X.ndim != 2:
        # 如果不是 2D 也不是 3D，则抛出错误
        raise ValueError(f"Data array must be 2D (T, P) or 3D (N_samples, T_per_sample, P). Got shape {X.shape}")
    # ------------------------------
    # 2. 可选：裁剪维度 P
    # ------------------------------
    # if n_max is not None and X.shape[1] > n_max:
    #     X = X[:, :n_max]
    P_original = X.shape[1]
    if n_max is not None and P_original > n_max:
        # P_original = 40, n_max = 20
        print(f"Truncating features from P={P_original} to P={n_max} to match graph dimensions.")
        X = X[:, :n_max]
    # ------------------------------
    # 3. 插值缺失值
    # ------------------------------
    # mask_nan = np.isnan(X)
    # X_masked = np.ma.masked_array(X, mask_nan)
    # X_interp = pd.DataFrame(X_masked).interpolate(limit_direction='both').values
    # X = np.nan_to_num(X_interp)

    mask_nan = np.isnan(X)
    X_masked = np.ma.masked_array(X, mask_nan)
    # X_interp = pd.DataFrame(X_masked).interpolate(limit_direction='both').values  # 此时不会报错
    X_interp = pd.DataFrame(X_masked).interpolate(limit_direction='both').values
    X = np.nan_to_num(X_interp)
    # ------------------------------
    # 4. 标准化
    # ------------------------------
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # T, P = X.shape  # T 现在是 T_total (19200), P 是 20
    # X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(T, P)
    # X = X_scaled
    T, P_effective = X.shape  # T 是 19200, P_effective 现在是 20
    X_scaled = scaler.fit_transform(X.reshape(-1, 1)).reshape(T, P_effective)
    X = X_scaled

    # ------------------------------
    # 5. 加载 graph.npy (P, P)
    # ------------------------------
    graph_file = os.path.join(data_path, "graph.npy")
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"graph.npy not found in {data_path}")

    mask = np.load(graph_file, allow_pickle=True)
    print(mask.shape)  # 此时应该是 (20, 20)

    # 可选裁剪 (这里 n_max 应该和 X 的维度 P_effective 相同)
    if n_max is not None and mask.shape[0] > n_max:
        mask = mask[:n_max, :n_max]

    # 确保 mask 和 X 的维度匹配
    if mask.shape[0] != P_effective:
        raise ValueError(
            f"Feature dimension P ({P_effective}) and GC mask dimension ({mask.shape[0]}) mismatch after preprocessing.")

    # ------------------------------
    # 6 & 7. 构造序列和反向序列 (保持不变，使用 T 和 P_effective)
    # ------------------------------
    test_x = X[:T - 1]   # X[:T - 1]
    test_y = X[1:T]    #  X[1:T]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T-1, 20)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T-1, 20)

    X_rev = np.ascontiguousarray(X[::-1])
    rev_x = X_rev[:T - 1]
    rev_y = X_rev[1:T]

    reversed_input_seq = torch.tensor(rev_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(rev_y, dtype=torch.float32).to(device)

    return input_seq, target_seq, reversed_input_seq, reversed_target_seq, mask, X

    # """
    # # 5. 加载 graph.npy (P, P)
    # # ------------------------------
    # graph_file = os.path.join(data_path, "graph.npy")
    #
    # if not os.path.exists(graph_file):
    #     raise FileNotFoundError(f"graph.npy not found in {data_path}")
    #
    # mask = np.load(graph_file, allow_pickle=True)
    # print(mask.shape)
    # # 可选裁剪
    # if n_max is not None and mask.shape[0] > n_max:
    #     mask = mask[:n_max, :n_max]
    #
    # # ------------------------------
    # # 6. 构造序列（与 DREAM4 完全一致）
    # # ------------------------------
    # test_x = X[:T - 1]   # (T-1, P)
    # test_y = X[1:T]      # (T-1, P)
    #
    # input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T-1, P)
    # target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)              # (T-1, P)
    #
    # # ------------------------------
    # # 7. 反向序列
    # # ------------------------------
    # X_rev = np.ascontiguousarray(X[::-1])
    # rev_x = X_rev[:T - 1]
    # rev_y = X_rev[1:T]
    #
    # reversed_input_seq = torch.tensor(rev_x, dtype=torch.float32).unsqueeze(0).to(device)
    # reversed_target_seq = torch.tensor(rev_y, dtype=torch.float32).to(device)
    #
    # return input_seq, target_seq, reversed_input_seq, reversed_target_seq, mask, X
    # """
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(P, type, epoch, hidden_size, learning_rate,
                                  lambda_gc_sparse_base,
                                  lag_agg='mean',
                                  cutoff_ratio=0.2,
                                    tau=10.0,
                                  ):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    target_device_id = 1
    if torch.cuda.is_available() and torch.cuda.device_count() > target_device_id:
        device_local = torch.device(f'cuda:{target_device_id}')

    else:
        device_local = torch.device('cpu')


    global device
    device = device_local  # 确保全局或内部变量一致
    try:
        # 假设 load_causaltime_seq 返回 (1, T, P) 和 (T, P) 的 torch.tensor
        # 实际数据加载逻辑可能需要调整以适应你的 CausalTime 数据集
        data_root = f"/home/user/wcj/KANGCI-main/realdataset/{type}/"
        input_seq, target_seq, reversed_input_seq, reversed_target_seq, GC, X_raw = \
            load_causaltime_seq(data_root, device_local, scaler_type='minmax', n_max=P)
        # print('1111')
        if isinstance(GC, np.ndarray):
            GC = GC.astype(float)
    except NameError:
        pass

    P_effective = input_seq.shape[2]

    # --- 模型构建 ---
    model_fwd = TCNModel(P, hidden_size, P).to(device)

    params = list(model_fwd.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate}
    ])

    loss_fn = nn.MSELoss()

    # --- 训练循环 ---
    smooth_flag = True  # 默认启用平滑
    best_score = -1e9
    best_auprc = -1e9
    # ------------------- 早停参数 -------------------
    patience = 20  # 容忍多少轮没有提升
    min_delta = 1e-5  # 最小提升幅度
    counter = 0  # 计数器
    previous_auroc = -1e9  # 记录上一轮的 AUROC
    for i in range(epoch):
        model_fwd.train()
        optimizer.zero_grad()
        inp_fwd = input_seq
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)  # L_fwd


        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1_san(
            model_fwd,
            input_seq,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,  # 或 True
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau
        )

        loss = predict_loss1 + L_gc


        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        GCs_np = GCs_raw.detach().cpu().numpy()
        score1,auprc1 = compute_roc_1219(GC, GCs_np, False)
        # update best by AUPRC fusion primarily, score secondarily
        if auprc1 > best_auprc:
            best_auprc = auprc1
        if score1 > best_score:
            best_score = score1
        if score1 < previous_auroc - min_delta:
            counter += 1  # 连续下降，计数器加 1
        else:
            counter = 0  # 指标没有下降（持平或上升），重置计数器
        previous_auroc = score1


        print(f"Epoch [{i + 1}/{epoch}] loss: {loss.item():.6f} | predict_loss1: {predict_loss1.item():.6f} | "
              f"Lsparse_fwd: {L_gc.item():.6f} | "
              f"score1: {score1:.4f}  | "
              f"AUPRC_fwd: {auprc1:.4f} | ")

        if counter > 0:
            print(f"  --- No improvement counter: {counter} / {patience} ---")

        # ------------------- 终止条件 (新增) -------------------
        if counter >= patience:
            print(
                f"🌟🌟🌟 Early stopping triggered after {i + 1} epochs! AUPRC did not improve for {patience} rounds. 🌟🌟🌟")
            break  # 退出 for i in range(epoch) 循环
    # End training
    # return best_score, best_auprc
    return best_score, best_auprc
def grid_search(param_grid):
    results_roc = []
    results_prc = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")



        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(20, 'medical', 500, hidden_size=params['hidden_size'],
                                                           learning_rate=params['learning_rate'],
                                                           lambda_gc_sparse_base=params['lambda_gc_sparse_base'],cutoff_ratio=params['cutoff_ratio'],
                                                           tau=params['tau'],lag_agg=params['lag_agg'])

        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))

    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc


if __name__ == '__main__':

    # softmax  91 90   9210 9053      new 9286 9204    0.93112
    param_grid = {
        'hidden_size': [128],  ##128   orgin256
        'learning_rate': [0.0008],  # 0.005 0.001   0.0009 best?    0.0008 auroc 0.93223
        'lambda_gc_sparse_base': [0.01],  #0.005
        'cutoff_ratio': [0.6],
        'lag_agg': ['softmax'],
        'data_path': ['/home/user/wcj/KANGCI-main/realdataset'],
        'tau': [0.1],  # 新增参数，用于 CSDRF 细化

    }
    best_params = grid_search(param_grid)
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.8920377867746289
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.8945821090061016
