import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from synthetic import data_segmentation, simulate_lorenz_96
import time
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt


def compute_roc_1219(gc_label, gc_predict, draw):
    # gc_predict = normalization(gc_predict)
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


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def compute_gradient_gc_smooth_universal_v3_1220_103v1(
        model,
        input_seq_local,
        create_graph=True,
        lag_agg_local='mean',
        freq_denoise=True,
        cutoff_ratio=0.2,
        lambda_gc=1.0,
        tau=0.1,
        h=0.05,

        # ===== 新增：一阶 prior 相关 =====
        use_first_order_prior=True,
        tau_prior=0.05,
        beta_prior=5.0
):
    """
    Gradient-based GC with second-order regularization
    + first-order stability-guided refinement

    返回：
        GCs_local : (P, P)  -- 二阶 GC（仅用于评估/可视化）
        L_gc      : scalar -- 用于 loss.backward() 的 GC 正则项
    """

    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # ---- 输入 requires_grad ----
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 前向 ----
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # ---- 时域平滑核（备用）----
    # if not freq_denoise:
    #     kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3)
    #     kernel = kernel.repeat(P, 1, 1)

    # ======================================================
    #                主循环：逐输出变量
    # ======================================================
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
            # g = grads_abs.permute(0, 2, 1)
            # g = F.pad(g, (1, 1), mode='reflect')
            # g = F.conv1d(g, kernel, groups=P)
            # grads_abs = g.permute(0, 2, 1)
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

        # ==================================================
        # 一阶 GC prior（稳定，仅用于 refinement）
        # ==================================================
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

        # ---- 保存二阶 GC（评估用）----
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc


def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(P, F, epoch, hidden_size, learning_rate,
                                                                lambda_gc_sparse_base,
                                                                gc_create_graph=True, gc_every=1,
                                                                lag_agg='mean',
                                                                cutoff_ratio=0.2,
                                                                tau=0.1,
                                                                grad_clip_quantile=0.99,  # 动态裁剪的分位数
                                                                ):
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    target_device_id = 1
    if torch.cuda.is_available() and torch.cuda.device_count() > target_device_id:
        device_local = torch.device(f'cuda:{target_device_id}')
        # print(f"Using device: {device_local}")
    else:
        device_local = torch.device('cpu')
        # print(f"CUDA device {target_device_id} not available. Falling back to CPU.")

    global device
    device = device_local  # 确保全局或内部变量一致

    X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    model = MLP(P, hidden_size, P).to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model.parameters()), 'lr': learning_rate}
    ])

    loss_fn = nn.MSELoss()

    # --- 训练循环 ---
    smooth_flag = True  # 默认启用平滑
    best_score = -1e9
    best_auprc = -1e9
    # ------------------- 早停参数 -------------------
    patience = 15  # 容忍多少轮没有提升
    min_delta = 1e-5  # 最小提升幅度
    counter = 0  # 计数器
    previous_auroc = -1e9  # 记录上一轮的 AUROC
    for i in range(epoch):
        model.train()
        optimizer.zero_grad()
        inp_fwd = input_seq
        outs_fwd_all = model(inp_fwd).squeeze(0)  # (T, P)
        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)  # L_fwd

        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model,
            input_seq,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,  # 或 True
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
        )

        loss = predict_loss1 + L_gc

        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        GCs_np = GCs_raw.detach().cpu().numpy()
        score1, auprc1 = compute_roc_1219(GC, GCs_np, False)
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
    results = []
    param_list = list(ParameterGrid(param_grid))

    # for params in param_list:
    #     print(f"Training with params: {params}")
    #
    #     avg_score = infer_Grangercausalityv4(40, 40, 300, hidden_size=params['hidden_size'], lam=params['lam'],
    #                                        lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
    #                                        )
    #     results.append((params, avg_score))
    #
    # best_params = max(results, key=lambda x: x[1])
    # print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    results_roc = []
    results_prc = []
    for params in param_list:
        print(f"Training with params: {params}")

        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(40, 20, 300,
                                                                                hidden_size=params['hidden_size'],
                                                                                learning_rate=params['learning_rate'],
                                                                                lambda_gc_sparse_base=params[
                                                                                    'lambda_gc_sparse_base'],
                                                                                cutoff_ratio=params['cutoff_ratio'],
                                                                                tau=params['tau'])

        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))

    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc
    # return best_params


if __name__ == '__main__':
    param_grid = {
        'hidden_size': [256],  ##128   orgin256
        'learning_rate': [0.001],  # 0.005 0.001
        'lambda_gc_sparse_base': [0.1],  #
        'cutoff_ratio': [0.6],
        'lag_agg': ['softmax'],  # 'softmax' 'quantile'
        'tau': [0.1]
    }
    # T=1000 F=40 Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8632855902777778
    # Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7093381769173176

    # param_grid = {
    #     'hidden_size': [10],
    #     'lam': [0.01],
    #     'lam_ridge': [20],
    #     'learning_rate': [0.001]
    # }  ###P=10 F=10

    best_params = grid_search(param_grid)
