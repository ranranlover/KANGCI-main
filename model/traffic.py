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

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def off_diagonal(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr


def read_dream4(size, type):
    GC = dream_read_label(
        r"/home/user/wcj/KANGCI-main/DREAM4 in-silico challenge"
        r"/DREAM4 gold standards/insilico_size" + str(size) + "_" + str(type) + "_goldstandard.tsv",
        size)
    data = sio.loadmat(r'/home/user/wcj/KANGCI-main/DREAM4 in-silico challenge'
                       r"/DREAM4 training data/insilico_size" + str(size) + "_" + str(type) + '_timeseries.mat')
    data = data['data']
    return GC, data





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
    if not freq_denoise:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3)
        kernel = kernel.repeat(P, 1, 1)

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
            g = grads_abs.permute(0, 2, 1)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)

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
            # gc_row_1st = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)
            #
            # # soft prior，不反传
            # gc_prior = torch.sigmoid(
            #     (gc_row_1st.detach() - tau_prior) * beta_prior
            # )
            #
            # gc_l1_loss = gc_l1_loss + (gc_prior * gc_row.abs()).sum()

            grads_1st = torch.autograd.grad(
                s_j, inp,
                create_graph=False,  # 一阶，不保留计算图
                retain_graph=True
            )[0]

            # 一阶GC计算
            g_1st = grads_1st.abs().squeeze(0)
            gc_row_1st = torch.sqrt((g_1st ** 2).mean(dim=0) + 1e-12)

            # 用一阶引导二阶
            gc_prior = torch.sigmoid(
                (gc_row_1st.detach() - tau_prior) * beta_prior
            )
            gc_l1_loss = gc_l1_loss + (gc_prior * gc_row.abs()).sum()
            # scores = gc_row_1st.detach()
            # rel = torch.softmax(beta_prior * (scores - scores.mean()), dim=0)  # 相对重要性
            # gc_prior = 1.0 - rel  # 一阶大 => rel 大 => 权重小
            # gc_l1_loss += (gc_prior * gc_row.abs()).sum()


        else:
            gc_l1_loss = gc_l1_loss + gc_row.abs().sum()

        # ---- 保存二阶 GC（评估用）----
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc
def compute_gradient_gc_smooth_universal_v3_1220_103v2(
    model,
    input_seq_local,
    create_graph=True,
    lag_agg_local='mean',
    freq_denoise=True,
    cutoff_ratio=0.2,
    lambda_gc=1.0,
    tau=0.1,
    lambd_l=0.005,

    # ===== 新增：一阶 prior 相关 =====
    use_first_order_prior=True,

):
    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # 1. 准备输入
    inp = input_seq_local.detach().clone().requires_grad_(True)
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_refined_loss = 0.0  # 二阶/平滑后的 Loss
    gc_prior_loss = 0.0  # 一阶 Prior Loss

    # 预准备平滑核
    if not freq_denoise:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3).repeat(P, 1, 1)
    # 2. 逐输出变量计算
    for j in range(P):
        s_j = outs[:, j].sum()

        # 计算梯度 (1, T, P)
        # 注意：只有在需要对梯度本身求导时才用 create_graph=True
        grads = torch.autograd.grad(
            s_j, inp,
            create_graph=create_graph,
            retain_graph=True
        )[0].squeeze(0)  # (T, P)

        # ---- 计算一阶 Prior (Raw Gradient Stability) ----
        # 使用 RMS 衡量原始梯度的强度，作为基础约束
        g_raw_rms = torch.sqrt((grads ** 2).mean(dim=0) + 1e-12)
        if use_first_order_prior:
            gc_prior_loss += g_raw_rms.sum()

        # ---- 进行平滑处理 (Refinement) ----
        g_abs = grads.abs()
        if freq_denoise:
            # 在时间维度 T 上做 FFT
            g_fft = torch.fft.rfft(g_abs.T, dim=-1)
            cutoff = max(1, int(g_fft.shape[-1] * cutoff_ratio))
            g_fft[:, cutoff:] = 0.0  # 滤除高频噪声
            g_smooth = torch.fft.irfft(g_fft, n=T, dim=-1).T.abs()
        else:
            g_smooth = F.conv1d(
                F.pad(g_abs.T.unsqueeze(1), (1, 1), mode='reflect'),
                kernel, groups=P
            ).squeeze(1).T

        # ---- 执行指定的聚合策略 (二阶 GC) ----
        if lag_agg_local == 'mean':
            gc_row = g_smooth.mean(dim=0)
        elif lag_agg_local == 'rms':
            gc_row = torch.sqrt((g_smooth ** 2).mean(dim=0) + 1e-12)
        elif lag_agg_local == 'softmax':
            gc_row = (torch.softmax(g_smooth / tau, dim=0) * g_smooth).sum(dim=0)
        elif lag_agg_local == 'quantile':
            gc_row = g_smooth.quantile(0.9, dim=0)
        else:
            gc_row = g_smooth.mean(dim=0)

        # 累加精细化后的 Loss
        gc_refined_loss += gc_row.sum()

        # 保存结果用于返回
        GCs_local[j, :] = gc_row

    # 3. 最终组合 Loss
    # lambda_gc 负责控制最终因果矩阵的稀疏性
    # lambd_l 负责提供一个基于原始梯度的稳定基准
    L_gc = lambda_gc * gc_refined_loss + lambd_l * gc_prior_loss

    return GCs_local, L_gc
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(P, type, epoch, hidden_size, learning_rate,
                                  lambda_gc_sparse_base,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean',
                                  cutoff_ratio=0.2,
                                  # --- 新增/调整的超参数 ---
                                  tau=0.1,
                                tau_prior=0.05,
                                beta_prior=5.0
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

    # --- 数据加载 (假设 load_causaltime_seq 函数存在) ---
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
    model_fwd = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device)


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

        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model_fwd,
            input_seq,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,  # 或 True
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
            tau_prior=tau_prior,
            beta_prior=beta_prior
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
                f" Early stopping triggered after {i + 1} epochs! AUPRC did not improve for {patience} rounds. ")
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


        # avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1208(20, 'traffic', 300, hidden_size=params['hidden_size'],
        #                                                    learning_rate=params['learning_rate']
        #                                                    , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                                    lambda_gc_sparse_base=params['lambda_gc_sparse_base'],lambda_gc_sparse_fusion=params['lambda_gc_sparse_fusion'])
        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(20, 'traffic', 450,
                                                                                hidden_size=params['hidden_size'],
                                                                                learning_rate=params['learning_rate'],
                                                                                lambda_gc_sparse_base=params['lambda_gc_sparse_base'],
                                                                                cutoff_ratio=params['cutoff_ratio'],lag_agg=params['lag_agg'],tau=params['tau'],
                                                                                tau_prior=params['tau_prior'],
                                                                                beta_prior=params['beta_prior'])
        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))

    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc


if __name__ == '__main__':
        # 0.7255 0.55896   0.73212  0.57261     0.7628  0.6249
    param_grid = {
        'hidden_size': [256],  ##128   orgin256
        'learning_rate': [0.005],  # 0.005 0.001  0.0008
        'lambda_gc_sparse_base': [0.0008,0.0009], # 0.0008
        'lag_agg': ['softmax'],#
        'tau': [0.1],
        'cutoff_ratio': [0.65,0.7,0.75],
        'data_path': ['/home/user/wcj/KANGCI-main/realdataset'],
        'tau_prior': [0.03],
        'beta_prior': [17,18,16],
    }
    best_params = grid_search(param_grid)
