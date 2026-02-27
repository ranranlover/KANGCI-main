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
    return score,aupr
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
def compute_gradient_gc_smooth_universal_v3_1220_103v1_pred(
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
    T, P = input_seq_local.shape[1], 40

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
            gc_row_1st = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

            # soft prior，不反传
            gc_prior = torch.sigmoid(
                (gc_row_1st.detach() - tau_prior) * beta_prior
            )

            gc_l1_loss = gc_l1_loss + (gc_prior * gc_row.abs()).sum()
        else:
            gc_l1_loss = gc_l1_loss + gc_row.abs().sum()

        # ---- 保存二阶 GC（评估用）----
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc
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
            gc_row_1st = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

            # soft prior，不反传
            gc_prior = torch.sigmoid(
                (gc_row_1st.detach() - tau_prior) * beta_prior
            )

            gc_l1_loss = gc_l1_loss + (gc_prior * gc_row.abs()).sum()
        else:
            gc_l1_loss = gc_l1_loss + gc_row.abs().sum()

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

    # --- 数据生成 & 8:2 划分 ---
    X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    split_idx = int(length * 0.8)

    train_x = X[:split_idx]
    train_y = X[1:split_idx + 1]

    val_x = X[split_idx - 1:length - 1]
    val_y = X[split_idx:length]

    train_input = torch.tensor(train_x, dtype=torch.float32).unsqueeze(0).to(device)
    train_target = torch.tensor(train_y, dtype=torch.float32).to(device)

    val_input = torch.tensor(val_x, dtype=torch.float32).unsqueeze(0).to(device)
    val_target = torch.tensor(val_y, dtype=torch.float32).to(device)

    # --- 模型 & 优化器 ---
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.Adam(model_fwd.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_score = -1e9
    best_auprc = -1e9

    patience = 20
    min_delta = 1e-5
    counter = 0
    previous_auroc = -1e9

    for i in range(epoch):
        model_fwd.train()
        optimizer.zero_grad()

        # --- 训练预测损失 ---
        outs_fwd_all = model_fwd(train_input).squeeze(0)
        losses_fwd = [loss_fn(outs_fwd_all[:, j], train_target[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)

        # --- 训练因果约束（只在训练集） ---
        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model_fwd,
            train_input,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
        )

        loss = predict_loss1 + L_gc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fwd.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 验证集：只算预测 MSE（不算 GC） ---
        model_fwd.eval()
        with torch.no_grad():
            val_outs = model_fwd(val_input).squeeze(0)
            val_losses = [loss_fn(val_outs[:, j], val_target[:, j]) for j in range(P)]
            val_predict_loss = sum(val_losses)

        # --- 评估指标（只用训练集的 GC 输出） ---
        GCs_np = GCs_raw.detach().cpu().numpy()
        score1, auprc1 = compute_roc_1219(GC, GCs_np, False)



        print(f"Epoch [{i + 1}/{epoch}] "
              f"train_loss: {loss.item():.6f} | train_mse: {predict_loss1.item():.6f} | train_Lgc: {L_gc.item():.6f} | "
              f"val_mse: {val_predict_loss.item():.6f} | "
              f"score: {score1:.4f} | AUPRC: {auprc1:.4f}")



    return best_score, best_auprc
def infer_Grangercausalityv(
        P, F, epoch, hidden_size, learning_rate,
        lambda_gc_sparse_base,
        lag_agg='mean',
        cutoff_ratio=0.2,
        tau=0.1
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
    device = device_local

    # --- 数据生成 & 8:2 划分 ---
    X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    split_idx = int(length * 0.6)

    # 训练集
    train_x = X[:split_idx]
    train_y = X[1:split_idx + 1]

    # # 验证集（注意对齐）
    # val_x = X[split_idx - 1:length - 1]
    # val_y = X[split_idx:length]

    # 验证集 - 与训练集完全分离
    val_x = X[split_idx:length - 1]  # 从split_idx开始，不包含训练集数据
    val_y = X[split_idx + 1:length]  # 对应的目标值

    train_input = torch.tensor(train_x, dtype=torch.float32).unsqueeze(0).to(device)
    train_target = torch.tensor(train_y, dtype=torch.float32).to(device)

    val_input = torch.tensor(val_x, dtype=torch.float32).unsqueeze(0).to(device)
    val_target = torch.tensor(val_y, dtype=torch.float32).to(device)

    # --- 模型 & 优化器 ---
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.Adam(model_fwd.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # 记录最小值
    min_train_mse = float('inf')
    min_val_mse = float('inf')
    mse = float('inf')

    for i in range(epoch):
        model_fwd.train()
        optimizer.zero_grad()

        # --- 训练预测损失（MSE）---
        outs_fwd_all = model_fwd(train_input).squeeze(0)
        train_mse = loss_fn(outs_fwd_all, train_target)

        # --- 训练因果约束（只在训练集）---
        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model_fwd,
            train_input,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
        )

        # 这里真正用于优化的 loss = train_mse + L_gc
        loss = train_mse + L_gc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fwd.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 验证集：只算预测 MSE（不算 GC） ---
        model_fwd.eval()
        with torch.no_grad():
            val_outs = model_fwd(val_input).squeeze(0)
            val_mse = loss_fn(val_outs, val_target)

            val_outs = model_fwd(val_input).squeeze(0)




            # 所有维度误差的总和 (Table 9 中的 MSE 列)
            # 公式: sum((pred - target)^2) / num_samples
            # 相当于 val_mse_mean * P
            val_mse_system_total = torch.mean(torch.sum((val_outs - val_target)**2, dim=1))
        # 更新最小值
        if train_mse.item() < min_train_mse:
            min_train_mse = train_mse.item()
        if val_mse.item() < min_val_mse:
            min_val_mse = val_mse.item()
        if val_mse_system_total.item() < mse:
            mse = val_mse_system_total.item()
        print(f"Epoch [{i + 1}/{epoch}] "
              f"train_mse: {train_mse.item():.6f} | train_gc_loss: {L_gc.item():.6f} | "
              f"val_mse: {val_mse.item():.6f}|"
              f"loss: {val_mse_system_total.item():.6f}")

    # 最终输出最小值
    print("==============================================")
    print(f"Training Loss: {min_train_mse:.6f}")
    print(f"Testing Loss: {min_val_mse:.6f}")
    print(f"MSE: {mse:.6f}")
    print("==============================================")

    return min_train_mse, min_val_mse
def infer_Grangercausalitynocausal(
        P, F, epoch, hidden_size, learning_rate,

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
    device = device_local

    # # --- 数据生成 & 8:2 划分 ---
    # X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # length = X.shape[0]
    #
    # split_idx = int(length * 0.8)
    #
    # # 训练集
    # train_x = X[:split_idx]
    # train_y = X[1:split_idx + 1]
    #
    # # # 验证集（注意对齐）
    # # val_x = X[split_idx - 1:length - 1]
    # # val_y = X[split_idx:length]
    #
    # # 验证集 - 与训练集完全分离
    # val_x = X[split_idx:length - 1]  # 从split_idx开始，不包含训练集数据
    # val_y = X[split_idx + 1:length]  # 对应的目标值
    #
    # train_input = torch.tensor(train_x, dtype=torch.float32).unsqueeze(0).to(device)
    # train_target = torch.tensor(train_y, dtype=torch.float32).to(device)
    #
    # val_input = torch.tensor(val_x, dtype=torch.float32).unsqueeze(0).to(device)
    # val_target = torch.tensor(val_y, dtype=torch.float32).to(device)
    #
    # # --- 模型 & 优化器 ---
    # model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    # optimizer = torch.optim.Adam(model_fwd.parameters(), lr=learning_rate)
    # loss_fn = nn.MSELoss()

    # --- 数据生成 ---
    X_raw, _ = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)

    # --- 1. 严格划分数据集以防泄露 ---
    split_idx = int(len(X_raw) * 0.8)
    train_data = X_raw[:split_idx]
    val_data = X_raw[split_idx:]

    # --- 2. 仅在训练集上拟合 Scaler ---
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    val_data_scaled = scaler.transform(val_data)  # 使用训练集的参数

    # 构建训练对
    train_x = torch.tensor(train_data_scaled[:-1], dtype=torch.float32).unsqueeze(0).to(device)
    train_y = torch.tensor(train_data_scaled[1:], dtype=torch.float32).to(device)

    # 构建验证对
    val_x = torch.tensor(val_data_scaled[:-1], dtype=torch.float32).unsqueeze(0).to(device)
    val_y = torch.tensor(val_data_scaled[1:], dtype=torch.float32).to(device)

    # --- 模型 ---
    model = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    # 记录最小值
    min_train_mse = float('inf')
    min_val_mse = float('inf')
    mse = float('inf')

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        # --- 训练预测损失（MSE）---
        outs_fwd_all = model(train_x).squeeze(0)
        train_mse = loss_fn(outs_fwd_all, train_y)

        # # --- 训练因果约束（只在训练集）---
        # GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
        #     model_fwd,
        #     train_input,
        #     create_graph=True,
        #     lag_agg_local=lag_agg,
        #     freq_denoise=True,
        #     cutoff_ratio=cutoff_ratio,
        #     lambda_gc=lambda_gc_sparse_base,
        #     tau=tau,
        # )

        # 这里真正用于优化的 loss = train_mse + L_gc
        loss = train_mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 验证集：只算预测 MSE（不算 GC） ---
        model.eval()
        with torch.no_grad():
            val_outs = model(val_x).squeeze(0)
            val_mse = loss_fn(val_outs, val_y)

            val_mse_system_total = torch.mean(torch.sum((val_outs - val_y)**2, dim=1))


        # 更新最小值
        if train_mse.item() < min_train_mse:
            min_train_mse = train_mse.item()
        if val_mse.item() < min_val_mse:
            min_val_mse = val_mse.item()
        if val_mse_system_total.item() < mse:
            mse = val_mse_system_total.item()
        print(f"Epoch [{i + 1}/{epoch}] "
              f"train_mse: {train_mse.item():.6f} | "
              f"val_mse: {val_mse.item():.6f} | "
              f"val_mse: {val_mse_system_total.item():.6f} | ")

    # 最终输出最小值
    print("==============================================")
    print(f"Training Loss: {min_train_mse:.6f}")
    print(f"Testing Loss: {min_val_mse:.6f}")
    print(f"MSE: {mse:.6f}")
    print("==============================================")

    return min_train_mse, min_val_mse


import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler


# 假设 KAN, simulate_lorenz_96, compute_gradient_gc_... 已在外部定义

def create_windowed_dataset(data, window_size):
    """
    将序列数据转换为滑动窗口样本
    输入 shape: [T, P]
    输出 shape: [Batch, Window, P], [Batch, P]
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i: i + window_size])
        y.append(data[i + window_size])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32)


def infer_Grangercausalityv1(
        P, F, epoch, hidden_size, learning_rate,
        lambda_gc_sparse_base,
        window_size=3,  # 新增：窗口大小
        lag_agg='mean',
        cutoff_ratio=0.2,
        tau=0.1
):
    # --- 基础配置 ---
    global_seed = 1
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # --- 数据生成与预处理 ---
    X_raw, _ = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)

    # 划分数据集
    split_idx = int(len(X_raw) * 0.6)
    train_raw = X_raw[:split_idx]
    val_raw = X_raw[split_idx:]

    # 标准化
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    # 构造滑动窗口数据集
    # train_input: [Batch, Window, P], train_target: [Batch, P]
    train_input, train_target = create_windowed_dataset(train_scaled, window_size)
    val_input, val_target = create_windowed_dataset(val_scaled, window_size)

    train_input, train_target = train_input.to(device), train_target.to(device)
    val_input, val_target = val_input.to(device), val_target.to(device)

    # --- 模型与优化器 ---
    # KAN 输入维度需要平铺窗口：P * window_size
    model = KAN([P * window_size, hidden_size, P], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_mse = float('inf')
    record_mse_total = float('inf')
    record_train_loss = 0.0

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        # 平铺窗口以适应模型输入: [Batch, Window * P]
        train_x_flat = train_input.view(train_input.shape[0], -1)
        preds = model(train_x_flat)

        # 1. 预测损失 (Mean MSE)
        train_mse = loss_fn(preds, train_target)

        # 2. 因果约束损失 (Jacobian)
        # 注意：这里传入计算 Jacobin 的输入需要是 unsqueeze(0) 形式以适配你之前的函数
        _, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1_pred(
            model,
            train_x_flat.unsqueeze(0),  # 适配原函数的 [1, T, P] 预期
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
        )

        total_loss = train_mse + L_gc
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # --- 验证 ---
        model.eval()
        with torch.no_grad():
            val_x_flat = val_input.view(val_input.shape[0], -1)
            val_preds = model(val_x_flat)

            # Testing Loss (Mean MSE)
            val_mse_mean = loss_fn(val_preds, val_target)
            # System MSE (Sum over P, Mean over T)
            val_mse_system = torch.mean(torch.sum((val_preds - val_target) ** 2, dim=1))

        if val_mse_mean.item() < best_val_mse:
            best_val_mse = val_mse_mean.item()
            record_mse_total = val_mse_system.item()
            record_train_loss = total_loss.item()

        if (i + 1) % 50 == 0:
            print(
                f"Epoch [{i + 1}/{epoch}] Train_L: {total_loss.item():.4f} | Val_Mean: {val_mse_mean.item():.4f} | Sys_MSE: {val_mse_system.item():.4f}")

    print("\n================ JRNGC Result ================")
    print(f"Training Loss: {record_train_loss:.6f}")
    print(f"Testing Loss:  {best_val_mse:.6f}")
    print(f"MSE (System):  {record_mse_total:.6f}")
    print("==============================================\n")
    return record_train_loss, best_val_mse, record_mse_total


def infer_Grangercausalitynocausalv1(
        P, F, epoch, hidden_size, learning_rate,
        window_size=3
):
    # --- 基础配置 ---
    global_seed = 1
    torch.manual_seed(global_seed)
    np.random.seed(global_seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # --- 数据处理 ---
    X_raw, _ = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    split_idx = int(len(X_raw) * 0.8)
    train_raw = X_raw[:split_idx]
    val_raw = X_raw[split_idx:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_raw)
    val_scaled = scaler.transform(val_raw)

    train_input, train_target = create_windowed_dataset(train_scaled, window_size)
    val_input, val_target = create_windowed_dataset(val_scaled, window_size)
    train_input, train_target = train_input.to(device), train_target.to(device)
    val_input, val_target = val_input.to(device), val_target.to(device)

    # --- 模型 ---
    model = KAN([P * window_size, hidden_size, P], base_activation=nn.Identity).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    best_val_mse = float('inf')
    record_mse_total = 0.0

    for i in range(epoch):
        model.train()
        optimizer.zero_grad()

        train_x_flat = train_input.view(train_input.shape[0], -1)
        preds = model(train_x_flat)
        loss = loss_fn(preds, train_target)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_x_flat = val_input.view(val_input.shape[0], -1)
            val_preds = model(val_x_flat)
            val_mse_mean = loss_fn(val_preds, val_target)
            val_mse_system = torch.mean(torch.sum((val_preds - val_target) ** 2, dim=1))

        if val_mse_mean.item() < best_val_mse:
            best_val_mse = val_mse_mean.item()
            record_mse_total = val_mse_system.item()

    print("\n============== No Causal Result ==============")
    print(f"Testing Loss:  {best_val_mse:.6f}")
    print(f"Training Loss:  {loss:.6f}")
    print(f"MSE (System):  {record_mse_total:.6f}")
    print("==============================================\n")
    return best_val_mse, record_mse_total
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
    # results_roc = []
    # results_prc = []
    for params in param_list:
        print(f"Training with params: {params}")

        infer_Grangercausalityv(40, 40, 100,hidden_size=params['hidden_size'],
                                                                                learning_rate=params['learning_rate'],
                                                                                lambda_gc_sparse_base=params[
                                                                                    'lambda_gc_sparse_base'],
                                                                                cutoff_ratio=params['cutoff_ratio'],
                                                                                tau=params['tau'])
        print('--------------no caual-----------------')
        infer_Grangercausalitynocausal(40, 40, 100, hidden_size=params['hidden_size'],
                                learning_rate=params['learning_rate'])

        # results_roc.append((params, avg_score[0]))
        # results_prc.append((params, avg_score[1]))

    # best_params_roc = max(results_roc, key=lambda x: x[1])
    # best_params_prc = max(results_prc, key=lambda x: x[1])
    # print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    # print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    # return best_params_roc
    # return best_params


if __name__ == '__main__':

    # param_grid = {
    #     'hidden_size': [15],
    #     'lam': [0.01],
    #     'lam_ridge': [16],
    #     'learning_rate': [0.001]
    # }  # T=500 p=10 F=10 AUROC=1.0

    # param_grid = {
    #     'hidden_size': [80],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.001]
    # }  ###best   AUROC=0.99  P=40 F=40

    param_grid = {
        'hidden_size': [256],  ##128   orgin256
        'learning_rate': [0.001],  # 0.005 0.001
        'lambda_gc_sparse_base': [0.1],  #
        'cutoff_ratio': [0.6],
        'lag_agg': ['softmax'],
        'tau': [0.1]
    }
    #T=1000 F=40 Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9991015625
    #Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9937831684293081
    #F=20/F=10 都是1
    # !!!compute_gradient_gc_smooth_universal_v3_1220_103v1  same parameters:
    # T=1000 F=40 Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.999296875
    # Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9934984729065727
    # F=20/F=10 都是1


    # T=500 F=40 Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9928993055555556
    # Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9577398581618286
    # !!!compute_gradient_gc_smooth_universal_v3_1220_103v1  same parameters:
    #Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9934157986111111
    #Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9579820321766964






    # param_grid = {
    #     'hidden_size': [10],
    #     'lam': [0.01],
    #     'lam_ridge': [20],
    #     'learning_rate': [0.001]
    # }  ###P=10 F=10

    best_params = grid_search(param_grid)
# 200
# Training Loss: 0.976375
# Testing Loss: 1.023402

# Training Loss: 0.015377
# Testing Loss: 0.947162
