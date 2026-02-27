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


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    # W0 = network.layers[0].base_weight  # shape: [H, D]
    # col_norms = torch.sqrt(torch.sum(W0 ** 2, dim=0) + 1e-12)  # group lasso 部分
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
        # return lam * torch.sum(col_norms)
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


import torch


def regularize_se(network, lam, penalty, lr, sens=None, sens_boost=0.8):
    x = network.layers[0].base_weight  # (H, D)
    if penalty == 'GL':
        col_norms = torch.norm(x, dim=0)
        if sens is not None:
            sens = sens.to(x.device)
            sens = (sens - sens.min()) / (sens.max() - sens.min() + 1e-12)
            weight_factor = 1.0 - sens_boost * sens  # 高敏感度 → 惩罚更小
            col_norms = col_norms * weight_factor
        return lam * torch.sum(col_norms)
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize_jacobian_g(model, x, target_idx, lam_jr):
    """
    计算模型输出相对于输入 x 的雅可比矩阵的 Frobenius 范数平方，作为正则化项。

    Args:
        model: KAN 模型实例 (models[j])
        x: 输入序列 (input_seq 或 reversed_input_seq)，形状 (1, T, P)
        target_idx: 当前模型预测的目标维度 j
        lam_jr: 雅可比正则化强度 (lambda_jr)

    Returns:
        标量的雅可比正则化损失。
    """

    # 1. 克隆输入并设置为需要梯度
    x_cloned = x.clone().detach().requires_grad_(True)

    # 2. 模型前向传播
    # KAN[P, hidden, 1] 模型接受 (B, T, P) 或 (T, P) 的输入，并输出 (B, T, 1) 或 (T, 1)
    # 这里的 x 是 (1, T, P)，模型输出 y_pred 是 (1, T, 1)
    # 我们关注的是 y_pred[:, :, 0] (形状为 (T,))
    y_pred = model(x_cloned)  # 形状 (1, T, 1)

    # 将输出展平为 (T,)
    y_flat = y_pred.view(-1)
    T = y_flat.size(0)

    # 3. 循环计算 Frobenius 范数平方 ||J||_F^2 的近似
    # J_ij = d(y_i) / d(x_j)
    jacobian_norm_sq = 0.0

    # 仅针对目标输出维度计算雅可比，即 d(y_flat) / d(x_cloned)
    # y_flat 是 T 维向量，x_cloned 是 (1*T*P) 维向量。J 是 T x (T*P) 矩阵

    # 因为 y_flat 的每个元素 y_t 只依赖于 x_t，所以 J 是一个稀疏矩阵，
    # 但由于 KAN 是全连接的，如果模型的定义是 y_t = f(x_t)，则雅可比是 P 维的
    #
    # **重要假设**:
    # KAN 模型的计算是 y_t = model(x_t)，模型在序列维度 T 上是共享参数的。
    # model(input_seq) 实际上是 batch_size=1 下对 T 个时间步的 $x_t \in R^P$ 的并行计算。
    #
    # 雅可比正则化应惩罚 $\left\| \frac{\partial y_t}{\partial x_t} \right\|_F^2$

    # 我们计算 d(y_t)/d(x_t) 的 Frobenius 范数平方，并对所有 T 求平均

    # x_cloned 形状为 (1, T, P)。我们需要遍历 T 个时间步
    for t in range(T):
        # 提取当前时间步的输出 y_t (标量) 和输入 x_t (P 维向量)
        y_t = y_flat[t].view(1)  # 当前时间步的预测值 (标量)
        x_t = x_cloned[0, t, :].view(1, -1)  # 当前时间步的输入 (1, P)

        # 计算 d(y_t) / d(x_t)
        # 注意：这里需要确保只对 x_cloned 在 t 处的梯度进行计算
        # 一种更简单且常见的方法是直接计算 d(y_flat) / d(x_cloned) 整个矩阵的范数。

        # --- 采用整体雅可比矩阵的 Frobenius 范数平方 ---
        # 这是一个简化且常用的近似，尽管它包含了跨时间步的梯度 (d(y_t)/d(x_{t'}))，
        # 但在序列模型中，由于因果性，这些跨时间步的梯度在理论上应为零。

        # 创建一个和 y_flat 形状相同的张量，用于计算 d(y_i)/d(x)
        grad_output = torch.zeros_like(y_flat)
        grad_output[t] = 1.0  # 仅关注 y_t

        # 计算 d(y_t) / d(x) (形状: 1, T, P)
        grads = torch.autograd.grad(
            y_flat, x_cloned, grad_outputs=grad_output,
            create_graph=True, retain_graph=True
        )[0]

        # 仅惩罚 d(y_t)/d(x_t) 部分的范数（即对角块）
        # grads[0, t, :] 是 d(y_t)/d(x_t)
        jacobian_norm_sq += (grads[0, t, :] ** 2).sum()

    # 对所有时间步的范数平方求平均
    jacobian_reg_loss = lam_jr * (jacobian_norm_sq / T)

    return jacobian_reg_loss


def ridge_regularize(model, lam_ridge):
    '''Apply ridge penalty at all subsequent layers.'''
    total_weight_sum = 0
    for layer in model.layers[1:]:
        weight_squared_sum = torch.sum(layer.base_weight ** 2)
        total_weight_sum += weight_squared_sum
    result = lam_ridge * total_weight_sum
    return result


def infer_Grangercausality(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate):
    # Set seed for random number generation (for reproducibility of results)
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    score = 0

    best_score = 0
    total_score = 0

    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda(1)
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda(1)

    X2 = X[::-1, :]  # reverse data
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda(1)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda(1)

    # component-wise generate p models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)

    models = nn.ModuleList(networks)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)

    for i in range(epoch):
        losses1 = []
        losses2 = []
        for j in range(0, P):
            network_output = models[j](input_seq).view(-1)
            loss_i = loss_fn(network_output, target_seq[:, j])
            losses1.append(loss_i)

        for j in range(P, 2 * P):
            network_output = models[j](reversed_input_seq).view(-1)
            loss_i = loss_fn(network_output, reversed_target_seq[:, j - P])
            losses2.append(loss_i)
        predict_loss1 = sum(losses1)
        predict_loss2 = sum(losses2)

        ridge_loss1 = sum([ridge_regularize(model, lam_ridge) for model in models[:P]])
        ridge_loss2 = sum([ridge_regularize(model, lam_ridge) for model in models[P:2 * P]])
        regularize_loss1 = sum([regularize(model, lam, "GL", learning_rate) for model in models[:P]])
        regularize_loss2 = sum([regularize(model, lam, "GL", learning_rate) for model in models[P:2 * P]])
        # regularize_loss1 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[:P]])
        # regularize_loss2 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # regularize_loss1 = sum([regularize_jacobian(m, lam, "JAC", learning_rate) for m in models[:P]])
        # regularize_loss2 = sum([regularize_jacobian(m, lam, "JAC", learning_rate) for m in models[P:2 * P]])

        GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        loss = predict_loss1 + predict_loss2 + regularize_loss1 + regularize_loss2 + ridge_loss1 + ridge_loss2

        # GCs = []
        # GC2s = []
        # for k in range(P):
        #     GCs.append(models[k].GC().detach().cpu().numpy())
        # GCs = np.array(GCs)
        #
        # for k in range(P, 2 * P):
        #     GC2s.append(models[k].GC().detach().cpu().numpy())
        # GC2s = np.array(GC2s)
        # GCs = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P, 2 * P)], dim=0)
        # if predict_loss1 < predict_loss2 and regularize_loss1 < regularize_loss2:
        #     result = GCs
        # elif predict_loss1 > predict_loss2 and regularize_loss1 > regularize_loss2:
        #     result = GC2s
        # else:
        #     result = np.where(
        #         np.abs(GCs.cpu().numpy() - GC2s.cpu().numpy()) < 0.05,
        #         (GCs + GC2s) / 2,
        #         np.maximum(GCs.cpu().numpy(), GC2s.cpu().numpy())
        #     )
        #
        # GCs = off_diagonal(GCs)
        # GC2s = off_diagonal(GC2s)
        #
        # result = off_diagonal(result)
        GCs_cpu = GCs.detach().cpu().numpy()
        GC2s_cpu = GC2s.detach().cpu().numpy()

        if predict_loss1 < predict_loss2 and regularize_loss1 < regularize_loss2:
            result = GCs_cpu
        elif predict_loss1 > predict_loss2 and regularize_loss1 > regularize_loss2:
            result = GC2s_cpu
        else:
            # 使用 CPU 数组进行所有操作
            condition = np.abs(GCs_cpu - GC2s_cpu) < 0.05
            true_value = (GCs_cpu + GC2s_cpu) / 2
            false_value = np.maximum(GCs_cpu, GC2s_cpu)
            result = np.where(condition, true_value, false_value)

        # 使用 CPU 数组进行后续操作
        GCs = off_diagonal(GCs_cpu)
        GC2s = off_diagonal(GC2s_cpu)
        result = off_diagonal(result)
        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_score < score_fusion:  # and score_fusion > 0.57
            best_score = score_fusion

            best_GCs = GCs.copy()  # 使用numpy的copy方法
            best_GC2s = GC2s.copy()  # 使用numpy的copy方法
            best_fused = result.copy()  # 使用numpy的copy方法
            # np.savetxt(
            #     f"Type={type},score={score_fusion},ridge={lam_ridge}, hidden_size={hidden_size},"
            #     f"learning_rate={learning_rate},lam={lam},epoch={i}.txt",
            #     result, fmt=f'%.5f')
        total_score += score
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}, '
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}')
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
    print('Score:' + str(best_score))
    return score


class FusionEdge(nn.Module):
    """
    Per-edge fusion network: map per-edge features -> alpha_ij in (0,1).
    Vectorized: accepts (P*P, feat_dim) and outputs (P*P,1) which we reshape to (P,P).
    """

    def __init__(self, in_dim=5, hidden=64):  # orign in_dim=5
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, max(16, hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(16, hidden // 2), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, in_dim) where N = P*P; returns (N,1)
        return self.net(x)


def compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                          target_seq, reversed_target_seq,
                          models, losses_fwd, losses_rev):
    """
    输入:
        GCs, GC2s: (P,P) forward/backward 的 GC 矩阵
        outs_fwd, outs_rev: list of (T-1,) 预测
        target_seq, reversed_target_seq: (T-1,P) 序列
        models: ModuleList of 2P models
        losses_fwd, losses_rev: 每个模型的 loss
    输出:
        feat_edges: (P*P, feat_dim)
    """
    P = GCs.shape[0]

    # --- 基础分数 ---
    g_fwd = GCs.view(-1)
    g_rev = GC2s.view(-1)
    absdiff = torch.abs(g_fwd - g_rev)

    # --- 残差方差比 ---
    res_var_fwd = torch.stack([
        torch.var((outs_fwd[j] - target_seq[:, j]).detach())
        for j in range(P)
    ])
    res_var_rev = torch.stack([
        torch.var((outs_rev[j] - reversed_target_seq[:, j]).detach())
        for j in range(P)
    ])
    res_ratio = (res_var_fwd / (res_var_rev + 1e-12)).unsqueeze(1).repeat(1, P).view(-1)

    # --- 梯度方向相似度 ---
    # grad_sim = compute_grad_similarity(models[:P], models[P:2 * P], losses_fwd, losses_rev, P)

    # --- in/out degree ---
    in_degree = GCs.sum(0).view(-1)  # (P,)
    out_degree = GCs.sum(1).view(-1).repeat(P)  # (P*P,)
    in_degree_rep = in_degree.unsqueeze(0).repeat(P, 1).view(-1)

    # --- 拼接 ---
    feat_edges = torch.stack([
        g_fwd.detach(),
        g_rev.detach(),
        absdiff.detach(),
        res_ratio.detach(),
        # grad_sim.detach(),
        in_degree_rep.detach(),
        out_degree.detach()
    ], dim=1)

    # --- 标准化 ---
    feat_edges = torch.log1p(torch.abs(feat_edges))
    feat_mean = feat_edges.mean(dim=0, keepdim=True)
    feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
    feat_edges = (feat_edges - feat_mean) / feat_std

    return feat_edges


def infer_Grangercausalityv4(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate, lambda_alpha_reg,
                             lambda_consistency):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    # X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}  # 0.01
    ])

    loss_fn = nn.MSELoss()

    for i in range(epoch):

        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward predictions
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        # GCs = torch.stack([models[k].GC_integrated(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)

        # print(GCs)
        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)

        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)

        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s
        # print(fused_GC_tensor)
        alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]
            out_r = outs_rev[j]
            out_r_aligned = torch.flip(out_r, dims=[0])
            a = alpha_row[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        consistency_loss = torch.norm(GCs - GC2s, p=2)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        lambda_fused = 1.0
        # lambda_alpha_reg = 5.0
        # lambda_consistency = 0.05
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                lambda_consistency * consistency_loss +
                lambda_fused * fused_predict_loss -
                lambda_alpha_reg * alpha_reg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- EMA 更新 ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()
        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
    return best_score


EPS = 1e-8


def compute_edge_features_timelag(GCs, GC2s, outs_fwd, outs_rev,
                                  target_seq, reversed_target_seq,
                                  models, losses_fwd, losses_rev):
    """
    生成每条边的固定长特征向量（不 stack 时间序列，避免长度不一致错误）。
    Inputs:
      GCs: tensor (P, P) forward GC matrix (aggregated)
      GC2s: tensor (P, P) reverse GC matrix
      outs_fwd: list of length P, each element tensor (T1,) predictions for target j from forward models
      outs_rev: list of length P, each element tensor (T2,) predictions for target j from reverse models
      target_seq: tensor (T1, P)
      reversed_target_seq: tensor (T2, P)
      models: list/modulelist of models
      losses_fwd, losses_rev: lists of loss scalars per target (length P)
    Returns:
      feat_edges: tensor shape (P*P, 6) on same device as GCs
      Ordering: for target j in 0..P-1, source i in 0..P-1, index = j*P + i
    """
    device = GCs.device
    P = GCs.shape[0]
    feats = []

    T_f = target_seq.shape[0]
    T_r = reversed_target_seq.shape[0]

    # precompute per-target scalars
    forward_mse = torch.tensor(
        [losses_fwd[j].detach().cpu().item() if isinstance(losses_fwd[j], torch.Tensor) else float(losses_fwd[j])
         for j in range(P)], dtype=torch.float32, device=device)  # (P,)
    reverse_mse = torch.tensor(
        [losses_rev[j].detach().cpu().item() if isinstance(losses_rev[j], torch.Tensor) else float(losses_rev[j])
         for j in range(P)], dtype=torch.float32, device=device)  # (P,)

    # For correlation computations we need tensors on device
    # But outs_fwd[j] is already tensor on device; target_seq[:, j] is on device

    for j in range(P):  # target
        out_f = outs_fwd[j]  # (T_f,)
        tgt_f = target_seq[:, j]  # (T_f,)
        out_r = outs_rev[j]  # (T_r,)
        tgt_r = reversed_target_seq[:, j]  # (T_r,)

        # align reversed out to forward time by flipping (only for comparing predictions)
        try:
            out_r_aligned = torch.flip(out_r, dims=[0])
        except Exception:
            out_r_aligned = out_r.clone()

        # summary scalars (forward)
        # mean absolute difference between forward and reversed (aligned) predictions (use min length)
        min_len = min(out_f.shape[0], out_r_aligned.shape[0])
        if min_len > 0:
            # align last min_len
            of = out_f[-min_len:]
            ora = out_r_aligned[-min_len:]
            mean_abs_diff_outs = torch.mean(torch.abs(of - ora)).to(device)
        else:
            mean_abs_diff_outs = torch.tensor(0.0, device=device)

        # pearson corr between out_f and tgt_f
        if out_f.shape[0] > 1:
            of_mean = torch.mean(out_f)
            tgt_mean = torch.mean(tgt_f)
            cov = torch.mean((out_f - of_mean) * (tgt_f - tgt_mean))
            denom = (torch.std(out_f) * torch.std(tgt_f) + EPS)
            corr = (cov / denom).clamp(-1.0, 1.0)
            corr = corr.to(device)
        else:
            corr = torch.tensor(0.0, device=device)

        # Now for each source i produce feature vector for edge (i -> j)
        for i in range(P):
            feat = torch.stack([
                GCs[j, i].to(device),  # forward aggregated GC (target j, source i)
                GC2s[j, i].to(device),  # reverse aggregated GC
                forward_mse[j].to(device),  # scalar
                reverse_mse[j].to(device),  # scalar
                mean_abs_diff_outs.to(device),  # scalar
                corr.to(device)  # scalar
            ])
            feats.append(feat)

    feat_edges = torch.stack(feats, dim=0)  # (P*P, 6)
    return feat_edges


def infer_Grangercausalityv5(P, L, type, epoch, hidden_size, lam, lam_ridge,
                             learning_rate, lambda_alpha_reg, lambda_consistency):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # --- 数据读取 ---
    GC, X = read_dream4(P, type)  # GC: (P,P), X: (T,P)
    GC = off_diagonal(GC)

    # --- 构造滞后数据 ---
    lagged_X, y = make_lagged_data(X, L)  # lagged_X: (T-L, P*L), y: (T-L,P)

    input_seq = torch.tensor(lagged_X, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T-L, P*L)
    target_seq = torch.tensor(y, dtype=torch.float32).to(device)  # (T-L, P)

    # --- 构造反向数据 ---
    X2 = np.ascontiguousarray(X[::-1, :])
    lagged_X2, y2 = make_lagged_data(X2, L)
    reversed_input_seq = torch.tensor(lagged_X2, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(y2, dtype=torch.float32).to(device)

    # --- 构建 2P 个模型 ---
    networks = []
    for _ in range(2 * P):
        network = KAN([P * L, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}
    ])

    loss_fn = nn.MSELoss()

    for i in range(epoch):
        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # --- forward predictions ---
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- 正则 ---
        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # --- GC 矩阵 ---
        GCs = torch.stack([models[k].GC(L) for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC(L) for k in range(P, 2 * P)], dim=0)

        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features_timelag(GCs, GC2s, outs_fwd, outs_rev,
                                                   target_seq, reversed_target_seq,
                                                   models, losses_fwd, losses_rev)

        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        # print(alphas.shape,GCs.shape,GC2s.shape)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        # --- 融合预测 ---
        alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]
            out_r = outs_rev[j]
            out_r_aligned = torch.flip(out_r, dims=[0])
            a = alpha_row[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        # --- 一致性与正则 ---
        consistency_loss = torch.norm(GCs - GC2s, p=2)
        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        lambda_fused = 1.0
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                lambda_consistency * consistency_loss +
                lambda_fused * fused_predict_loss +
                lambda_alpha_reg * alpha_reg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- EMA 更新 ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    return best_score


# def infer_Grangercausalityv4_inter(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate, lambda_alpha_reg,
#                              lambda_consistency):
#     global device
#     global_seed = 1
#     torch.manual_seed(global_seed)
#     torch.cuda.manual_seed_all(global_seed)
#     np.random.seed(global_seed)
#
#     best_score = 0
#     best_score1 = 0
#     best_score2 = 0
#
#     # simulate and preprocess
#     # X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
#     GC, X = read_dream4(P, type)
#     GC = off_diagonal(GC)
#
#     length = X.shape[0]
#
#     test_x = X[:length - 1, :]
#     test_y = X[1:length, :]
#     input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
#     target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)
#
#     X2 = np.ascontiguousarray(X[::-1, :])
#     X2 = np.ascontiguousarray(X2)
#
#     reversed_x = X2[:length - 1, :]
#     reversed_y = X2[1:length, :]
#
#     reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
#     reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)
#
#     # build 2P models
#     networks = []
#     for _ in range(2 * P):
#         network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
#         networks.append(network)
#     models = nn.ModuleList(networks)
#
#     fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)
#
#     params = list(models.parameters()) + list(fusion_edge.parameters())
#     optimizer = torch.optim.Adam([
#         {'params': list(models.parameters()), 'lr': learning_rate},
#         {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}  # 0.01
#     ])
#
#     loss_fn = nn.MSELoss()
#
#     for i in range(epoch):
#
#         models.train()
#         fusion_edge.train()
#         optimizer.zero_grad()
#
#         # forward predictions
#         outs_fwd, losses_fwd = [], []
#         for j in range(0, P):
#             out_j = models[j](input_seq).view(-1)
#             outs_fwd.append(out_j)
#             losses_fwd.append(loss_fn(out_j, target_seq[:, j]))
#
#         outs_rev, losses_rev = [], []
#         for j in range(P, 2 * P):
#             out_j = models[j](reversed_input_seq).view(-1)
#             outs_rev.append(out_j)
#             losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))
#
#         predict_loss1 = sum(losses_fwd)
#         predict_loss2 = sum(losses_rev)
#
#         ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
#         ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
#         regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
#         regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])
#
#         GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
#         GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)
#
#         # GCs = torch.stack([models[k].GC(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
#         # GC2s = torch.stack([models[k].GC(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
#         #                    dim=0)
#
#         # print(GCs)
#         # --- 提取 edge 特征 ---
#         feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
#                                            target_seq, reversed_target_seq,
#                                            models, losses_fwd, losses_rev)
#
#         alphas_flat = fusion_edge(feat_edges)
#         alphas = alphas_flat.view(P, P)
#
#         fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s
#         # print(fused_GC_tensor)
#         alpha_row = torch.mean(alphas, dim=1, keepdim=True)
#         fused_predict_loss = torch.tensor(0.0, device=device)
#         for j in range(P):
#             out_f = outs_fwd[j]
#             out_r = outs_rev[j]
#             out_r_aligned = torch.flip(out_r, dims=[0])
#             a = alpha_row[j].view(1)
#             fused_pred = a * out_f + (1.0 - a) * out_r_aligned
#             fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])
#
#         consistency_loss = torch.norm(GCs - GC2s, p=2)
#
#         eps = 1e-8
#         entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
#         alpha_reg = torch.mean(entropy)
#
#         lambda_fused = 1.0
#         # 在构造 loss 前（已得到 GCs, GC2s）
#         sparsity_lambda = 1e-3  # 调参
#         graph_sparsity_loss = sparsity_lambda * (GCs.abs().sum() + GC2s.abs().sum())
#         # 或者你想 L2: graph_sparsity_loss = sparsity_lambda * (GCs.pow(2).sum()+GC2s.pow(2).sum())
#
#         loss = (predict_loss1 + predict_loss2 +
#                 regularize_loss1 + regularize_loss2 +
#                 ridge_loss1 + ridge_loss2 +
#                 lambda_consistency * consistency_loss +
#                 lambda_fused * fused_predict_loss -
#                 lambda_alpha_reg * alpha_reg +
#                 graph_sparsity_loss)  # <- 新增这一项
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
#         optimizer.step()
#
#         # --- EMA 更新 ---
#         if 'alpha_ema' not in locals():
#             alpha_ema = alphas.detach().clone()
#         else:
#             alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()
#
#         fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
#         score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)
#
#         with torch.no_grad():
#             fused_np = fused_GC_tensor.detach().cpu().numpy()
#             GCs_np = GCs.detach().cpu().numpy()
#             GC2s_np = GC2s.detach().cpu().numpy()
#
#             score1 = compute_roc(GC, off_diagonal(GCs_np), False)
#             score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
#             score_fusion = compute_roc(GC, off_diagonal(fused_np), False)
#
#             if best_score < score_fusion:
#                 best_score = score_fusion
#                 best_score1 = score1
#                 best_score2 = score2
#
#         print(
#             f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
#             f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
#             f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
#             f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
#             f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
#         )
#
#     return best_score
def infer_Grangercausalityv4_inter(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate, lambda_alpha_reg,
                                   lambda_consistency):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}
    ])

    loss_fn = nn.MSELoss()
    lambda_graph = 1e-2  # graph supervision weight

    for i in range(epoch):

        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward predictions
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        # extract edge features
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)

        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)

        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s
        alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]
            out_r = outs_rev[j]
            out_r_aligned = torch.flip(out_r, dims=[0])
            a = alpha_row[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        consistency_loss = torch.norm(GCs - GC2s, p=2)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # graph sparsity
        sparsity_lambda = 1e-3
        graph_sparsity_loss = sparsity_lambda * (GCs.abs().sum() + GC2s.abs().sum())

        # graph supervision loss
        graph_supervision_loss = F.binary_cross_entropy_with_logits(
            fused_GC_tensor.view(-1),
            torch.tensor(GC, dtype=torch.float32, device=device).view(-1)
        )

        # total loss
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                lambda_consistency * consistency_loss +
                fused_predict_loss -
                lambda_alpha_reg * alpha_reg +
                graph_sparsity_loss +
                lambda_graph * graph_supervision_loss)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # EMA update
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        # compute ROC
        with torch.no_grad():
            fused_eval = torch.sigmoid(alpha_ema * GCs + (1 - alpha_ema) * GC2s)
            fused_np = fused_eval.detach().cpu().numpy()
            GCs_np = torch.sigmoid(GCs).detach().cpu().numpy()
            GC2s_np = torch.sigmoid(GC2s).detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    return best_score


def make_lagged_data(X, L):
    """
    构造滞后数据:
    X: (T, P)
    return:
        lagged_X: (T-L, P*L)
        y: (T-L, P)
    """
    T, P = X.shape
    lagged_X = []
    for t in range(L, T):
        xt = []
        for l in range(1, L + 1):
            xt.append(X[t - l, :])  # t-1, t-2, ..., t-L
        xt = np.concatenate(xt, axis=0)  # (P*L,)
        lagged_X.append(xt)
    lagged_X = np.stack(lagged_X, axis=0)  # (T-L, P*L)
    y = X[L:, :]  # 对应预测目标
    return lagged_X, y


def infer_Grangercausalityv4_inge(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate, lambda_alpha_reg,
                                  lambda_consistency):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    # X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}  # 0.01
    ])

    loss_fn = nn.MSELoss()

    for i in range(epoch):

        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward predictions
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # regularize_loss1 = sum([regularize_jacobian(m, lam, "JAC", learning_rate) for m in models[:P]])
        # regularize_loss2 = sum([regularize_jacobian(m, lam, "JAC", learning_rate) for m in models[P:2 * P]])

        # GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        # GCs = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P, 2 * P)], dim=0)
        GCs = torch.stack([models[k].GC_integrated_v1(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        GC2s = torch.stack(
            [models[k].GC_integrated_v1(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
            dim=0)
        # GCs = torch.stack([models[k].GC_integrated(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)
        # GCs = torch.stack([models[k].GC_integrated_v11(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated_v11(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],dim=0)

        # print(GCs)
        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)

        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)

        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s
        # print(fused_GC_tensor)
        # alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        # fused_predict_loss = torch.tensor(0.0, device=device)
        # for j in range(P):
        #     out_f = outs_fwd[j]
        #     out_r = outs_rev[j]
        #     out_r_aligned = torch.flip(out_r, dims=[0])
        #     a = alpha_row[j].view(1)
        #     fused_pred = a * out_f + (1.0 - a) * out_r_aligned
        #     fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        consistency_loss = torch.norm(GCs - GC2s, p=2)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        lambda_fused = 1.0
        # lambda_alpha_reg = 5.0
        # lambda_consistency = 0.05
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                lambda_consistency * consistency_loss -
                lambda_alpha_reg * alpha_reg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- EMA 更新 ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '

            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    return best_score


def infer_Grangercausalityv4_inge_plus_v1(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,
                                          lambda_alpha_reg, lambda_consistency,
                                          lambda_gc_sparse=0.05, gc_create_graph=True, gc_every=1):
    """
    改写版：使用基于输入-输出梯度的 GC 替换原先基于权重/sensitivity 的 GC 估计。
    新增参数:
      - lambda_gc_sparse: ℓ1 正则强度 (λ)，用于对 GC 矩阵稀疏化；设为 0 则不把稀疏项加入 loss（仅做监控）
      - gc_create_graph: 如果 True，则计算 GC 时使用 create_graph=True，从而允许 Lsparse 反向传播至模型参数（会产生二阶导）
      - gc_every: 每隔多少个 epoch 才计算/应用 GC 稀疏项（减少开销）
    其余签名与原来保持一致。
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P) or (T,)

    X2 = np.ascontiguousarray(X[::-1, :])
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a list of models ----
    def compute_gradient_gc_for_models(models_subset, input_seq_local, create_graph=False):
        """
        Compute GC matrix (P x P) using gradient method:
          For each target i (model i), s_i = sum_t out_i(t)
          grads = grad(s_i, input_seq_local, create_graph=create_graph)
          GC[i, j] = mean_t |grads[0, t, j]|
        Args:
          models_subset: iterable of models (length P)
          input_seq_local: tensor (1, T, P)
          create_graph: whether to create grad graph (for backprop through GC)
        Returns:
          GCs: tensor shape (P, P) on same device
        """
        device_local = input_seq_local.device
        P_local = len(models_subset)
        T_local = input_seq_local.shape[1]

        # make a detached copy that requires grad (so autograd.grad works cleanly)
        inp = input_seq_local.detach().clone().requires_grad_(True)

        GCs_local = torch.zeros((P_local, input_seq_local.shape[2]), device=device_local)

        # iterate targets
        for i_model, model_i in enumerate(models_subset):
            out_i = model_i(inp).view(-1)  # assume shape (T,)
            si = out_i.sum()  # scalar
            grads = torch.autograd.grad(si, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # take absolute and mean across time dim (dim=1)
            # grads.abs().mean(dim=1) -> shape (1, P) -> squeeze to (P,)
            gc_row = grads.abs().mean(dim=1).squeeze(0)
            GCs_local[i_model, :] = gc_row

        return GCs_local

    # ---------------------------------------------------------------

    # training loop
    for i in range(epoch):
        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward predictions (unchanged)
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        # ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        # regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        # regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # --- compute GC matrices using gradient method ---
        # compute every gc_every epochs to save compute if desired
        if (i % gc_every) == 0:
            # if lambda_gc_sparse > 0 and gc_create_graph True, the computed GCs will carry a graph
            GCs = compute_gradient_gc_for_models(models[:P], input_seq, create_graph=gc_create_graph)
            GC2s = compute_gradient_gc_for_models(models[P:2 * P], reversed_input_seq, create_graph=gc_create_graph)
        else:
            # if not computing this epoch, compute without graph for monitoring (cheap)
            with torch.no_grad():
                GCs = compute_gradient_gc_for_models(models[:P], input_seq, create_graph=False)
                GC2s = compute_gradient_gc_for_models(models[P:2 * P], reversed_input_seq, create_graph=False)

        # --- compute fused alphas and fused GC tensor as before ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        # consistency + alpha entropy regularization (as before)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # --- GC sparsity loss: ℓ1 on gradient-based GC (optionally backpropagatable) ---
        if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
            # If gc_create_graph=True and lambda_gc_sparse>0, this will be part of computational graph
            Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)

        # combine loss: note alpha_reg kept with minus sign as in your original code
        loss = (predict_loss1 + predict_loss2 -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- EMA 更新 for alphas (unchanged) ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '

            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    return best_score


def infer_Grangercausalityv4_inge_plus_v2(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,
                                          lambda_alpha_reg, lambda_consistency,
                                          lambda_gc_sparse=0.05, gc_create_graph=False, gc_every=1):
    """
    改写版：使用基于输入-输出梯度的 GC 替换原先基于权重/sensitivity 的 GC 估计。
    新增参数:
      - lambda_gc_sparse: ℓ1 正则强度 (λ)，用于对 GC 矩阵稀疏化；设为 0 则不把稀疏项加入 loss（仅做监控）
      - gc_create_graph: 如果 True，则计算 GC 时使用 create_graph=True，从而允许 Lsparse 反向传播至模型参数（会产生二阶导）
      - gc_every: 每隔多少个 epoch 才计算/应用 GC 稀疏项（减少开销）
    最终损失按照你文字描述：L = L_pred + L_sparse，其中 L_sparse = λ * ||GC||_1
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P) or (T,)

    X2 = np.ascontiguousarray(X[::-1, :])
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(models.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a list of models ----
    def compute_gradient_gc_for_models(models_subset, input_seq_local, create_graph=False):
        """
        Compute GC matrix (P x P) using gradient method:
          For each target i (model i), s_i = sum_t out_i(t)
          grads = grad(s_i, input_seq_local, create_graph=create_graph)
          GC[i, j] = mean_t |grads[0, t, j]|
        Args:
          models_subset: iterable of models (length P)
          input_seq_local: tensor (1, T, P)
          create_graph: whether to create grad graph (for backprop through GC)
        Returns:
          GCs: tensor shape (P, P) on same device
        """
        device_local = input_seq_local.device
        P_local = len(models_subset)
        T_local = input_seq_local.shape[1]

        # make a detached copy that requires grad (so autograd.grad works cleanly)
        inp = input_seq_local.detach().clone().requires_grad_(True)

        GCs_local = torch.zeros((P_local, input_seq_local.shape[2]), device=device_local)

        # iterate targets
        for i_model, model_i in enumerate(models_subset):
            out_i = model_i(inp).view(-1)  # assume shape (T,)
            si = out_i.sum()  # scalar (sum over time) as in your derivation
            grads = torch.autograd.grad(si, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # take absolute and mean across time dim (dim=1)
            gc_row = grads.abs().mean(dim=1).squeeze(0)  # shape (P,)
            GCs_local[i_model, :] = gc_row

        return GCs_local

    # ---------------------------------------------------------------

    # training loop
    for i in range(epoch):
        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward predictions (unchanged)
        outs_fwd, losses_fwd = [], []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev, losses_rev = [], []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- compute GC matrices using gradient method ---
        # compute every gc_every epochs to save compute if desired
        if (i % gc_every) == 0:
            # if gc_create_graph True and lambda_gc_sparse>0, the computed GCs will carry a graph
            # (we choose create_graph according to gc_create_graph so the sparse loss can backprop)
            create_graph_flag = gc_create_graph and (lambda_gc_sparse > 0.0)
            GCs = compute_gradient_gc_for_models(models[:P], input_seq, create_graph=create_graph_flag)
            GC2s = compute_gradient_gc_for_models(models[P:2 * P], reversed_input_seq, create_graph=create_graph_flag)
        else:
            # if not computing this epoch, compute without graph for monitoring (cheap)
            with torch.no_grad():
                GCs = compute_gradient_gc_for_models(models[:P], input_seq, create_graph=False)
                GC2s = compute_gradient_gc_for_models(models[P:2 * P], reversed_input_seq, create_graph=False)

        # --- fused alphas and fused GC tensor (kept for monitoring/evaluation but NOT used in loss) ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        # --- compute sparse GC loss according to the paper/text: Lsparse = lambda * ||GC||_1 ---
        if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
            # If GCs were computed with create_graph=True, these tensors carry graph and Lsparse will backprop
            Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)

        # --- final loss: only prediction loss + GC sparsity (as in your formula L = L_pred + L_sparse) ---
        loss = (predict_loss1 + predict_loss2) + (Lsparse1 + Lsparse2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- EMA 更新 for alphas (unchanged, used for evaluation) ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        # 更清晰的日志：输出预测损失、稀疏损失以及评估分数
        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.6f}, predict_loss2: {predict_loss2.item():.6f}, '
            f'Lsparse1: {Lsparse1.item():.6f}, Lsparse2: {Lsparse2.item():.6f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    return best_score


# def infer_Grangercausalityv4_inge_plus_try(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,
#                                            lambda_alpha_reg, lambda_consistency,
#                                            lambda_gc_sparse,data_path, gc_create_graph=True, gc_every=1,
#                                            lag_agg='mean', normalize_input=False):
#     """
#     改写后：
#     - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
#     - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
#     新增可选参数（非必要改变签名，只是内部可选）:
#       - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
#       - normalize_input: 是否先对 input_seq 做按变量（col）标准化
#     其它行为与原先保持一致（fusion, alpha, sparse 等）。
#     """
#     global device
#     global_seed = 1
#     torch.manual_seed(global_seed)
#     torch.cuda.manual_seed_all(global_seed)
#     np.random.seed(global_seed)
#
#     best_score = 0
#     best_score1 = 0
#     best_score2 = 0
#
#     print(f"Loading real dataset: {type} from {data_path}")
#     GC, X = read_real_dataset(data_path, type, n_max=P, scaler_type='minmax')
#
#     # 自动修正 P 为实际读取到的特征数量
#     if X.shape[1] != P:
#         print(f"Adjusting P from {P} to actual data dimension {X.shape[1]}")
#         P = X.shape[1]
#     length = X.shape[0]
#     test_x = X[:length - 1, :]
#     test_y = X[1:length, :]
#     input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
#     target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)
#
#     X2 = np.ascontiguousarray(X[::-1, :])
#     reversed_x = X2[:length - 1, :]
#     reversed_y = X2[1:length, :]
#
#     reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
#     reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)
#
#     # Optional: compute column-wise mean/std for normalization (to make gradient magnitudes comparable)
#     if normalize_input:
#         # compute on training input (forward); use same stats for reversed for simplicity
#         mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
#         std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
#     else:
#         mean_cols = None
#         std_cols = None
#
#     # build two multi-output models (each outputs P targets for all time steps)
#     # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
#     model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
#     model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
#
#     fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)
#
#     params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
#     optimizer = torch.optim.Adam([
#         {'params': list(model_fwd.parameters()), 'lr': learning_rate},
#         {'params': list(model_rev.parameters()), 'lr': learning_rate},
#         {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
#     ])
#
#     loss_fn = nn.MSELoss()
#
#     # ---- helper: compute gradient-based GC for a multi-output model ----
#     # def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
#     #     """
#     #     Compute GC matrix (P x P) from a single multi-output model:
#     #       - model(input_seq_local) -> outs: (1, T, P)
#     #       - For each output j in [0,P-1]:
#     #           s_j = outs[0, :, j].sum()   # scalar aggregated over time
#     #           grads = grad(s_j, inp) -> shape (1, T, P)
#     #           GC[j, i] = aggregate_t |grads[0,t,i]|  (agg = mean or max)
#     #     Args:
#     #       model: single multi-output model
#     #       input_seq_local: tensor (1, T, P)
#     #       create_graph: whether to create grad graph (for backprop through GC)
#     #       lag_agg_local: 'mean' or 'max'
#     #     Returns:
#     #       GCs: tensor shape (P, P) on same device
#     #     """
#     #     device_local = input_seq_local.device
#     #     P_local = input_seq_local.shape[2]
#     #     T_local = input_seq_local.shape[1]
#     #
#     #     # optionally normalize input to reduce scale effects
#     #     if mean_cols is not None and std_cols is not None:
#     #         inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
#     #     else:
#     #         inp = input_seq_local
#     #
#     #     # ensure a fresh tensor that requires grad
#     #     inp = inp.detach().clone().requires_grad_(True)
#     #
#     #     outs = model(inp).squeeze(0)  # (T, P)
#     #
#     #     GCs_local = torch.zeros((P_local, P_local), device=device_local)
#     #
#     #     # compute per-output gradients
#     #     # Note: this loop does P backward/grad calls; for modest P this is fine and precise.
#     #     for j in range(P_local):
#     #         out_j = outs[:, j]  # (T,)
#     #         s_j = out_j.sum()  # scalar
#     #         grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
#     #         # aggregate across time dim: mean or max
#     #         if lag_agg_local == 'mean':
#     #             gc_row = grads.abs().mean(dim=1).squeeze(0)  # (P,)
#     #         elif lag_agg_local == 'max':
#     #             gc_row = grads.abs().max(dim=1)[0].squeeze(0)  # (P,)
#     #         else:
#     #             # default to mean
#     #             gc_row = grads.abs().mean(dim=1).squeeze(0)
#     #
#     #         GCs_local[j, :] = gc_row
#     #
#     #     return GCs_local  # shape (P, P)
#     def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
#         """
#         Robust gradient-based GC for a multi-output model.
#         Expected:
#             input_seq_local: (1, T, P)
#         Returns:
#             GCs_local: (P, P), where row j = effects on output j
#         """
#
#         # -----------------------------
#         # 0.  Basic format check
#         # -----------------------------
#         device_local = input_seq_local.device
#         assert input_seq_local.ndim == 3, f"input_seq_local must be (1,T,P), got {input_seq_local.shape}"
#         B, T_local, P_local = input_seq_local.shape
#         assert B == 1, f"Batch size must be 1, got {B}"
#
#         # -----------------------------
#         # 1. Normalize input (if global mean/std exist)
#         # -----------------------------
#         if mean_cols is not None and std_cols is not None:
#             inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
#         else:
#             inp = input_seq_local
#
#         # detach → clone → requires_grad
#         inp = inp.detach().clone().requires_grad_(True)
#
#         # -----------------------------
#         # 2. Forward pass and robust output normalization
#         # -----------------------------
#         outs = model(inp)
#
#         # Case A: model returns (1,T,P)
#         if outs.ndim == 3 and outs.shape[0] == 1:
#             outs = outs.squeeze(0)
#
#         # Case B: model returns (T,P) -> ok
#         elif outs.ndim == 2 and outs.shape == (T_local, P_local):
#             pass
#
#         # Case C: model returns (batch,P) or (P)
#         elif outs.ndim == 2 and outs.shape[0] == 1 and outs.shape[1] == P_local:
#             # replicate prediction across time (constant output)
#             outs = outs.repeat(T_local, 1)
#
#         # Case D: model returns (P,P) or other square shape (common bug)
#         elif outs.ndim == 2 and outs.shape[0] == outs.shape[1] == P_local:
#             # treat output as constant across time
#             outs = outs.unsqueeze(0).repeat(T_local, 1)
#
#         else:
#             raise RuntimeError(f"[GC ERROR] Unexpected model output shape: {outs.shape}. "
#                                f"Expected (1,T,P), (T,P), (1,P), or (P,P).")
#
#         # Finally ensure correct shape
#         if outs.shape != (T_local, P_local):
#             raise RuntimeError(f"[GC ERROR] After normalization, outs has shape {outs.shape}, "
#                                f"expected {(T_local, P_local)}")
#
#         # -----------------------------
#         # 3. Allocate GC matrix
#         # -----------------------------
#         GCs_local = torch.zeros((P_local, P_local), device=device_local)
#
#         # -----------------------------
#         # 4. Loop over output dimensions j
#         # -----------------------------
#         for j in range(P_local):
#             out_j = outs[:, j]  # shape (T,)
#             s_j = out_j.sum()  # scalar
#
#             grads = torch.autograd.grad(
#                 s_j, inp,
#                 create_graph=create_graph,
#                 retain_graph=True,
#                 allow_unused=True
#             )[0]
#
#             # if no gradient depends on this output → row all zero
#             if grads is None:
#                 gc_row = torch.zeros(P_local, device=device_local)
#             else:
#                 # ensure grads shape is (1,T,P)
#                 if grads.ndim == 2:
#                     grads = grads.unsqueeze(0)
#                 if grads.ndim != 3:
#                     raise RuntimeError(f"[GC ERROR] grads has wrong shape: {grads.shape}, expected (1,T,P)")
#
#                 # ---- aggregate along TIME dimension ----
#                 if lag_agg_local == 'mean':
#                     gc_row = grads.abs().mean(dim=1).squeeze(0)  # (P,)
#                 elif lag_agg_local == 'max':
#                     gc_row = grads.abs().max(dim=1)[0].squeeze(0)
#                 else:
#                     gc_row = grads.abs().mean(dim=1).squeeze(0)
#
#                 # final length check
#                 if gc_row.numel() != P_local:
#                     gc_row = gc_row.view(-1)[:P_local]
#                     if gc_row.numel() != P_local:
#                         raise RuntimeError(f"[GC ERROR] gc_row size mismatch: {gc_row.shape}")
#
#             # assign row j
#             GCs_local[j, :] = gc_row
#
#         return GCs_local
#
#     # ---------------------------------------------------------------
#
#     # training loop
#     for i in range(epoch):
#         model_fwd.train()
#         model_rev.train()
#         fusion_edge.train()
#         optimizer.zero_grad()
#
#         # optionally normalize inputs for forward/backward pass (same scheme as GC)
#         if mean_cols is not None and std_cols is not None:
#             inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
#             inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
#         else:
#             inp_fwd = input_seq
#             inp_rev = reversed_input_seq
#
#         # forward predictions
#         outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
#         outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)
#
#         # compute per-target losses (same as before)
#         losses_fwd = []
#         for j in range(P):
#             losses_fwd.append(loss_fn(outs_fwd_all[:, j], target_seq[:, j]))
#         losses_rev = []
#         for j in range(P):
#             losses_rev.append(loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]))
#
#         predict_loss1 = sum(losses_fwd)
#         predict_loss2 = sum(losses_rev)
#
#         # --- compute GC matrices using gradient method ---
#         if (i % gc_every) == 0:
#             # compute possibly with create_graph to allow Lsparse backprop
#             GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=gc_create_graph,
#                                                 lag_agg_local=lag_agg)
#             GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=gc_create_graph,
#                                                  lag_agg_local=lag_agg)
#         else:
#             # monitoring only (no graph)
#             with torch.no_grad():
#                 GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
#                 GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=False,
#                                                      lag_agg_local=lag_agg)
#
#         # --- compute fused alphas and fused GC tensor as before ---
#         # Note: you used outs_fwd, outs_rev lists previously; adapt to new shape
#         # We provide outs_fwd_list and outs_rev_list (list of per-target tensors) to existing compute_edge_features
#         outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
#         outs_rev = [outs_rev_all[:, j] for j in range(P)]
#
#         feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
#                                            target_seq, reversed_target_seq,
#                                            None, losses_fwd,
#                                            losses_rev)  # models->None as not needed, adapt compute_edge_features if required
#         alphas_flat = fusion_edge(feat_edges)
#         alphas = alphas_flat.view(P, P)
#         fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s
#
#         eps = 1e-8
#         entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
#         alpha_reg = torch.mean(entropy)
#
#         # --- GC sparsity loss: ℓ1 on gradient-based GC (optionally backpropagatable) ---
#         if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
#             Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
#             Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
#         else:
#             Lsparse1 = torch.tensor(0.0, device=device)
#             Lsparse2 = torch.tensor(0.0, device=device)
#
#         # combine loss
#         loss = (predict_loss1 + predict_loss2 -
#                 lambda_alpha_reg * alpha_reg +
#                 Lsparse1 + Lsparse2)
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
#         optimizer.step()
#
#         # --- EMA 更新 for alphas (unchanged) ---
#         if 'alpha_ema' not in locals():
#             alpha_ema = alphas.detach().clone()
#         else:
#             alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()
#
#         fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
#         score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)
#
#         with torch.no_grad():
#             fused_np = fused_GC_tensor.detach().cpu().numpy()
#             GCs_np = GCs.detach().cpu().numpy()
#             GC2s_np = GC2s.detach().cpu().numpy()
#
#             score1 = compute_roc(GC, off_diagonal(GCs_np), False)
#             score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
#             score_fusion = compute_roc(GC, off_diagonal(fused_np), False)
#
#             if best_score < score_fusion:
#                 best_score = score_fusion
#                 best_score1 = score1
#                 best_score2 = score2
#
#                 best_GCs = GCs.detach().clone()
#                 best_GC2s = GC2s.detach().clone()
#                 best_fused = fused_GC_tensor.detach().clone()
#
#         print(
#             f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
#             f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f}, '
#             f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
#         )
#
#     # plot best GC heatmap (unchanged)
#     plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
#
#     return best_score

def infer_Grangercausalityv4_inge_plus_try(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,
                                           lambda_alpha_reg, lambda_consistency,
                                           lambda_gc_sparse,data_path, gc_create_graph=True, gc_every=1,
                                           lag_agg='mean', normalize_input=False):
    """
    Robust and runnable version of your function.
    - Keeps your original signature and behavior.
    - Sanitizes X if it's 3D (e.g. (T,P,P)) by flattening last two dims when sensible.
    - Robust compute_gradient_gc_for_model that handles several model output shapes.
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0.0
    best_score1 = 0.0
    best_score2 = 0.0

    print(f"Loading real dataset: {type} from {data_path}")
    GC, X = read_real_dataset(data_path, type, n_max=P, scaler_type='minmax')

    # === SANITIZE X if user-provided data has unexpected dims (e.g., (T, P, P)) ===
    if isinstance(X, np.ndarray) and X.ndim == 3:
        T0, A, B = X.shape
        if A == B:
            # flatten last two dims to (T, A*A)
            print(f"[WARN] Input X is 3D {X.shape}; flattening last two dims -> (T, {A * B})")
            X = X.reshape(T0, A * B)
        else:
            raise RuntimeError(f"Unsupported data shape {X.shape} — cannot auto-flatten")

    # Ensure X is 2D (T, P)
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise RuntimeError(f"Expected X to be 2D (T,P) after preprocessing, got {type(X)} with shape {getattr(X, 'shape', None)}")

    # Auto-correct P
    if X.shape[1] != P:
        print(f"Adjusting P from {P} to actual data dimension {X.shape[1]}")
        P = X.shape[1]

    length = X.shape[0]
    if length < 2:
        raise RuntimeError("Time dimension too short (need at least 2 time steps)")

    # build train/test sequences (single long sequence concatenated)
    test_x = X[:length - 1, :]   # (T-1, P)
    test_y = X[1:length, :]      # (T-1, P)
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)   # (1, T-1, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)              # (T-1, P)

    # reversed sequences
    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # Optional normalization stats
    if normalize_input:
        mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
    else:
        mean_cols = None
        std_cols = None

    # instantiate models (expect KAN, FusionEdge to exist in your codebase)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # --------------------
    # helper: normalize/flatten model outputs into (T_local, P_local)
    # --------------------
    def _normalize_model_outs(outs, T_local, P_local):
        """
        Accepts various model outputs and returns tensor of shape (T_local, P_local).
        Supported input shapes:
          - (1, T, P) -> squeeze -> (T,P)
          - (T, P)    -> keep
          - (1, P)    -> repeat to (T,P)
          - (P, P)    -> treat as constant over time -> (T,P) (rare)
        Raises descriptive RuntimeError on unexpected shapes.
        """
        # convert to tensor if needed
        if not isinstance(outs, torch.Tensor):
            outs = torch.tensor(outs, device=device)

        if outs.ndim == 3 and outs.shape[0] == 1:
            outs = outs.squeeze(0)
        elif outs.ndim == 2 and outs.shape == (T_local, P_local):
            pass
        elif outs.ndim == 2 and outs.shape[0] == 1 and outs.shape[1] == P_local:
            outs = outs.repeat(T_local, 1)
        elif outs.ndim == 2 and outs.shape[0] == outs.shape[1] == P_local:
            # e.g., (P,P) unexpected, treat each row as a constant output over time -> take first row as proxy
            outs = outs.unsqueeze(0).repeat(T_local, 1)
        else:
            raise RuntimeError(f"[GC ERROR] Unexpected model output shape: {outs.shape}. Expected (1,T,P),(T,P),(1,P) or (P,P).")
        if outs.shape != (T_local, P_local):
            raise RuntimeError(f"[GC ERROR] After normalization, outs shape {outs.shape} != {(T_local, P_local)}")
        return outs

    # --------------------
    # compute_gradient_gc_for_model (robust)
    # --------------------
    def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
        """
        Compute gradient-based GC matrix (P_local x P_local) for a multi-output model.
        Input:
            - model: a model mapping (1,T,P) -> outputs (ideally (1,T,P) or (T,P))
            - input_seq_local: (1, T, P) torch tensor
        Returns:
            - GCs_local: (P, P) torch tensor on same device
        """
        device_local = input_seq_local.device

        # Accept & repair tiny deviations: if 4D because X was (T,P,P)
        if input_seq_local.ndim == 4:
            # try to flatten last two dims if square
            b, t, a, b2 = input_seq_local.shape
            if a == b2:
                input_seq_local = input_seq_local.reshape(b, t, a * b2)
                print(f"[FIX] Flattened input_seq_local from 4D to {input_seq_local.shape}")
            else:
                raise RuntimeError(f"Unexpected 4D input_seq_local shape {input_seq_local.shape}")

        if input_seq_local.ndim != 3:
            raise RuntimeError(f"input_seq_local must be (1,T,P), got {input_seq_local.shape}")

        B, T_local, P_local = input_seq_local.shape
        if B != 1:
            # coerce to batch 1 by taking first batch (preserve generality)
            input_seq_local = input_seq_local[0:1, :, :]
            B = 1

        # normalize if stats available
        if mean_cols is not None and std_cols is not None:
            inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
        else:
            inp = input_seq_local

        # require grad on input copy
        inp = inp.detach().clone().requires_grad_(True)

        # forward
        outs_raw = model(inp)

        # normalize to (T_local, P_local)
        outs = _normalize_model_outs(outs_raw, T_local, P_local)

        GCs_local = torch.zeros((P_local, P_local), device=device_local)

        # compute gradients per-output
        for j in range(P_local):
            out_j = outs[:, j]            # (T_local,)
            s_j = out_j.sum()             # scalar

            grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True, allow_unused=True)[0]

            if grads is None:
                # no dependence -> zeros
                gc_row = torch.zeros(P_local, device=device_local)
            else:
                # grads expected (1, T_local, P_local) or (T_local, P_local)
                if grads.ndim == 2:
                    grads = grads.unsqueeze(0)
                if grads.ndim != 3:
                    raise RuntimeError(f"[GC ERROR] grads has wrong shape: {grads.shape}, expected (1,T,P) or (T,P)")
                # aggregate along time axis
                if lag_agg_local == 'mean':
                    gc_row = grads.abs().mean(dim=1).squeeze(0)   # (P_local,)
                elif lag_agg_local == 'max':
                    gc_row = grads.abs().max(dim=1)[0].squeeze(0)
                else:
                    gc_row = grads.abs().mean(dim=1).squeeze(0)

                # safety trim/reshape
                if gc_row.numel() != P_local:
                    gc_row = gc_row.view(-1)[:P_local]
                    if gc_row.numel() != P_local:
                        raise RuntimeError(f"[GC ERROR] gc_row size mismatch after reshape: {gc_row.shape}")

            GCs_local[j, :] = gc_row

        return GCs_local

    # ---------------------------------------------------------------
    # Training loop (keeps original loss composition)
    # ---------------------------------------------------------------
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # possibly normalize inputs for forward/backward pass
        if mean_cols is not None and std_cols is not None:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        # forward preds (robust squeeze)
        outs_fwd_all = _normalize_model_outs(model_fwd(inp_fwd), inp_fwd.shape[1], P)  # (T, P)
        outs_rev_all = _normalize_model_outs(model_rev(inp_rev), inp_rev.shape[1], P)  # (T, P)

        # per-target losses
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # compute GC matrices
        if (i % gc_every) == 0:
            GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
        else:
            with torch.no_grad():
                GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=False, lag_agg_local=lag_agg)

        # prepare outs lists as before
        outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           None, losses_fwd,
                                           losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)

        loss = (predict_loss1 + predict_loss2 -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # EMA update for alphas
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            score_fusion = compute_roc(GC, off_diagonal(fused_np), False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2
                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        # lightweight progress print
        print(
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    # plot best GC heatmap (if available)
    try:
        plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
    except Exception as e:
        print(f"[WARN] Could not plot heatmap: {e}")

    return best_score

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
import math





import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np


def plot_best_gc_heatmap(GC_true, GCs_best, GC2s_best, fused_best, save_path=None):
    """
    绘制 ROC 最优时的 GC 热力图（2x2 排布）。
    参数：
        GC_true: 真实因果矩阵 (numpy 或 torch.Tensor)
        GCs_best, GC2s_best, fused_best: 最优的三个GC矩阵 (torch.Tensor)
        save_path: 若给定，则保存图像到路径
    """

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    GC_true = to_np(GC_true)
    GCs_best = to_np(GCs_best)
    GC2s_best = to_np(GC2s_best)
    fused_best = to_np(fused_best)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # 2行2列布局
    axes = axes.flatten()

    titles = ["Ground Truth", "Forward GC", "Reverse GC", "Fused GC"]
    matrices = [GC_true, GCs_best, GC2s_best, fused_best]

    for ax, mat, title in zip(axes, matrices, titles):
        sns.heatmap(mat, ax=ax, cmap="YlGnBu", square=True, cbar=True)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Cause variable (X_i)", fontsize=10)
        ax.set_ylabel("Effect variable (Y_j)", fontsize=10)
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# def read_real_dataset(dataset_name, root_path="/home/user/wcj/KANGCI-main/realdataset", scaler_type="minmax", use_graph_as_gc=True):
#     """
#     从 realdataset/{dataset}/gen_data.npy 中读取数据，输出:
#         GC: (P, P)
#         X : (T, P)
#
#     dataset_name: 'medical' / 'pm25' / 'traffic'
#     """
#
#     # def read_dream4(size, type):
#     #     GC = dream_read_label(
#     #         r"/home/user/wcj/KANGCI-main/DREAM4 in-silico challenge"
#     #         r"/DREAM4 gold standards/insilico_size" + str(size) + "_" + str(type) + "_goldstandard.tsv",
#     #         size)
#     #     data = sio.loadmat(r'/home/user/wcj/KANGCI-main/DREAM4 in-silico challenge'
#     #                        r"/DREAM4 training data/insilico_size" + str(size) + "_" + str(type) + '_timeseries.mat')
#     #     data = data['data']
#     #     return GC, data
#     dataset_path = os.path.join(root_path, dataset_name)
#     data_path = os.path.join(dataset_path, "gen_data.npy")
#     graph_path = os.path.join(dataset_path, "graph.npy")
#
#     if not os.path.exists(data_path):
#         raise FileNotFoundError(f"找不到 {data_path}")
#
#     print(f"Loading dataset from {data_path}")
#     data_np = np.load(data_path)   # shape: [T, P]
#
#     print("Original data shape:", data_np.shape)
#
#     # ----------- 补 NaN ----------
#     mask = np.isnan(data_np)
#     data = np.ma.masked_array(data_np, mask)
#     data_interp = pd.DataFrame(data).interpolate().values
#     data_np = np.nan_to_num(data_interp)
#
#     # ----------- 标准化 ----------
#     if scaler_type == "minmax":
#         scaler = MinMaxScaler()
#     else:
#         scaler = StandardScaler()
#
#     T, P = data_np.shape
#     data_np = scaler.fit_transform(data_np.reshape(-1, 1)).reshape(T, P)
#
#     X = data_np.astype(np.float32)
#
#     # ----------- GC ----------
#     if os.path.exists(graph_path):
#         graph = np.load(graph_path).astype(np.float32)
#         graph = graph[:P, :P]   # 对齐维度
#
#         GC = graph if use_graph_as_gc else np.zeros((P, P), dtype=np.float32)
#     else:
#         GC = np.zeros((P, P), dtype=np.float32)
#
#     return GC, X


import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def read_real_dataset(data_path, dataset_type, n_max=None, scaler_type='minmax'):
    """
    读取真实数据集 (PM2.5, Traffic, Finance, Medical)
    并复用 load_data 的预处理逻辑，返回 (GC, X) 供 Granger 模型使用。

    Args:
        data_path (str): 数据集根目录 (例如 './dataset/traffic/')
        dataset_type (str): 'pm2.5', 'traffic', 'finance', 'medical'
        n_max (int): 限制最大变量数 P (用于 Traffic/Finance)
        scaler_type (str): 'minmax' 或 'standard'

    Returns:
        GC (np.ndarray): (P, P) 形状的 Ground Truth 矩阵
        X (np.ndarray): (T, P) 形状的时间序列数据
    """

    data_file = os.path.join(data_path, 'gen_data.npy')
    graph_file = os.path.join(data_path, 'graph.npy')

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    # === 1. 加载 GC (Graph/Mask) ===
    # 根据你的代码逻辑，Finance 和 Medical 需要 allow_pickle=True
    if dataset_type in ['finance', 'medical']:
        mask = np.load(graph_file, allow_pickle=True)
    else:
        mask = np.load(graph_file)

    # === 2. 加载 Data (X) 并根据类型处理 ===
    data_raw = np.load(data_file, allow_pickle=True)

    # 针对不同数据集的特殊截取逻辑
    if dataset_type == 'traffic':
        # 对应你代码中: if n_max < data_np.shape[1]: ...
        if n_max is not None and n_max < data_raw.shape[1]:
            data_raw = data_raw[:, :n_max]
        # Mask 也要截取
        if n_max is not None and n_max < mask.shape[0]:
            mask = mask[:n_max, :n_max]

    elif dataset_type == 'finance':
        # 对应你代码中: data_np = data_np[:,:n_max]
        if n_max is not None and n_max < data_raw.shape[1]:
            data_raw = data_raw[:, :n_max]
        # Finance 的 mask 可能是稀疏矩阵或 list，视情况可能需要截取，这里假设它是 (P,P) 矩阵
        if isinstance(mask, np.ndarray) and mask.ndim == 2:
            if n_max is not None and n_max < mask.shape[0]:
                mask = mask[:n_max, :n_max]

    elif dataset_type == 'medical':
        # === Medical 特殊处理 ===
        # 对应 load_medical_data: 你的代码截取了前 200 个病人
        limit = 200
        if len(data_raw) > limit:
            data_raw = data_raw[:limit]

        # Medical 数据是 List[Array(T_i, P)]，Granger 模型通常需要单个长序列 (Total_T, P)
        # 我们对每个病人单独做插值预处理，然后拼接
        processed_list = []
        for subject in data_raw:
            # 单个病人数据是 (T, P) 或 (P, T)? 通常是 (T, P)
            # 调用下面的通用预处理 helper，但不做缩放(最后统一缩放)
            sub_clean = _preprocess_helper(np.array(subject), scaler_type=None)
            processed_list.append(sub_clean)

        # 拼接所有病人数据
        data_raw = np.concatenate(processed_list, axis=0)

    # === 3. 通用预处理 (插值 + 归一化) ===
    # 除了 Medical 已经在循环里做了插值，其他数据需要在这里处理
    # 为了统一逻辑，我们在 _preprocess_helper 里判断
    if dataset_type != 'medical':
        X = _preprocess_helper(data_raw, scaler_type=scaler_type)
    else:
        # Medical 已经做了插值和拼接，现在做统一缩放
        X = _preprocess_helper(data_raw, scaler_type=scaler_type, skip_interp=True)

    # === 4. 格式修正 ===
    # 确保 mask 对角线为 0 (Granger Causality 通常不考虑自回归作为边)
    GC = np.array(mask, dtype=float)
    np.fill_diagonal(GC, 0)

    return GC, X


def _preprocess_helper(data_ori, scaler_type='minmax', skip_interp=False):
    """
    完全复刻你提供的 load_data 中的核心清洗逻辑
    """
    original_shape = data_ori.shape

    # 1. 插值处理 (Interpolate)
    if not skip_interp:
        mask = np.isnan(data_ori)
        # 如果全是数值，这一步不会改变数据；如果有NaN，masked_array会标记
        data_masked = np.ma.masked_array(data_ori, mask)
        # 使用 pandas 插值
        try:
            # 你的原始代码: data_interp = pd.DataFrame(data).interpolate().values
            # 注意：如果某列全空，interpolate 可能还是 NaN，最后用 nan_to_num
            data_interp = pd.DataFrame(data_masked).interpolate(limit_direction='both').values
            data_ori = np.nan_to_num(data_interp)
        except:
            # 容错：如果数据维度有问题导致 DataFrame 失败，直接 nan_to_num
            data_ori = np.nan_to_num(data_ori)

    # 2. 归一化 (Scaling)
    if scaler_type is not None:
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        # 你的原始逻辑：scaler.fit_transform(data_ori.reshape(-1, 1)).squeeze()
        # 这是一种全局缩放（Global Scaling），保留了变量间的相对大小
        data_ori = scaler.fit_transform(data_ori.reshape(-1, 1)).squeeze()
        data_ori = data_ori.reshape(original_shape)

    return data_ori

def off_diagonal(mat):
    """ 清除对角线 """
    mat = mat.copy()
    np.fill_diagonal(mat, 0)
    return mat

def compute_edge_features_1205(GCs, GC2s, outs_fwd, outs_rev,
                          target_seq, reversed_target_seq,
                          models, losses_fwd, losses_rev):
    """
    输入:
        GCs, GC2s: (P,P) forward/backward 的 GC 矩阵
        outs_fwd, outs_rev: list of (T-1,) 预测
        target_seq, reversed_target_seq: (T-1,P) 序列
        models: ModuleList of 2P models (未被使用，但保留签名)
        losses_fwd, losses_rev: list of P loss tensors (per-target loss)
    输出:
        feat_edges: (P*P, feat_dim=8)
    """
    # P = GCs.shape[0]
    # device = GCs.device
    # epsilon = 1e-12
    # # 将损失列表转换为张量 (P,)
    # losses_fwd_tensor = torch.stack(losses_fwd).to(device)
    # losses_rev_tensor = torch.stack(losses_rev).to(device)
    #
    # # --- 基础分数 ---
    # g_fwd = GCs.view(-1)
    # g_rev = GC2s.view(-1)
    # absdiff = torch.abs(g_fwd - g_rev)
    #
    # # --- 残差方差比 ---
    # res_var_fwd = torch.stack([
    #     torch.var((outs_fwd[j] - target_seq[:, j]).detach())
    #     for j in range(P)
    # ]).to(device)
    # res_var_rev = torch.stack([
    #     torch.var((outs_rev[j] - reversed_target_seq[:, j]).detach())
    #     for j in range(P)
    # ]).to(device)
    # # res_ratio: (P,) -> (P, 1) -> (P, P) -> (P*P,)
    # res_ratio = (res_var_fwd / (res_var_rev + 1e-12)).unsqueeze(1).repeat(1, P).view(-1)
    #
    # # --- in/out degree ---
    # in_degree = GCs.sum(0).view(-1)              # (P,)
    # out_degree = GCs.sum(1).view(-1).repeat(P)   # (P*P,)
    # in_degree_rep = in_degree.unsqueeze(0).repeat(P, 1).view(-1) # (P*P,)
    #
    # # --- 新增：基于预测损失的特征 ---
    # # 1. 目标 j 的正向损失 (反映拟合程度，用于所有指向 j 的边)
    # # target_loss_fwd: (P,) -> (P, 1) -> (P, P) -> (P*P,)
    # target_loss_fwd = losses_fwd_tensor.unsqueeze(1).repeat(1, P).view(-1)
    #
    # # 2. 目标 j 的损失比率 (反映正反模型对 j 的相对优势)
    # # loss_ratio: (P,) -> (P, 1) -> (P, P) -> (P*P,)
    # loss_ratio = (losses_fwd_tensor / (losses_rev_tensor + 1e-12)).unsqueeze(1).repeat(1, P).view(-1)
    #
    # # --- 拼接 (feat_dim = 8) ---
    # feat_edges = torch.stack([
    #     g_fwd.detach(),
    #     g_rev.detach(),
    #     absdiff.detach(),
    #     res_ratio.detach(),
    #     in_degree_rep.detach(),
    #     out_degree.detach(),
    #     target_loss_fwd.detach(),    # 新增特征 1
    #     loss_ratio.detach()          # 新增特征 2
    # ], dim=1)
    #
    # # --- 标准化 ---
    # # 对所有特征应用 log1p 变换 (处理值域较大或偏态分布的特征)
    # feat_edges = torch.log1p(torch.abs(feat_edges))
    #
    # # 标准化 (Z-score)
    # feat_mean = feat_edges.mean(dim=0, keepdim=True)
    # feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
    # feat_edges = (feat_edges - feat_mean) / feat_std
    #
    # return feat_edges
    """
        【优化版】融合特征提取：加入了拓扑归一化GC和对称对数损失比。
        feat_dim = 10
        """
    P = GCs.shape[0]
    device = GCs.device
    epsilon = 1e-12

    # 将损失列表转换为张量 (P,)
    losses_fwd_tensor = torch.stack(losses_fwd).to(device)
    losses_rev_tensor = torch.stack(losses_rev).to(device)

    # --- 基础分数 ---
    g_fwd = GCs.view(-1)
    g_rev = GC2s.view(-1)
    absdiff = torch.abs(g_fwd - g_rev)

    # --- 残差方差比 ---
    res_var_fwd = torch.stack([
        torch.var((outs_fwd[j] - target_seq[:, j]).detach())
        for j in range(P)
    ]).to(device)
    res_var_rev = torch.stack([
        torch.var((outs_rev[j] - reversed_target_seq[:, j]).detach())
        for j in range(P)
    ]).to(device)
    res_ratio = (res_var_fwd / (res_var_rev + epsilon)).unsqueeze(1).repeat(1, P).view(-1)

    # --- 拓扑和度特征 ---
    in_degree = GCs.sum(0).view(-1)  # (P,)
    out_degree = GCs.sum(1).view(-1).repeat(P)  # (P*P,)
    in_degree_rep = in_degree.unsqueeze(0).repeat(P, 1).view(-1)  # (P*P,)

    # --- 新增：GC 矩阵的拓扑归一化特征 (优化 1) ---
    # Col Norm: 目标 j 预测难度归一化 (对 j 的总影响)
    col_sum = GCs.sum(dim=1, keepdim=True)  # (P, 1)
    GC_col_norm = GCs / (col_sum + epsilon)  # (P, P)
    gc_col_norm_flat = GC_col_norm.view(-1)  # (P*P,)

    # Row Norm: 原因 i 影响强度归一化 (i 的总输出影响)
    row_sum = GCs.sum(dim=0, keepdim=True)  # (1, P)
    GC_row_norm = GCs / (row_sum + epsilon)  # (P, P)
    gc_row_norm_flat = GC_row_norm.view(-1)  # (P*P,)

    # --- 新增：基于预测损失的特征 ---
    # 1. 目标 j 的正向损失 (反映拟合程度，用于所有指向 j 的边)
    target_loss_fwd = losses_fwd_tensor.unsqueeze(1).repeat(1, P).view(-1)

    # 2. 对称对数损失比 (优化 2.1)
    sym_log_loss_ratio = torch.log((losses_fwd_tensor + epsilon) / (losses_rev_tensor + epsilon))
    sym_log_loss_ratio_rep = sym_log_loss_ratio.unsqueeze(1).repeat(1, P).view(-1)  # (P*P,)

    # --- 拼接 (feat_dim = 10) ---
    feat_list = [
        g_fwd.detach(),
        g_rev.detach(),
        absdiff.detach(),
        res_ratio.detach(),
        in_degree_rep.detach(),
        out_degree.detach(),
        gc_col_norm_flat.detach(),  # 新增特征 7
        gc_row_norm_flat.detach(),  # 新增特征 8
        target_loss_fwd.detach(),
        sym_log_loss_ratio_rep.detach()  # 替换了 loss_ratio
    ]
    feat_edges = torch.stack(feat_list, dim=1)

    # --- 标准化 ---
    # 对除对称对数损失比之外的特征应用 log1p(|x|) 变换
    # 对称对数损失比已经是 log 变换，并且值域对称，直接 Z-score 即可。
    feat_edges[:, :-1] = torch.log1p(torch.abs(feat_edges[:, :-1]))

    # 标准化 (Z-score)
    feat_mean = feat_edges.mean(dim=0, keepdim=True)
    feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
    feat_edges = (feat_edges - feat_mean) / feat_std

    return feat_edges
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



def compute_gradient_gc_smooth_universal(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                         smooth=True):
    """
    通用 GC 计算：计算原始梯度绝对值并进行平滑，但不进行裁剪。
    裁剪和聚合在外部处理。
    返回: grads_abs (1, T, P)
    """
    device_local = input_seq_local.device
    P_local = input_seq_local.shape[2]

    inp = input_seq_local.detach().clone().requires_grad_(True)
    outs = model(inp).squeeze(0)  # (T, P)

    # Note: 循环内仍然需要进行 P 次 backward() 调用
    # 我们将聚合留到外部进行，这里只计算所有梯度，但为了节省内存和计算，我们仍按行 j 计算并聚合
    GCs_local = torch.zeros((P_local, P_local), device=device_local)

    for j in range(P_local):
        out_j = outs[:, j]
        s_j = out_j.sum()

        grads = torch.autograd.grad(
            s_j, inp,
            create_graph=create_graph,
            retain_graph=True
        )[0]  # (1, T, P)

        grads_abs = grads.abs()  # (1, T, P)

        # --- temporal smoothing (3-point moving average) ---
        if smooth:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device_local).view(1, 1, 3)
            # reshape to (batch=1, channels=P, T)
            g = grads_abs.permute(0, 2, 1)  # (1, P, T)
            g = torch.nn.functional.pad(g, (1, 1), mode='reflect')  # pad time dim
            g = torch.nn.functional.conv1d(g, kernel.repeat(P_local, 1, 1), groups=P_local)
            grads_abs = g.permute(0, 2, 1)  # back to (1, T, P)

        # --- 外部裁剪和聚合: 在这里只进行聚合 (与原始代码保持一致) ---
        if lag_agg_local == 'mean':
            gc_row = grads_abs.mean(dim=1).squeeze(0)
        elif lag_agg_local == 'max':
            gc_row = grads_abs.max(dim=1)[0].squeeze(0)
        else:
            gc_row = grads_abs.mean(dim=1).squeeze(0)

        GCs_local[j, :] = gc_row

    return GCs_local
def infer_Grangercausalityv4_inge_plus_try_tosparse_1208(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False):
    """
    说明:
    - P: 保留的参数（变量个数），但如果与数据实际列数不一致，会打印提示并以实际列数为准构建模型。
    - type: 子数据集名称，字符串 "AQI" / "Traffic" / "Medical"
    - 其它超参见调用处
    返回:
    - best_score, best_auprc
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import pandas as pd

    # ---------------- reproducibility ----------------
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)



    # device_local = globals().get('device', torch.device('cpu'))
    target_device_id = 1
    if torch.cuda.is_available() and torch.cuda.device_count() > target_device_id:
        device_local = torch.device(f'cuda:{target_device_id}')
        print(f"Using device: {device_local}")
    else:
        device_local = torch.device('cpu')
        print(f"CUDA device {target_device_id} not available. Falling back to CPU.")
    # ---------------- Load data (CausalTime -> match DREAM4 format) ----------------
    data_root = f"/home/user/wcj/KANGCI-main/realdataset/{type}/"
    # reuse your load_causaltime_seq function (must be available in scope)
    try:
        input_seq, target_seq, reversed_input_seq, reversed_target_seq, mask, X_raw = \
            load_causaltime_seq(data_root,  device_local, scaler_type='minmax', n_max=P)
    except Exception as e:
        raise RuntimeError(f"call to load_causaltime_seq failed: {e}")

    # X_raw: (T, P_loaded)
    T_loaded, P_loaded = X_raw.shape

    if P != P_loaded:
        print(f"Warning: provided P={P} differs from data's variable count P_loaded={P_loaded}. "
              f"Using P_loaded={P_loaded} for model construction (keeps P parameter in signature).")
        P_effective = P_loaded
    else:
        P_effective = P

    # Convert mask (graph) to adjacency matrix (float) if needed
    if isinstance(mask, np.ndarray):
        GC = mask.astype(float)
    else:
        try:
            GC = np.array(mask, dtype=float)
        except:
            # fallback: identity (no edges) if cannot parse
            GC = np.eye(P_effective, dtype=float)
    # Ensure GC shape matches P_effective
    if GC.shape[0] != P_effective or GC.shape[1] != P_effective:
        # try to crop or pad
        crop_n = min(GC.shape[0], P_effective)
        GC2 = np.zeros((P_effective, P_effective), dtype=float)
        GC2[:crop_n, :crop_n] = GC[:crop_n, :crop_n]
        GC = GC2

    # ---------------- build models ----------------
    # NOTE: KAN, FusionEdge are assumed defined in your code base.
    model_fwd = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device_local)
    model_rev = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device_local)
    fusion_edge = FusionEdge(in_dim=10, hidden=32).to(device_local)  # adjust in_dim if needed

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate*0.01}
    ])

    loss_fn = nn.MSELoss()

    # optionally compute normalization stats for input normalization
    mean_cols = None
    std_cols = None
    if normalize_input:
        X_np = X_raw.astype(np.float32)
        mean_cols = torch.tensor(X_np.mean(axis=0), dtype=torch.float32).to(device_local)
        std_cols = torch.tensor(X_np.std(axis=0) + 1e-8, dtype=torch.float32).to(device_local)

    # ---------------- GC computation helpers (gradient-based) ----------------
    def compute_gradient_gc_smooth(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                   smooth=True, grad_clip=3.0):
        device_l = input_seq_local.device
        P_l = input_seq_local.shape[2]

        inp = input_seq_local.detach().clone().requires_grad_(True)
        outs = model(inp).squeeze(0)  # (T, P)

        GCs_local = torch.zeros((P_l, P_l), device=device_l)

        for j in range(P_l):
            out_j = outs[:, j]
            s_j = out_j.sum()

            grads = torch.autograd.grad(
                s_j, inp,
                create_graph=create_graph,
                retain_graph=True
            )[0]  # (1, T, P)

            grads_abs = grads.abs()  # (1, T, P)

            if smooth:
                # 3-point moving average along time axis
                kernel = torch.tensor([0.25, 0.5, 0.25], device=device_l).view(1, 1, 3)
                g = grads_abs.permute(0, 2, 1)  # (1, P, T)
                g = F.pad(g, (1, 1), mode='reflect')
                g = F.conv1d(g, kernel.repeat(P_l, 1, 1), groups=P_l)
                grads_abs = g.permute(0, 2, 1)

            if grad_clip is not None:
                grads_abs = grads_abs.clamp(max=grad_clip)

            if lag_agg_local == 'mean':
                gc_row = grads_abs.mean(dim=1).squeeze(0)
            elif lag_agg_local == 'max':
                gc_row = grads_abs.max(dim=1)[0].squeeze(0)
            else:
                gc_row = grads_abs.mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local

    # ---------------- training loop ----------------
    # 🌟 新增：分块处理长序列的 GC 计算 (解决 OOM)
    def compute_batched_gc(model, full_input_seq, chunk_size, create_graph, lag_agg_local, smooth_flag,
                           grad_clip_val):
        T = full_input_seq.shape[1]
        P = full_input_seq.shape[2]
        device = full_input_seq.device

        accumulated_GCs = torch.zeros((P, P), device=device, requires_grad=create_graph)

        num_chunks = (T + chunk_size - 1) // chunk_size  # 向上取整的批次数

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            input_chunk = full_input_seq[:, start:end, :]  # (1, T_chunk, P)

            # 对每个小块调用 compute_gradient_gc_smooth
            GCs_chunk = compute_gradient_gc_smooth(
                model,
                input_chunk,
                create_graph=create_graph,
                lag_agg_local=lag_agg_local,
                smooth=smooth_flag,
                grad_clip=grad_clip_val
            )

            accumulated_GCs = accumulated_GCs + GCs_chunk
            # 关键：每次计算后，释放内存
            del GCs_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return accumulated_GCs / num_chunks
    best_score = -1e9
    best_auprc = -1e9
    best_GCs = None
    best_GC2s = None
    best_fused = None
    # 🌟 关键参数：定义分块大小 (GC_CHUNK_SIZE)
    GC_CHUNK_SIZE = 128  # 推荐值，如果 OOM 持续，请尝试 1024 或更小  128
    # 从 compute_gradient_gc_smooth 的定义中提取参数
    smooth_flag = True
    grad_clip_val = 10.0
    # ensure tensors on device
    # ------------------- 早停参数 -------------------
    patience = 8  # 容忍多少轮没有提升
    min_delta = 1e-5  # 最小提升幅度
    counter = 0  # 计数器
    previous_auroc = -1e9  # 记录上一轮的 AUROC
    # ------------------------------------------------
    input_seq = input_seq.to(device_local)
    target_seq = target_seq.to(device_local)
    reversed_input_seq = reversed_input_seq.to(device_local)
    reversed_target_seq = reversed_target_seq.to(device_local)

    # if normalization requested, apply (use same mean/std for both forward/backward)
    if normalize_input and (mean_cols is not None and std_cols is not None):
        inp_fwd_base = (input_seq - mean_cols.to(device_local)) / std_cols.to(device_local)
        inp_rev_base = (reversed_input_seq - mean_cols.to(device_local)) / std_cols.to(device_local)
    else:
        inp_fwd_base = input_seq
        inp_rev_base = reversed_input_seq

    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        inp_fwd = inp_fwd_base
        inp_rev = inp_rev_base

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P_effective)
        outs_rev_all = model_rev(inp_rev).squeeze(0)

        # per-target MSE losses
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P_effective)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P_effective)]

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        loss_t_fwd = (outs_fwd_all - target_seq) ** 2
        loss_t_rev = (outs_rev_all - reversed_target_seq) ** 2

        W_fwd_normalized = loss_t_fwd / (loss_t_fwd.sum(dim=0, keepdim=True) + 1e-8)
        W_rev_normalized = loss_t_rev / (loss_t_rev.sum(dim=0, keepdim=True) + 1e-8)

        # # compute GC matrices
        # if (i % gc_every) == 0:
        #     GCs = compute_gradient_gc_smooth(model_fwd, input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
        #     GC2s = compute_gradient_gc_smooth(model_rev, reversed_input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
        # else:
        #     with torch.no_grad():
        #         GCs = compute_gradient_gc_smooth(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
        #         GC2s = compute_gradient_gc_smooth(model_rev, reversed_input_seq, create_graph=False, lag_agg_local=lag_agg)

        # compute GC matrices
        if (i % gc_every) == 0:
            # 🌟 替换：使用分块计算 GCs
            GCs = compute_batched_gc(model_fwd, input_seq, GC_CHUNK_SIZE, gc_create_graph, lag_agg, smooth_flag,
                                     grad_clip_val)
            GC2s = compute_batched_gc(model_rev, reversed_input_seq, GC_CHUNK_SIZE, gc_create_graph, lag_agg,
                                      smooth_flag, grad_clip_val)
        else:
            with torch.no_grad():
                # 🌟 替换：使用分块计算 GCs (无梯度)
                GCs = compute_batched_gc(model_fwd, input_seq, GC_CHUNK_SIZE, False, lag_agg, smooth_flag,
                                         grad_clip_val)
                GC2s = compute_batched_gc(model_rev, reversed_input_seq, GC_CHUNK_SIZE, False, lag_agg, smooth_flag,
                                          grad_clip_val)

        # prepare per-target outputs for edge feature computation
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P_effective)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P_effective)]

        # compute edge features and fusion alphas (assume compute_edge_features_1205 available)
        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)  # expected shape (P*P,)
        alphas = alphas_flat.view(P_effective, P_effective)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # sparsity losses
        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse_base * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device_local)
            Lsparse2 = torch.tensor(0.0, device=device_local)
            Lsparse_fused = torch.tensor(0.0, device=device_local)

        loss = (predict_loss1 + predict_loss2 -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2 + Lsparse_fused)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # EMA for alphas (for stable evaluation)
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()

        # compute evaluation metrics (assume compute_roc, compute_auprc available)
        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, GCs_np, False)
            score2 = compute_roc(GC, GC2s_np, False)
            score_fusion = compute_roc(GC, fused_np, False)

            auprc1 = compute_auprc(GC, GCs_np, False)
            auprc2 = compute_auprc(GC, GC2s_np, False)
            auprc_fusion = compute_auprc(GC, fused_np, False)

            # update best by AUPRC fusion primarily, score secondarily
            if auprc_fusion > best_auprc:
                best_auprc = auprc_fusion
                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

            if score_fusion > best_score:
                best_score = score_fusion
        # ------------------- ❗ 早停逻辑 (新增) -------------------
        if score_fusion < previous_auroc - min_delta:
            counter += 1  # 连续下降，计数器加 1
        else:
            counter = 0  # 指标没有下降（持平或上升），重置计数器
        previous_auroc = score_fusion
        print(f"Epoch [{i + 1}/{epoch}] loss: {loss.item():.6f} predict_loss1: {predict_loss1.item():.6f} "
              f"predict_loss2: {predict_loss2.item():.6f} Lsparse_fwd: {Lsparse1.item():.6f} Lsparse2: {Lsparse2.item():.6f} Lsparse_fused: {Lsparse_fused.item():.6f} "
              f"score1: {score1:.4f} score2: {score2:.4f} score_fusion: {score_fusion:.4f} "
              f"AUPRC_fwd: {auprc1:.4f} AUPRC_rev: {auprc2:.4f} AUPRC_fusion: {auprc_fusion:.4f}")
        # ------------------- 终止条件 (新增) -------------------
        # 打印计数器，让用户知道何时接近早停
        if counter > 0:
            print(f"  --- No improvement counter: {counter} / {patience} ---")

        # ------------------- 终止条件 (新增) -------------------
        if counter >= patience:
            print(
                f"🌟🌟🌟 Early stopping triggered after {i + 1} epochs! AUPRC did not improve for {patience} rounds. 🌟🌟🌟")
            break  # 退出 for i in range(epoch) 循环


    # End training
    # optionally: return best matrices as well
    return best_score, best_auprc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# --- 辅助函数 1: 通用 GC 计算 (平滑 + 聚合，无裁剪) ---
def compute_gradient_gc_smooth_universal(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                         smooth=True):
    """
    通用 GC 计算：计算原始梯度绝对值并进行平滑，不进行裁剪。
    裁剪和动态 L1 权重在外部处理。
    返回: GCs_local (P, P)
    """
    device_local = input_seq_local.device
    P_local = input_seq_local.shape[2]

    inp = input_seq_local.detach().clone().requires_grad_(True)
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P_local, P_local), device=device_local)

    for j in range(P_local):
        out_j = outs[:, j]
        s_j = out_j.sum()

        grads = torch.autograd.grad(
            s_j, inp,
            create_graph=create_graph,
            retain_graph=True
        )[0]  # (1, T, P)

        grads_abs = grads.abs()  # (1, T, P)

        # --- temporal smoothing (3-point moving average) ---
        if smooth:
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device_local).view(1, 1, 3)
            g = grads_abs.permute(0, 2, 1)  # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            # 使用 groups=P_local 实现每个变量独立平滑
            g = F.conv1d(g, kernel.repeat(P_local, 1, 1), groups=P_local)
            grads_abs = g.permute(0, 2, 1)  # back to (1, T, P)

        # --- 外部裁剪和聚合: 在这里只进行聚合 (沿时间 T 维度的 mean/max) ---
        if lag_agg_local == 'mean':
            gc_row = grads_abs.mean(dim=1).squeeze(0)
        elif lag_agg_local == 'max':
            gc_row = grads_abs.max(dim=1)[0].squeeze(0)
        else:
            gc_row = grads_abs.mean(dim=1).squeeze(0)

        GCs_local[j, :] = gc_row

    return GCs_local

# --- 辅助函数 2: 分块 GC 计算 (处理长序列 OOM) ---
def compute_batched_gc(model, full_input_seq, chunk_size, create_graph, lag_agg_local, smooth_flag):
    """
    分块计算 GCs，并求平均。
    """
    T = full_input_seq.shape[1]
    P = full_input_seq.shape[2]
    device = full_input_seq.device

    # 积累 GCs 的总和 (如果 create_graph=True, accumulated_GCs 也需要梯度)
    accumulated_GCs = torch.zeros((P, P), device=device, requires_grad=create_graph)

    num_chunks = (T + chunk_size - 1) // chunk_size

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        input_chunk = full_input_seq[:, start:end, :]  # (1, T_chunk, P)

        # 调用通用 GC 函数
        GCs_chunk = compute_gradient_gc_smooth_universal(
            model,
            input_chunk,
            create_graph=create_graph,
            lag_agg_local=lag_agg_local,
            smooth=smooth_flag,
        )

        accumulated_GCs = accumulated_GCs + GCs_chunk
        del GCs_chunk
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return accumulated_GCs / num_chunks


import numpy as np
from sklearn import metrics


def compute_roc_new_1211(gc_label, gc_predict, include_self_causality):  # 注意: 参数名统一

    # 强制检查形状一致性，这是你遇到错误 [400, 1600] 的根本原因
    if gc_label.size != gc_predict.size:
        # 如果尺寸不匹配，先尝试修正为 P x P
        P = gc_predict.shape[0]  # 假设预测矩阵是正确的 P x P 尺寸
        if gc_label.size == P * P:
            gc_label = gc_label.reshape(P, P)
        else:
            # 如果实在不匹配，必须报错，或者强制裁切/填充 (但最好在数据加载处修正)
            raise ValueError(f"GC matrix size mismatch! Label size: {gc_label.size}, Predict size: {gc_predict.size}")

    # 1. 扁平化操作
    if include_self_causality:  # 您的需求：包含对角线 (P*P 元素)
        gc_label_flat = gc_label.flatten().astype(float)
        gc_predict_flat = gc_predict.flatten().astype(float)
    else:  # 排除对角线 (如果不需要自因果)
        P = gc_label.shape[0]
        off_diag_mask = np.flatnonzero(1 - np.eye(P))
        gc_label_flat = gc_label.ravel()[off_diag_mask].astype(float)
        gc_predict_flat = gc_predict.ravel()[off_diag_mask].astype(float)

    # 2. 检查正样本
    if np.sum(gc_label_flat) < 1e-8 or gc_predict_flat.size == 0:
        return 0.0

    # 3. 归一化和计算 AUC
    max_score = np.max(gc_predict_flat)
    normalized_predict = gc_predict_flat / max_score if max_score > 1e-8 else gc_predict_flat

    score = metrics.roc_auc_score(gc_label_flat, normalized_predict)
    return score

def infer_Grangercausalityv4_inge_plus_try_tosparse_1211(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False,
                                  # --- 新增/调整的超参数 ---
                                  lambda_confidence=0.01,  # 置信度损失的权重
                                  grad_clip_quantile=0.99,  # 动态裁剪的分位数
                                  GC_CHUNK_SIZE=128  # 分块计算大小
                                  ):
    """
    V1211 优化版本：
    - 集成动态梯度裁剪 (基于分位数)。
    - 集成基于预测损失的 Alpha 置信度正则化 (L_confidence)。
    - 集成预测损失加权的 L1 稀疏 (W L1 Sparsity)。
    - 采用分块计算 GC (compute_batched_gc) 避免 OOM。
    """
    # 假设所需的库和外部函数（load_causaltime_seq, KAN, FusionEdge,
    # compute_roc, compute_auprc, compute_edge_features_1205）已在环境中定义。

    # --- 初始化和设备设置 ---
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
        # Fallback for testing/simulated data
        # Fallback: 使用你的原始 DREAM4 数据加载逻辑
        GC, X = read_dream4(P, type)
        # GC = off_diagonal(GC)
        length = X.shape[0]

        test_x = X[:length - 1, :]
        test_y = X[1:length, :]
        input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
        target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)

        X2 = np.ascontiguousarray(X[::-1, :])
        reversed_x = X2[:length - 1, :]
        reversed_y = X2[1:length, :]

        reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
        reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)
        X_raw = X  # 用于计算 P

    P_effective = input_seq.shape[2]

    # --- 模型构建 ---
    model_fwd = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device)
    model_rev = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=10, hidden=32).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}
    ])

    loss_fn = nn.MSELoss()

    # Normalization (保持不变)
    mean_cols = None
    std_cols = None
    if normalize_input:
        X_np = X_raw.astype(np.float32)
        mean_cols = torch.tensor(X_np.mean(axis=0), dtype=torch.float32).to(device)
        std_cols = torch.tensor(X_np.std(axis=0) + 1e-8, dtype=torch.float32).to(device)
        inp_fwd_base = (input_seq - mean_cols.unsqueeze(0).unsqueeze(0)) / std_cols.unsqueeze(0).unsqueeze(0)
        inp_rev_base = (reversed_input_seq - mean_cols.unsqueeze(0).unsqueeze(0)) / std_cols.unsqueeze(0).unsqueeze(0)
    else:
        inp_fwd_base = input_seq
        inp_rev_base = reversed_input_seq

    # --- 训练循环 ---
    smooth_flag = True  # 默认启用平滑
    best_score = -1e9
    best_auprc = -1e9
    # ------------------- 早停参数 -------------------
    patience = 8  # 容忍多少轮没有提升
    min_delta = 1e-5  # 最小提升幅度
    counter = 0  # 计数器
    previous_auroc = -1e9  # 记录上一轮的 AUROC
    # ------------------------------------------------
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        inp_fwd = inp_fwd_base
        inp_rev = inp_rev_base

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P_effective)
        outs_rev_all = model_rev(inp_rev).squeeze(0)

        # Per-target MSE losses
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P_effective)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P_effective)]

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- GC 矩阵计算和动态裁剪 ---
        if (i % gc_every) == 0:
            # 1. 计算原始 GCs (使用分块)
            GCs_raw = compute_batched_gc(model_fwd, input_seq, GC_CHUNK_SIZE, gc_create_graph, lag_agg, smooth_flag)
            GC2s_raw = compute_batched_gc(model_rev, reversed_input_seq, GC_CHUNK_SIZE, gc_create_graph, lag_agg,
                                          smooth_flag)

            # 2. 动态裁剪 (只在创建图时进行)
            if gc_create_graph:
                with torch.no_grad():
                    combined_gc_values = torch.cat([GCs_raw.view(-1), GC2s_raw.view(-1)])
                    nonzero_combined = combined_gc_values[combined_gc_values > 1e-6]
                    if nonzero_combined.numel() > 0:
                        clip_threshold = torch.quantile(nonzero_combined, grad_clip_quantile)
                    else:
                        clip_threshold = 1.0

                GCs = GCs_raw.clamp(max=clip_threshold)
                GC2s = GC2s_raw.clamp(max=clip_threshold)
            else:
                GCs = GCs_raw
                GC2s = GC2s_raw

        else:
            with torch.no_grad():
                GCs = compute_batched_gc(model_fwd, input_seq, GC_CHUNK_SIZE, False, lag_agg, smooth_flag)
                GC2s = compute_batched_gc(model_rev, reversed_input_seq, GC_CHUNK_SIZE, False, lag_agg, smooth_flag)

        # --- Edge Features & Fusion ---
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P_effective)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P_effective)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P_effective, P_effective)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # --- 损失计算 ---

        # 1. L_confidence (置信度损失)
        L_confidence = lambda_confidence * (
                (1.0 - alphas) * predict_loss1 + alphas * predict_loss2
        ).mean()

        # 2. 加权稀疏损失 (W L1 Sparsity)
        losses_fwd_tensor = torch.stack(losses_fwd)
        losses_rev_tensor = torch.stack(losses_rev)

        loss_norm_fwd = losses_fwd_tensor / (predict_loss1.detach() + eps)
        loss_norm_rev = losses_rev_tensor / (predict_loss2.detach() + eps)

        W_fwd_weighted = loss_norm_fwd.unsqueeze(1).repeat(1, P_effective)
        W_rev_weighted = loss_norm_rev.unsqueeze(1).repeat(1, P_effective)

        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.sum(W_fwd_weighted * torch.abs(GCs))
            Lsparse2 = lambda_gc_sparse_base * torch.sum(W_rev_weighted * torch.abs(GC2s))
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device_local)
            Lsparse2 = torch.tensor(0.0, device=device_local)
            Lsparse_fused = torch.tensor(0.0, device=device_local)

        # 3. 组合总损失
        loss = (predict_loss1 + predict_loss2
                - lambda_alpha_reg * alpha_reg
                + Lsparse1 + Lsparse2 + Lsparse_fused
                + L_confidence)  # ❗ 新增置信度损失项

        # --- 反向传播 ---
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # --- 评估和记录 (保持不变) ---
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        # compute evaluation metrics (assume compute_roc, compute_auprc available)
        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc_new_1211(GC, GCs_np, True)
            score2 = compute_roc_new_1211(GC, GC2s_np, True)
            score_fusion = compute_roc_new_1211(GC, fused_np, True)

            auprc1 = compute_auprc(GC, GCs_np, False)
            auprc2 = compute_auprc(GC, GC2s_np, False)
            auprc_fusion = compute_auprc(GC, fused_np, False)

            # update best by AUPRC fusion primarily, score secondarily
            if auprc_fusion > best_auprc:
                best_auprc = auprc_fusion
                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

            if score_fusion > best_score:
                best_score = score_fusion

        if score_fusion < previous_auroc - min_delta:
            counter += 1  # 连续下降，计数器加 1
        else:
            counter = 0  # 指标没有下降（持平或上升），重置计数器
        previous_auroc = score_fusion

            # ------------------- 终止条件 (新增) -------------------
        # if counter >= patience:
        #     print(f"Early stopping triggered after {i + 1} epochs. AUPRC did not improve for {patience} rounds.")
        #     break  # 退出 for i in range(epoch) 循环
        # Print progress
        print(f"Epoch [{i + 1}/{epoch}] loss: {loss.item():.6f} L_conf: {L_confidence.item():.4f} predict_loss1: {predict_loss1.item():.6f} "
              f"predict_loss2: {predict_loss2.item():.6f} Lsparse_fwd: {Lsparse1.item():.6f} Lsparse2: {Lsparse2.item():.6f} Lsparse_fused: {Lsparse_fused.item():.6f} "
              f"score1: {score1:.4f} score2: {score2:.4f} score_fusion: {score_fusion:.4f} "
              f"AUPRC_fwd: {auprc1:.4f} AUPRC_rev: {auprc2:.4f} AUPRC_fusion: {auprc_fusion:.4f}")
        # print(
        #     f'Epoch [{i + 1}/{epoch}], L_pred1: {predict_loss1.item():.4f}, L_pred2: {predict_loss2.item():.4f}, '
        #     f'L_conf: {L_confidence.item():.4f}, L_sparse_fused: {Lsparse_fused.item():.4f}, '
        #     f'ROC_fwd: {score1:.4f}, ROC_rev: {score2:.4f}, ROC_fusion: {score_fusion:.4f},'
        #     f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        # )
        # 打印计数器，让用户知道何时接近早停
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
def frequency_domain_denoise(grads_abs, cutoff_ratio=0.2):
    """
    对梯度序列进行频域去噪
    grads_abs: (1, T, P) - 原始梯度绝对值
    cutoff_ratio: 保留低频分量的比例 (0.0 到 1.0)
    """
    # 1. 转换到频域 (针对时间轴 T)
    # rfft 产生 (1, P, T//2 + 1) 的复数张量
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)

    # 2. 创建低通滤波器掩码
    n_freq = g_fft.shape[-1]
    cutoff = max(1, int(n_freq * cutoff_ratio))

    mask = torch.zeros_like(g_fft)
    mask[..., :cutoff] = 1.0  # 只保留低频

    # 3. 滤波并转换回时域
    g_fft_filtered = g_fft * mask
    grads_denoised = torch.fft.irfft(g_fft_filtered, n=grads_abs.shape[1], dim=-1)

    # 4. 转回 (1, T, P) 并确保非负（因为是梯度绝对值的去噪）
    return grads_denoised.transpose(1, 2).abs()


def frequency_domain_learnable_filter(grads_abs, weight_vec):
    """
    grads_abs: (1, T, P)
    weight_vec: (T//2 + 1,) - 可学习的权重参数，初始化为高频衰减的形式
    """
    # 1. 变换到频域
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)  # (1, P, T//2+1)

    # 2. 软过滤：不是 0 或 1，而是乘以一个 0~1 之间的权重
    # 使用 sigmoid 确保权重在 0-1 之间
    soft_mask = torch.sigmoid(weight_vec)
    g_fft_filtered = g_fft * soft_mask

    # 3. 还原到时域
    grads_denoised = torch.fft.irfft(g_fft_filtered, n=grads_abs.shape[1], dim=-1)

    return grads_denoised.transpose(1, 2).abs()


def frequency_domain_soft_denoise(grads_abs, cutoff_ratio=0.2, temperature=0.05):
    """
    使用类似 Sigmoid 的函数实现平滑的低通滤波
    """
    T = grads_abs.shape[1]
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)
    n_freq = g_fft.shape[-1]

    # 创建平滑降落的频率坐标
    freq_indices = torch.linspace(0, 1, n_freq, device=grads_abs.device)

    # 使用 Sigmoid 构造平滑阈值界面
    # 当 freq > cutoff_ratio 时，权重迅速但平滑地降向 0
    mask = torch.sigmoid((cutoff_ratio - freq_indices) / temperature)

    g_fft_filtered = g_fft * mask
    grads_denoised = torch.fft.irfft(g_fft_filtered, n=T, dim=-1)

    return grads_denoised.transpose(1, 2).abs()


def frequency_domain_wiener_denoise(grads_abs, noise_floor=0.1):
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)

    # 计算功率谱
    power_spec = torch.abs(g_fft) ** 2

    # 维纳滤波收缩因子: H = S / (S + N)
    shrinkage = power_spec / (power_spec + noise_floor)

    g_fft_filtered = g_fft * shrinkage
    grads_denoised = torch.fft.irfft(g_fft_filtered, n=grads_abs.shape[1], dim=-1)

    return grads_denoised.transpose(1, 2).abs()


def frequency_domain_denoise_pro(grads_abs, cutoff_ratio=0.2, temperature=0.05):
    """
    改进版：平滑频域去噪
    temperature: 控制滤波边界的平滑度，越小越接近硬截断
    """
    T = grads_abs.shape[1]
    # 1. 变换
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)
    n_freq = g_fft.shape[-1]

    # 2. 构造平滑滤波器 (Soft Mask)
    # 计算频率轴坐标 (0 到 1)
    freq_indices = torch.linspace(0, 1, n_freq, device=grads_abs.device)

    # 使用 Sigmoid 构造平滑的低通 mask
    # 当 freq_indices > cutoff_ratio 时，权重平滑下降
    mask = torch.sigmoid((cutoff_ratio - freq_indices) / temperature)

    # 3. 滤波与还原
    g_fft_filtered = g_fft * mask
    grads_denoised = torch.fft.irfft(g_fft_filtered, n=T, dim=-1)

    return grads_denoised.transpose(1, 2).abs()
def compute_gradient_gc_smooth_universal_v3(model, input_seq_local, create_graph=True,
                                            lag_agg_local='mean', freq_denoise=False, cutoff_ratio=0.2):
    """
    集成频域去噪的 GC 计算函数 (修复版)
    修复了 retain_graph 导致的 backward 报错问题
    """
    # 1. 基础配置与准备
    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # 确保输入需要梯度
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # 前向传播
    outs = model(inp).squeeze(0)

    GCs_local = torch.zeros((P, P), device=device)

    # 2. 预先准备平滑算子
    if not freq_denoise:
        # 创建平滑卷积核
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3)
        kernel = kernel.repeat(P, 1, 1)

    # 3. 循环计算 GC
    for j in range(P):
        s_j = outs[:, j].sum()

        # --- 【关键修复】 ---
        # 如果 create_graph=True (训练中)，必须全程 retain_graph=True，
        # 否则函数外的 loss.backward() 会因为图被释放而报错。
        # 只有在 create_graph=False (纯推理/验证) 时，才允许在最后一步释放图。
        if create_graph:
            retain = True
        else:
            # 推理模式下，如果是最后一次循环，就不需要保留图了
            retain = (j < P - 1)

        # 计算梯度
        grads = torch.autograd.grad(
            s_j, inp,
            create_graph=create_graph,
            retain_graph=retain
        )[0]

        grads_abs = grads.abs()

        # --- 平滑处理 ---
        if freq_denoise:
            # 假设 frequency_domain_denoise 已定义
            grads_abs = frequency_domain_denoise(grads_abs, cutoff_ratio=cutoff_ratio)
        else:
            # 时域平滑 (Conv1d)
            g = grads_abs.permute(0, 2, 1)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)

        # --- 聚合 ---
        if lag_agg_local == 'max':
            gc_row = grads_abs.squeeze(0).max(dim=0)[0]
        else:
            gc_row = grads_abs.squeeze(0).mean(dim=0)

        GCs_local[j, :] = gc_row

        # 清理临时变量
        del grads, grads_abs

    return GCs_local
import torch
import torch.nn.functional as F

def frequency_domain_denoise(grads_abs, cutoff_ratio=0.2):
    """
    对梯度序列进行频域去噪（低通滤波）
    grads_abs: (1, T, P)
    """
    # (1, P, T//2 + 1)
    g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)

    n_freq = g_fft.shape[-1]
    cutoff = max(1, int(n_freq * cutoff_ratio))

    mask = torch.zeros_like(g_fft)
    mask[..., :cutoff] = 1.0

    g_fft_filtered = g_fft * mask
    grads_denoised = torch.fft.irfft(
        g_fft_filtered,
        n=grads_abs.shape[1],
        dim=-1
    )

    return grads_denoised.transpose(1, 2).abs()




# best
def compute_gradient_gc_smooth_universal_v3_1220(
    model,
    input_seq_local,
    create_graph=True,
    lag_agg_local='mean',
    freq_denoise=True,
    cutoff_ratio=0.2,
    lambda_gc=1.0,
    tau=0.1,
    h=0.05  # 新增：稀疏化阈值参数
):
    """
    Gradient-based GC with *in-graph* L1 regularization (second-order)

    返回：
        GCs_local : (P, P)  -- GC 矩阵（仅用于评估/可视化）
        L_gc      : scalar -- 直接用于 loss.backward() 的 GC-L1 正则项

    核心特性：
    - GC-L1 在梯度计算阶段立即加入（无 detach）
    - 支持时域 / 频域平滑
    - create_graph=True → 二阶梯度正则
    """

    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # ---- 输入显式 requires_grad ----
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 前向 ----
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # ---- 时域平滑卷积核 ----
    if not freq_denoise:
        kernel = torch.tensor(
            [0.25, 0.5, 0.25],
            device=device
        ).view(1, 1, 3)
        kernel = kernel.repeat(P, 1, 1)

    # ---- 逐输出变量计算 GC ----
    for j in range(P):
        # 聚合时间，保证梯度稳定
        s_j = outs[:, j].sum()

        # 关键：训练时必须保留完整计算图
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
            grads_abs = frequency_domain_denoise(
                grads_abs,
                cutoff_ratio=cutoff_ratio
            )
        else:
            # 时域 Conv1d 平滑
            g = grads_abs.permute(0, 2, 1)     # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)     # (1, T, P)

        # ---- lag 聚合 ----
        # if lag_agg_local == 'max':
        #     gc_row = grads_abs.squeeze(0).max(dim=0)[0]
        # elif lag_agg_local == 'mean':
        #     g = grads_abs.squeeze(0)  # (T, P)
        #     gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)
        #
        # else:
        #     gc_row = grads_abs.squeeze(0).mean(dim=0)

        # ---- lag 聚合 ----
        g = grads_abs.squeeze(0)  # (T, P)

        if lag_agg_local == 'mean':
            # 原始实现：时间平均
            gc_row = g.mean(dim=0)

        elif lag_agg_local == 'max':
            # 强瞬时因果（不太稳定，仅作对照）
            gc_row = g.max(dim=0)[0]

        elif lag_agg_local == 'rms':
            # 推荐：均方根，稳定且强调有效因果
            gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

        elif lag_agg_local == 'lp':
            # Lp pooling（p=2 等价于 RMS，可自行调 p）
            p = 2.0
            gc_row = (g.pow(p).mean(dim=0) + 1e-12).pow(1.0 / p)

        elif lag_agg_local == 'softmax':
            # 平滑 max（推荐做结构发现）
            tau = tau  # 0.1~0.5
            w = torch.softmax(g / tau, dim=0)
            gc_row = (w * g).sum(dim=0)

        elif lag_agg_local == 'quantile':
            # 抗噪聚合
            q = 0.9
            gc_row = g.quantile(q, dim=0)


        else:
            raise ValueError(f"Unknown lag_agg_local: {lag_agg_local}")

        # ======= ★ 关键一步：立即构造 L1（不 detach） ★ =======
        # bad
        # eps = 1e-6
        # gc_l1_loss = gc_l1_loss + torch.sqrt(gc_row ** 2 + eps).sum()



        gc_l1_loss = gc_l1_loss + gc_row.abs().sum()


        # 保存 GC（仅用于评估）
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc
def compute_gradient_gc_smooth_universal_v3_1220_v1(
    model,
    input_seq_local,
    create_graph=True,
    lag_agg_local='mean',
    freq_denoise=True,
    cutoff_ratio=0.2,
    lambda_gc=1.0,
        tau=0.1
):
    """
    Gradient-based GC with *in-graph* L1 regularization (second-order)

    返回：
        GCs_local : (P, P)  -- GC 矩阵（仅用于评估/可视化）
        L_gc      : scalar -- 直接用于 loss.backward() 的 GC-L1 正则项

    核心特性：
    - GC-L1 在梯度计算阶段立即加入（无 detach）
    - 支持时域 / 频域平滑
    - create_graph=True → 二阶梯度正则
    """

    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # ---- 输入显式 requires_grad ----
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 前向 ----
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # ---- 时域平滑卷积核 ----
    if not freq_denoise:
        kernel = torch.tensor(
            [0.25, 0.5, 0.25],
            device=device
        ).view(1, 1, 3)
        kernel = kernel.repeat(P, 1, 1)

    # ---- 逐输出变量计算 GC ----
    for j in range(P):
        # 聚合时间，保证梯度稳定
        s_j = outs[:, j].sum()

        # 关键：训练时必须保留完整计算图
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
            grads_abs = frequency_domain_denoise(
                grads_abs,
                cutoff_ratio=cutoff_ratio
            )
        else:
            # 时域 Conv1d 平滑
            g = grads_abs.permute(0, 2, 1)     # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)     # (1, T, P)

        # ---- lag 聚合 ----
        # if lag_agg_local == 'max':
        #     gc_row = grads_abs.squeeze(0).max(dim=0)[0]
        # elif lag_agg_local == 'mean':
        #     g = grads_abs.squeeze(0)  # (T, P)
        #     gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)
        #
        # else:
        #     gc_row = grads_abs.squeeze(0).mean(dim=0)

        # ---- lag 聚合 ----
        g = grads_abs.squeeze(0)  # (T, P)

        g_mean = g.mean(dim=0)
        g_std = g.std(dim=0) + 1e-6

        if lag_agg_local == 'mean':
            # 原始实现：时间平均
            gc_row = g.mean(dim=0)

        elif lag_agg_local == 'max':
            # 强瞬时因果（不太稳定，仅作对照）
            gc_row = g.max(dim=0)[0]

        elif lag_agg_local == 'rms':
            # 推荐：均方根，稳定且强调有效因果
            gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)

        elif lag_agg_local == 'lp':
            # Lp pooling（p=2 等价于 RMS，可自行调 p）
            p = 2.0
            gc_row = (g.pow(p).mean(dim=0) + 1e-12).pow(1.0 / p)

        elif lag_agg_local == 'softmax':
            # 平滑 max（推荐做结构发现）
            tau = tau  # 0.1~0.5
            w = torch.softmax(g / tau, dim=0)
            gc_row = (w * g).sum(dim=0)
            gc_row = gc_row / g_std.detach()


        elif lag_agg_local == 'quantile':
            # 抗噪聚合
            q = 0.9
            gc_row = g.quantile(q, dim=0)

        elif lag_agg_local == 'exp_weighted':
            # 时间指数加权（最近更重要）
            T = g.shape[0]
            weights = torch.exp(
                -torch.arange(T, device=g.device, dtype=g.dtype) / (0.3 * T)
            )
            weights = weights / weights.sum()
            gc_row = (g * weights[:, None]).sum(dim=0)

        else:
            raise ValueError(f"Unknown lag_agg_local: {lag_agg_local}")

        # ======= ★ 关键一步：立即构造 L1（不 detach） ★ =======
        # bad
        # eps = 1e-6
        # gc_l1_loss = gc_l1_loss + torch.sqrt(gc_row ** 2 + eps).sum()



        gc_l1_loss = gc_l1_loss + gc_row.abs().sum()


        # 保存 GC（仅用于评估）
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc


import torch
import torch.nn.functional as F


def sparsify_causality_graph(A, h=0.1):
    """
    根据论文公式进行因果图稀疏化
    公式:
    A_tilde[i,j] = max(0, A[i,j] - A[j,i]), i != j
    A_tilde[i,i] = A[i,i]

    参数:
        A: 原始因果矩阵 (P, P)
        h: 稀疏化阈值
    """
    # 1. 备份对角线元素 (A_i,i)
    diag_elements = torch.diag(A)

    # 2. 计算 A - A^T 并取 max(0, x)
    # 这步会自动处理 i != j 的情况，但会将对角线置为 0 (因为 A_ii - A_ii = 0)
    A_tilde = torch.relu(A - A.transpose(0, 1))

    # 3. 还原对角线元素
    # 使用 diagonal().copy_() 直接在原处修改或重新组合
    indices = torch.arange(A.size(0))
    A_tilde[indices, indices] = diag_elements

    # 4. 阈值过滤 (Sparsity threshold h)
    # 只有大于等于 h 的值才保留，其余置为 0
    A_tilde = torch.where(A_tilde >= h, A_tilde, torch.zeros_like(A_tilde))

    return A_tilde
# seem good yes
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
def compute_gradient_gc_smooth_universal_v3_1220_103v1_san(
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
def compute_gradient_gc_smooth_universal_v3_1220_103_v2(
    model,
    input_seq_local,
    spectral_gate,      # ★ 新增：可学习频域模块
    create_graph=True,
    lag_agg_local='mean',
    freq_denoise=True,
    cutoff_ratio=0.2,   # 保留参数但不再使用（兼容旧接口）
    lambda_gc=1.0,
    tau=0.1,
    h=0.05
):
    """
    Gradient-based GC with learnable frequency-domain smoothing
    """

    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]

    # ---- 输入显式 requires_grad ----
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 前向 ----
    outs = model(inp).squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # ---- 时域平滑卷积核（仅 freq_denoise=False 时使用）----
    if not freq_denoise:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device)
        kernel = kernel.view(1, 1, 3).repeat(P, 1, 1)

    # ---- 逐输出变量计算 GC ----
    for j in range(P):

        s_j = outs[:, j].sum()

        retain = True if create_graph else (j < P - 1)

        grads = torch.autograd.grad(
            s_j,
            inp,
            create_graph=create_graph,
            retain_graph=retain
        )[0]  # (1, T, P)

        grads_abs = grads.abs()

        # ======================================================
        # ★ 可学习频域分支
        # ======================================================
        if freq_denoise:

            gc_row = frequency_domain_denoise_learnable(
                grads_abs,
                spectral_gate=spectral_gate,
                n_fft=16,
                hop_length=8
            )

        # ======================================================
        # 原始时域分支（保持不变）
        # ======================================================
        else:
            g = grads_abs.permute(0, 2, 1)  # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            g = g.permute(0, 2, 1).squeeze(0)  # (T, P)

            if lag_agg_local == 'mean':
                gc_row = g.mean(dim=0)
            elif lag_agg_local == 'max':
                gc_row = g.max(dim=0)[0]
            elif lag_agg_local == 'rms':
                gc_row = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)
            elif lag_agg_local == 'softmax':
                w = torch.softmax(g / tau, dim=0)
                gc_row = (w * g).sum(dim=0)
            elif lag_agg_local == 'quantile':
                gc_row = g.quantile(0.9, dim=0)
            else:
                raise ValueError(f"Unknown lag_agg_local: {lag_agg_local}")

        # ---- L1 正则（in-graph）----
        gc_l1_loss = gc_l1_loss + gc_row.abs().sum()

        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss
    return GCs_local, L_gc
def compute_gradient_gc_smooth_universal_v3_1220_103v3(
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
    gc_l_loss = 0.0

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

        gc_l1_loss = gc_l1_loss + gc_row.abs().sum()


        # ==================================================
        # 一阶 GC prior（稳定，仅用于 refinement）
        # ==================================================
        if use_first_order_prior:
            gc_row_1st = torch.sqrt((g ** 2).mean(dim=0) + 1e-12)
            # L1 正则
            gc_l_loss = gc_l_loss + gc_row_1st.sum()
        else:
            gc_l_loss = gc_l_loss + gc_row.abs().sum()

        # ---- 保存二阶 GC（评估用）----
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss+ lambd_l * gc_l_loss

    return GCs_local, L_gc

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F


# 1. 高效 Ricci 曲率计算 (基于 Balanced Forman Curvature)
def calculate_ricci_curvature(A, include_4cycles=False):
    """
    使用矩阵运算加速计算 Ricci 曲率
    A: (P, P) 邻接矩阵 (Binary: 0 or 1)
    """
    P = A.shape[0]
    device = A.device

    # 计算度数 (out-degree)
    d = A.sum(dim=1)
    # 避免除以 0
    d_inv = 1.0 / torch.where(d > 0, d, torch.ones_like(d))

    # 计算三角形数量: (A^2 * A) 的对角线附近即为三角形
    # 这里计算每条边 (i,j) 参与的三角形数量
    triangles = torch.mm(A, A) * A

    # 初始化 Ricci 矩阵
    Ric = torch.zeros_like(A)

    # 提取存在的边索引
    rows, cols = torch.where(A > 0)
    if rows.numel() == 0:
        return Ric

    # 计算 Eq(7) 核心项
    di = d[rows]
    dj = d[cols]
    num_tri = triangles[rows, cols]

    # 基础项: 2/di + 2/dj - 2 + 2*#Δ/max(di,dj) + #Δ/min(di,dj)
    term_base = 2 * d_inv[rows] + 2 * d_inv[cols] - 2
    term_tri = (2 * num_tri / torch.max(di, dj)) + (num_tri / torch.min(di, dj))

    # 4-cycles (#□) 逻辑 (可选，通常为了速度设为0或简化)
    term_cycle = 0
    if include_4cycles:
        # 简化版：通过 A^2 的非对角元素估算路径
        A2 = torch.mm(A, A)
        term_cycle = (A2[rows, cols] - num_tri - 1).clamp(min=0) * 0.1  # 权重系数

    Ric[rows, cols] = term_base + term_tri + term_cycle
    return Ric


# 2. CSDRF 核心细化逻辑
def apply_csdrf_refinement(GCs_local, tau=5.0, adj_threshold=0.1, C_plus=1.0):
    """
    CSDRF: 基于因果信息指导的随机离散里奇流
    GCs_local: (P, P) 梯度因果强度
    """
    # 离散化与计算图脱钩，确保不影响二阶梯度回传
    with torch.no_grad():
        P = GCs_local.shape[0]
        device = GCs_local.device

        # 构建初始邻接表
        A = (GCs_local > adj_threshold).float()

        # A. 计算当前曲率
        Ric = calculate_ricci_curvature(A)

        # B. 寻找瓶颈 (Ricci 曲率最小的边)
        mask = (A > 0)
        if not mask.any(): return GCs_local

        min_ric = Ric[mask].min()
        bottlenecks = (Ric == min_ric).nonzero()
        # 随机选一个瓶颈边 (i, j)
        b_idx = torch.randint(0, bottlenecks.shape[0], (1,))
        i, j = bottlenecks[b_idx].squeeze()

        # C. 候选边评分 (k, l) where k ∈ B1(i), l ∈ B1(j)
        # 这里的逻辑是：在瓶颈两端寻找“捷径”以分散压力
        neighbors_i = A[i].nonzero().flatten()
        neighbors_j = A[j].nonzero().flatten()

        candidates = []
        causal_scores = []

        for k in neighbors_i:
            for l in neighbors_j:
                if k != l and A[k, l] == 0:
                    # 关键逻辑：改善得分 = 因果强度 * 预估提升(此处设为1)
                    # 对应论文中的 x_kl = (Ric_after - Ric_before) * T
                    score = GCs_local[k, l]
                    candidates.append((k, l))
                    causal_scores.append(score)

        # D. 执行采样更新
        refined_GCs = GCs_local.clone()

        if len(candidates) > 0:
            causal_scores = torch.stack(causal_scores)
            probs = F.softmax(causal_scores * tau, dim=0)
            c_idx = torch.multinomial(probs, 1)
            new_k, new_l = candidates[c_idx]

            # 添加新边：在 GCs 中增强该位置的值
            refined_GCs[new_k, new_l] = GCs_local.max() * 1.1

            # E. 移除冗余边 (Curvature Balancing)
            # 找到曲率最大的边（通常是高度冗余的三角形边）
            max_ric = Ric[mask].max()
            if max_ric > C_plus:
                redundants = (Ric == max_ric).nonzero()
                r_idx = torch.randint(0, redundants.shape[0], (1,))
                ri, rj = redundants[r_idx].squeeze()
                refined_GCs[ri, rj] = 0.0

        return refined_GCs


# 3. 集成后的主函数
def compute_gradient_gc_smooth_universal_v3_1220_1224(
        model,
        input_seq_local,
        create_graph=True,
        lag_agg_local='mean',
        freq_denoise=True,
        cutoff_ratio=0.2,
        lambda_gc=1.0,
        use_csdrf=True,  # 新增开关
        tau=10.0,  # 新增参数，用于 CSDRF 细化
        adj_threshold=0.05,
        C_plus=1.2  # 根据曲率平衡需求调整
):
    device = input_seq_local.device
    T, P = input_seq_local.shape[1], input_seq_local.shape[2]
    inp = input_seq_local.detach().clone().requires_grad_(True)

    outs = model(inp).squeeze(0)
    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # 预准备时域核
    if not freq_denoise:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3).repeat(P, 1, 1)

    for j in range(P):
        s_j = outs[:, j].sum()
        retain = True if create_graph else (j < P - 1)

        grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=retain)[0]
        grads_abs = grads.abs()

        # 平滑处理
        if freq_denoise:
            grads_abs = frequency_domain_denoise(grads_abs, cutoff_ratio=cutoff_ratio)
        else:
            g = grads_abs.permute(0, 2, 1)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)

        # 聚合
        gc_row = grads_abs.squeeze(0).max(dim=0)[0] if lag_agg_local == 'max' else grads_abs.squeeze(0).mean(dim=0)

        # L1 正则项（保持可导）
        gc_l1_loss += gc_row.abs().sum()
        GCs_local[j, :] = gc_row

    L_gc = lambda_gc * gc_l1_loss

    # --- CSDRF 细化步骤 ---
    if use_csdrf:
        # 注意：此处返回的 GCs_refined 带有离散重构信息
        GCs_final = apply_csdrf_refinement(
            GCs_local,
            tau=10.0,
            adj_threshold=0.05,
            C_plus=1.2  # 根据曲率平衡需求调整
        )
    else:
        GCs_final = GCs_local

    return GCs_final, L_gc




def compute_gradient_gc_smooth_universal_v3_1223(
        model,
        input_seq_local,
        create_graph=True,
        lag_agg_local='mean',
        freq_denoise=True,
        cutoff_ratio=0.2,
        lambda_gc=1.0,
        soft_slope=10.0
):
    """
    升级版：基于可微频域掩码的梯度因果度量 (GC)

    创新点：
    1. 采用 Sigmoid-based Soft Mask 替代硬截断，确保二阶梯度平滑。
    2. 频域处理直接嵌入计算图，支持端到端优化。
    3. 能够更精细地滤除梯度中的瞬时噪声，提取长程因果趋势。
    """
    device = input_seq_local.device
    # input_seq_local: (1, T, P)
    T = input_seq_local.shape[1]
    P = input_seq_local.shape[2]

    # ---- 1. 输入显式要求梯度 ----
    # 我们克隆输入并开启梯度，以便追踪 output 对 input 的敏感度
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # ---- 2. 模型前向推理 ----
    outs = model(inp)
    if outs.dim() > 2:
        outs = outs.squeeze(0)  # 变为 (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    gc_l1_loss = 0.0

    # 预计算时域卷积核（作为备选平滑方案）
    if not freq_denoise:
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3).repeat(P, 1, 1)

    # ---- 3. 逐变量计算因果强度 ----
    for j in range(P):
        # 聚合第 j 个变量在所有时间点的输出，计算对输入的总梯度
        s_j = outs[:, j].sum()

        # 计算梯度：d(outs_j) / d(input)
        # grads shape: (1, T, P)
        grads = torch.autograd.grad(
            s_j,
            inp,
            create_graph=create_graph,
            retain_graph=True  # 保持计算图以供后续变量 j+1 使用
        )[0]

        # 提取梯度幅值
        grads_abs = grads.abs()

        # ---- 4. 频域创新处理 (Differentiable Frequency Smoothing) ----
        if freq_denoise:
            # 执行实数快速傅里叶变换 (RFFT)
            # 变换维度为 T (时间轴)
            grads_fft = torch.fft.rfft(grads_abs, dim=1)
            n_freq = grads_fft.shape[1]

            # 构造平滑滤波器：Sigmoid Low-pass Filter
            # cutoff_ratio 定义了截断位置，soft_slope 定义了截断的陡峭程度
            freq_idx = torch.linspace(0, 1, n_freq, device=device).view(1, -1, 1)
            # mask 在 cutoff_ratio 之前接近 1，之后平滑降至 0
            mask = torch.sigmoid(soft_slope * (cutoff_ratio - freq_idx))

            # 应用掩码并逆变换回时域
            grads_fft_filtered = grads_fft * mask
            grads_abs = torch.fft.irfft(grads_fft_filtered, n=T, dim=1)
            # 确保由于数值计算带来的微小负数被消除
            grads_abs = torch.relu(grads_abs)
        else:
            # 传统时域平滑 (Conv1d)
            g = grads_abs.permute(0, 2, 1)  # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)  # (1, T, P)

        # ---- 5. 时间维度聚合 (Lag Aggregation) ----
        # 将 T 维度的梯度转化为一个标量因果值
        if lag_agg_local == 'max':
            # 提取每个输入通道在整个时间序列中的最大贡献
            gc_row = grads_abs.squeeze(0).max(dim=0)[0]
        else:
            # 提取平均贡献（更稳定）
            gc_row = grads_abs.squeeze(0).mean(dim=0)

        # ---- 6. 构造正则化项 (Causal Sparsity) ----
        # 直接累加而不 detach，使得 lambda_gc * L_gc 可以影响模型权重的更新
        gc_l1_loss = gc_l1_loss + gc_row.sum()

        # 记录 GC 矩阵（用于展示变量 i -> 变量 j 的强度）
        GCs_local[j, :] = gc_row

    # 最终的因果损失
    L_gc = lambda_gc * gc_l1_loss

    return GCs_local, L_gc


import torch
import torch.nn as nn
import torch.nn.functional as F


def get_smooth_spectral_mask(n_freq, cutoff_ratio, soft_slope, device):
    """预计算平滑的频域滤波器掩码"""
    freq_idx = torch.linspace(0, 1, n_freq, device=device)
    # 使用 Sigmoid 构造平滑的低通滤波器
    mask = torch.sigmoid(soft_slope * (cutoff_ratio - freq_idx))
    return mask.view(1, 1, -1)


def compute_gradient_gc_smooth_universal_v4(
        model,
        input_seq_local,
        create_graph=True,
        lag_agg_local='mean',
        freq_denoise=True,
        cutoff_ratio=0.2,
        soft_slope=15.0,  # 掩码平滑斜率
        noise_floor=0.05,  # 自适应噪声阈值
        lambda_gc=1.0
):
    """
    优化版：支持二阶梯度平滑、自适应频域掩码与向量化计算
    """
    device = input_seq_local.device
    B, T, P = input_seq_local.shape  # 假设 input_seq_local 为 (1, T, P)

    # 1. 显式开启输入梯度
    inp = input_seq_local.detach().clone().requires_grad_(True)

    # 2. 模型前向
    outs = model(inp)
    if outs.dim() > 2:
        outs = outs.squeeze(0)  # (T, P)

    GCs_local = torch.zeros((P, P), device=device)
    total_gc_l1 = 0.0

    # 3. 预准备平滑工具
    if freq_denoise:
        n_freq = T // 2 + 1
        # 预计算位置掩码 (Position-based Mask)
        pos_mask = get_smooth_spectral_mask(n_freq, cutoff_ratio, soft_slope, device)
    else:
        # 预准备时域卷积核
        kernel = torch.tensor([0.25, 0.5, 0.25], device=device).view(1, 1, 3).repeat(P, 1, 1)

    # 4. 逐输出变量计算梯度
    for j in range(P):
        # 聚合输出以求梯度
        s_j = outs[:, j].sum()

        # 计算梯度 grads: (1, T, P)
        grads = torch.autograd.grad(
            s_j,
            inp,
            create_graph=create_graph,
            retain_graph=True  # 必须保持图以支持后续变量 j 的计算
        )[0]

        # 提取幅值（保持导数流）
        grads_abs = grads.abs()

        if freq_denoise:
            # ---- 创新：向量化自适应频域去噪 ----
            # (1, P, T) -> (1, P, n_freq)
            g_fft = torch.fft.rfft(grads_abs.transpose(1, 2), dim=-1)
            mag = g_fft.abs()

            # (A) 位置掩码 (过滤高频)
            # (B) 能量掩码 (自适应去噪)：抑制低于平均能量的频率成分
            avg_mag = mag.mean(dim=-1, keepdim=True)
            energy_mask = torch.sigmoid(soft_slope * (mag - avg_mag * noise_floor))

            # 融合掩码
            final_mask = pos_mask * energy_mask

            g_fft_filtered = g_fft * final_mask
            # 还原回时域 (1, P, T)
            grads_denoised = torch.fft.irfft(g_fft_filtered, n=T, dim=-1)
            grads_abs = grads_denoised.transpose(1, 2).abs()
        else:
            # 时域 Conv1d 平滑
            g = grads_abs.permute(0, 2, 1)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel, groups=P)
            grads_abs = g.permute(0, 2, 1)

        # 5. Lag 聚合
        if lag_agg_local == 'max':
            gc_row = grads_abs.squeeze(0).max(dim=0)[0]
        else:
            gc_row = grads_abs.squeeze(0).mean(dim=0)

        # 6. 构造 L1 正则项 (直接累加带图张量)
        total_gc_l1 = total_gc_l1 + gc_row.sum()

        # 仅存入数值用于可视化 (detach 释放内存)
        GCs_local[j, :] = gc_row.detach()

    # 最终 Loss
    L_gc = lambda_gc * total_gc_l1

    return GCs_local, L_gc
def frequency_domain_denoise_learnable(
    grads_abs,              # (1, T, P)
    spectral_gate,          # LearnableSpectralGate
    n_fft=16,
    hop_length=8
):
    """
    Learnable frequency-domain GC aggregation

    return:
        gc_row: (P,)  -- 每个输入变量的频域 GC 强度
    """
    B, T, P = grads_abs.shape
    device = grads_abs.device

    # ---- reshape for STFT ----
    x = grads_abs.permute(0, 2, 1).reshape(B * P, T)  # (B*P, T)

    # ---- STFT ----
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        window=torch.hann_window(n_fft, device=device),
        return_complex=True
    )  # (B*P, F, T')

    mag = X.abs().view(B, P, X.shape[1], X.shape[2])  # (B, P, F, T')

    # ---- 可学习频域门控 ----
    mag_gated, gate = spectral_gate(mag)

    # ---- 频域 GC 聚合（RMS over freq & time）----
    gc_freq = torch.sqrt(
        (mag_gated ** 2).mean(dim=(-1, -2)) + 1e-12
    )  # (B, P)

    return gc_freq.squeeze(0)  # (P,)
class LearnableSpectralGate(nn.Module):
    """
    Learnable frequency-wise gate for STFT magnitude
    """
    def __init__(self, n_freq, hidden=32, init_bias=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_freq, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_freq)
        )
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(self, mag):
        """
        mag: (B, P, F, T')
        return:
            gated_mag: (B, P, F, T')
            gate:      (B, P, F, 1)
        """
        spec = mag.mean(dim=-1)              # (B, P, F)
        gate = torch.sigmoid(self.net(spec)) # (B, P, F)
        gate = gate.unsqueeze(-1)            # (B, P, F, 1)
        return mag * gate, gate
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(P, type, epoch, hidden_size, learning_rate,
                                  lambda_gc_sparse_base,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean',
                                  cutoff_ratio=0.2,
                                    tau=10.0,  # 新增参数，用于 CSDRF 细化

                                  ):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    target_device_id = 0
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
        data_root = f"/home/yx/KANGCI-main/realdataset/{type}/"
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
    patience = 15  # 容忍多少轮没有提升
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

        # GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220(
        #     model_fwd,
        #     input_seq,
        #     create_graph=True,
        #     lag_agg_local=lag_agg,
        #     freq_denoise=True,  # 或 True
        #     cutoff_ratio=cutoff_ratio,
        #     lambda_gc=lambda_gc_sparse_base,
        #     tau=tau,
        #
        # )

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
        'hidden_size': [256],  ##128   orgin256
        'learning_rate': [0.0008],  # 0.005 0.001   0.0009 best?    0.0008 auroc 0.93223
        'lambda_gc_sparse_base': [0.01],  #0.005
        'cutoff_ratio': [0.6],
        'lag_agg': ['softmax'],
        'data_path': ['/home/yx/KANGCI-main/realdataset'],
        'tau': [0.1],  # 新增参数，用于 CSDRF 细化

    }
    best_params = grid_search(param_grid)
# 256 Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9351962107380064
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9298710442105848


# 128 Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9358577439072796
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9330708482825034
# Best params: {'C_plus': 1.0, 'adj_threshold': 0.03, 'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9368632743245746
# Best params: {'C_plus': 1.0, 'adj_threshold': 0.03, 'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9343436557616585

#64Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 64, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9166997433251304
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 64, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9181600335285297
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 64, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9183668069116986
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 64, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9184169030571512


# 32 Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 32, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9121748564473023
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 32, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.903335616479768
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 32, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9121483951205315
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 32, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9033137073799578

# 512 Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/yx/KANGCI-main/realdataset', 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9397740202693763
# Best params: {'cutoff_ratio': 0.6, 'data_path': '/home/yx/KANGCI-main/realdataset', 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.01, 'learning_rate': 0.0008, 'tau': 0.1} with avg score: 0.9324951260181962
