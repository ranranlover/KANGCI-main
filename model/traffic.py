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
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)
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
    test_x = X[:T - 1]
    test_y = X[1:T]

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

        if counter > 0:
            print(f"  --- No improvement counter: {counter} / {patience} ---")

        # ------------------- 终止条件 (新增) -------------------
        if counter >= patience:
            print(
                f"🌟🌟🌟 Early stopping triggered after {i + 1} epochs! AUPRC did not improve for {patience} rounds. 🌟🌟🌟")
            break  # 退出 for i in range(epoch) 循环
    # End training
    # return best_score, best_auprc
    return best_score, best_auprc  # 假设这些变量在实际代码中被正确初始化和更新

























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


def compute_gradient_gc_smooth_universal_v3_1220(
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

        # # ---- lag 聚合 ----
        # if lag_agg_local == 'max':
        #     gc_row = grads_abs.squeeze(0).max(dim=0)[0]
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
        gc_l1_loss = gc_l1_loss + gc_row.abs().sum()

        # 保存 GC（仅用于评估）
        GCs_local[j, :] = gc_row

        del grads, grads_abs

    L_gc = lambda_gc * gc_l1_loss
    return GCs_local, L_gc
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

   #Best params: {'beta_prior': 7, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.0007, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.7320141125939561
#Best params: {'beta_prior': 7, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.001, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.5906369669145081

# Best params: {'beta_prior': 7, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.001, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.7225609756097561
# Best params: {'beta_prior': 5, 'cutoff_ratio': 0.8, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.001, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.5803885521820633

# Best params: {'beta_prior': 10, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.7602393005062127
# Best params: {'beta_prior': 10, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.6136090117221212

# Best params: {'beta_prior': 15, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.7546402822518792
# Best params: {'beta_prior': 15, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.6138240999136659

# Best params: {'beta_prior': 17, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.7628854118729866
# Best params: {'beta_prior': 17, 'cutoff_ratio': 0.7, 'data_path': '/home/user/wcj/KANGCI-main/realdataset', 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0008, 'learning_rate': 0.005, 'tau': 0.1, 'tau_prior': 0.03} with avg score: 0.6249421059996019
