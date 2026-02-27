import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from ComputeROC import compute_roc, compute_auprc
from src.efficient_kan import KAN
from src.efficient_kan.KAN_1219 import KAN_1219
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

    #component-wise generate p models
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

        if best_score < score_fusion :  # and score_fusion > 0.57
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
    def __init__(self, in_dim=5, hidden=64): # orign in_dim=5
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
    in_degree = GCs.sum(0).view(-1)              # (P,)
    out_degree = GCs.sum(1).view(-1).repeat(P)   # (P*P,)
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
def infer_Grangercausalityv4(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,lambda_alpha_reg,lambda_consistency):
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
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}#  0.01
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
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
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
    forward_mse = torch.tensor([losses_fwd[j].detach().cpu().item() if isinstance(losses_fwd[j], torch.Tensor) else float(losses_fwd[j])
                                for j in range(P)], dtype=torch.float32, device=device)  # (P,)
    reverse_mse = torch.tensor([losses_rev[j].detach().cpu().item() if isinstance(losses_rev[j], torch.Tensor) else float(losses_rev[j])
                                for j in range(P)], dtype=torch.float32, device=device)  # (P,)

    # For correlation computations we need tensors on device
    # But outs_fwd[j] is already tensor on device; target_seq[:, j] is on device

    for j in range(P):  # target
        out_f = outs_fwd[j]          # (T_f,)
        tgt_f = target_seq[:, j]     # (T_f,)
        out_r = outs_rev[j]          # (T_r,)
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
                GCs[j, i].to(device),                  # forward aggregated GC (target j, source i)
                GC2s[j, i].to(device),                 # reverse aggregated GC
                forward_mse[j].to(device),             # scalar
                reverse_mse[j].to(device),             # scalar
                mean_abs_diff_outs.to(device),         # scalar
                corr.to(device)                        # scalar
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
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
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
        for l in range(1, L+1):
            xt.append(X[t-l, :])  # t-1, t-2, ..., t-L
        xt = np.concatenate(xt, axis=0)  # (P*L,)
        lagged_X.append(xt)
    lagged_X = np.stack(lagged_X, axis=0)  # (T-L, P*L)
    y = X[L:, :]  # 对应预测目标
    return lagged_X, y
def infer_Grangercausalityv4_inge(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,lambda_alpha_reg,lambda_consistency):
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
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}#  0.01
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

        GCs = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P, 2 * P)], dim=0)
        # GCs = torch.stack([models[k].GC_integrated_v1(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated_v1(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],dim=0)

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

        # consistency_loss = torch.norm(GCs - GC2s, p=2)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        lambda_fused = 1.0
        # lambda_alpha_reg = 5.0
        # lambda_consistency = 0.05
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2  -
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
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
           
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )
    plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.81)


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
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P) or (T,)

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
            out_i = model_i(inp).view(-1)    # assume shape (T,)
            si = out_i.sum()                 # scalar
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
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
          
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
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P) or (T,)

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
            out_i = model_i(inp).view(-1)    # assume shape (T,)
            si = out_i.sum()                 # scalar (sum over time) as in your derivation
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
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.6f}, predict_loss2: {predict_loss2.item():.6f}, '
            f'Lsparse1: {Lsparse1.item():.6f}, Lsparse2: {Lsparse2.item():.6f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    return best_score

def infer_Grangercausalityv4_inge_plus_try(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse, gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False):
    """
    改写后：
    - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
    - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
    新增可选参数（非必要改变签名，只是内部可选）:
      - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
      - normalize_input: 是否先对 input_seq 做按变量（col）标准化
    其它行为与原先保持一致（fusion, alpha, sparse 等）。
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    best_auprc = 0.0
    best_auprc1 = 0.0
    best_auprc2 = 0.0
    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # Optional: compute column-wise mean/std for normalization (to make gradient magnitudes comparable)
    if normalize_input:
        # compute on training input (forward); use same stats for reversed for simplicity
        mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
    else:
        mean_cols = None
        std_cols = None

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)

    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a multi-output model ----
    def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
        """
        Compute GC matrix (P x P) from a single multi-output model:
          - model(input_seq_local) -> outs: (1, T, P)
          - For each output j in [0,P-1]:
              s_j = outs[0, :, j].sum()   # scalar aggregated over time
              grads = grad(s_j, inp) -> shape (1, T, P)
              GC[j, i] = aggregate_t |grads[0,t,i]|  (agg = mean or max)
        Args:
          model: single multi-output model
          input_seq_local: tensor (1, T, P)
          create_graph: whether to create grad graph (for backprop through GC)
          lag_agg_local: 'mean' or 'max'
        Returns:
          GCs: tensor shape (P, P) on same device
        """
        device_local = input_seq_local.device
        P_local = input_seq_local.shape[2]
        T_local = input_seq_local.shape[1]

        # optionally normalize input to reduce scale effects
        if mean_cols is not None and std_cols is not None:
            inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
        else:
            inp = input_seq_local

        # ensure a fresh tensor that requires grad
        inp = inp.detach().clone().requires_grad_(True)

        outs = model(inp).squeeze(0)  # (T, P)

        GCs_local = torch.zeros((P_local, P_local), device=device_local)

        # compute per-output gradients
        # Note: this loop does P backward/grad calls; for modest P this is fine and precise.
        for j in range(P_local):
            out_j = outs[:, j]        # (T,)
            s_j = out_j.sum()         # scalar
            grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # aggregate across time dim: mean or max
            if lag_agg_local == 'mean':
                gc_row = grads.abs().mean(dim=1).squeeze(0)   # (P,)
            elif lag_agg_local == 'max':
                gc_row = grads.abs().max(dim=1)[0].squeeze(0) # (P,)
            else:
                # default to mean
                gc_row = grads.abs().mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local  # shape (P, P)

    # ---------------------------------------------------------------

    # training loop
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # optionally normalize inputs for forward/backward pass (same scheme as GC)
        if mean_cols is not None and std_cols is not None:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        # forward predictions
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)   # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)   # (T, P)

        # compute per-target losses (same as before)
        losses_fwd = []
        for j in range(P):
            losses_fwd.append(loss_fn(outs_fwd_all[:, j], target_seq[:, j]))
        losses_rev = []
        for j in range(P):
            losses_rev.append(loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- compute GC matrices using gradient method ---
        if (i % gc_every) == 0:
            # compute possibly with create_graph to allow Lsparse backprop
            GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
        else:
            # monitoring only (no graph)
            with torch.no_grad():
                GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=False, lag_agg_local=lag_agg)

        # --- compute fused alphas and fused GC tensor as before ---
        # Note: you used outs_fwd, outs_rev lists previously; adapt to new shape
        # We provide outs_fwd_list and outs_rev_list (list of per-target tensors) to existing compute_edge_features
        outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           None, losses_fwd, losses_rev)  # models->None as not needed, adapt compute_edge_features if required
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s


        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # --- GC sparsity loss: ℓ1 on gradient-based GC (optionally backpropagatable) ---
        if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)
            Lsparse_fused = torch.tensor(0.0, device=device)  # 确保有定义

        # combine loss
        loss = (predict_loss1 + predict_loss2  -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2+Lsparse_fused)

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

            # --- 2. 计算 AUPRC ---
            # ❗ 新增 AUPRC 计算
            auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)
            # if best_score < score_fusion:
            if best_auprc < auprc_fusion:
                # best_score = score_fusion
                # best_score1 = score1
                # best_score2 = score2

                # ❗ 记录最佳 AUPRC
                best_auprc = auprc_fusion
                best_auprc1 = auprc1
                best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()
            if best_score < score_fusion:
            # if best_auprc < auprc_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                # ❗ 记录最佳 AUPRC
                # best_auprc = auprc_fusion
                # best_auprc1 = auprc1
                # best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    # plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)

    return best_score,best_auprc
    # return best_auprc
def infer_Grangercausalityv4_inge_plus_try_tosparse(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base,lambda_gc_sparse_fusion, gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False):
    """
    改写后：
    - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
    - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
    新增可选参数（非必要改变签名，只是内部可选）:
      - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
      - normalize_input: 是否先对 input_seq 做按变量（col）标准化
    其它行为与原先保持一致（fusion, alpha, sparse 等）。
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    best_auprc = 0.0
    best_auprc1 = 0.0
    best_auprc2 = 0.0
    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # Optional: compute column-wise mean/std for normalization (to make gradient magnitudes comparable)
    if normalize_input:
        # compute on training input (forward); use same stats for reversed for simplicity
        mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
    else:
        mean_cols = None
        std_cols = None

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)

    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a multi-output model ----
    def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
        """
        Compute GC matrix (P x P) from a single multi-output model:
          - model(input_seq_local) -> outs: (1, T, P)
          - For each output j in [0,P-1]:
              s_j = outs[0, :, j].sum()   # scalar aggregated over time
              grads = grad(s_j, inp) -> shape (1, T, P)
              GC[j, i] = aggregate_t |grads[0,t,i]|  (agg = mean or max)
        Args:
          model: single multi-output model
          input_seq_local: tensor (1, T, P)
          create_graph: whether to create grad graph (for backprop through GC)
          lag_agg_local: 'mean' or 'max'
        Returns:
          GCs: tensor shape (P, P) on same device
        """
        device_local = input_seq_local.device
        P_local = input_seq_local.shape[2]
        T_local = input_seq_local.shape[1]

        # optionally normalize input to reduce scale effects
        if mean_cols is not None and std_cols is not None:
            inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
        else:
            inp = input_seq_local

        # ensure a fresh tensor that requires grad
        inp = inp.detach().clone().requires_grad_(True)

        outs = model(inp).squeeze(0)  # (T, P)

        GCs_local = torch.zeros((P_local, P_local), device=device_local)

        # compute per-output gradients
        # Note: this loop does P backward/grad calls; for modest P this is fine and precise.
        for j in range(P_local):
            out_j = outs[:, j]        # (T,)
            s_j = out_j.sum()         # scalar
            grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # aggregate across time dim: mean or max
            if lag_agg_local == 'mean':
                gc_row = grads.abs().mean(dim=1).squeeze(0)   # (P,)
            elif lag_agg_local == 'max':
                gc_row = grads.abs().max(dim=1)[0].squeeze(0) # (P,)
            else:
                # default to mean
                gc_row = grads.abs().mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local  # shape (P, P)

    # ---------------------------------------------------------------

    # training loop
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # optionally normalize inputs for forward/backward pass (same scheme as GC)
        if mean_cols is not None and std_cols is not None:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        # forward predictions
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)   # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)   # (T, P)

        # compute per-target losses (same as before)
        losses_fwd = []
        for j in range(P):
            losses_fwd.append(loss_fn(outs_fwd_all[:, j], target_seq[:, j]))
        losses_rev = []
        for j in range(P):
            losses_rev.append(loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- compute GC matrices using gradient method ---
        if (i % gc_every) == 0:
            # compute possibly with create_graph to allow Lsparse backprop
            GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=gc_create_graph, lag_agg_local=lag_agg)
        else:
            # monitoring only (no graph)
            with torch.no_grad():
                GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=False, lag_agg_local=lag_agg)

        # --- compute fused alphas and fused GC tensor as before ---
        # Note: you used outs_fwd, outs_rev lists previously; adapt to new shape
        # We provide outs_fwd_list and outs_rev_list (list of per-target tensors) to existing compute_edge_features
        outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           None, losses_fwd, losses_rev)  # models->None as not needed, adapt compute_edge_features if required
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s


        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # --- GC sparsity loss: ℓ1 on gradient-based GC (optionally backpropagatable) ---
        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse_base * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)
            Lsparse_fused = torch.tensor(0.0, device=device)  # 确保有定义

        # combine loss
        loss = (predict_loss1 + predict_loss2  -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2+Lsparse_fused)

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

            # --- 2. 计算 AUPRC ---
            # ❗ 新增 AUPRC 计算
            auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)
            # if best_score < score_fusion:
            if best_auprc < auprc_fusion:
                # best_score = score_fusion
                # best_score1 = score1
                # best_score2 = score2

                # ❗ 记录最佳 AUPRC
                best_auprc = auprc_fusion
                best_auprc1 = auprc1
                best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()
            if best_score < score_fusion:
            # if best_auprc < auprc_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                # ❗ 记录最佳 AUPRC
                # best_auprc = auprc_fusion
                # best_auprc1 = auprc1
                # best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f},Lsparse_fused: {Lsparse_fused.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    # plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)

    return best_score,best_auprc
    # return best_auprc

def infer_Grangercausalityv4_inge_plus_try_tosparse_1208(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base,lambda_gc_sparse_fusion, gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False):
    """
    改写后：
    - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
    - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
    新增可选参数（非必要改变签名，只是内部可选）:
      - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
      - normalize_input: 是否先对 input_seq 做按变量（col）标准化
    其它行为与原先保持一致（fusion, alpha, sparse 等）。
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    best_auprc = 0.0
    best_auprc1 = 0.0
    best_auprc2 = 0.0
    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # Optional: compute column-wise mean/std for normalization (to make gradient magnitudes comparable)
    if normalize_input:
        # compute on training input (forward); use same stats for reversed for simplicity
        mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
    else:
        mean_cols = None
        std_cols = None

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)

    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a multi-output model ----
    # def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean', weights_t=None):
    #     """
    #     Compute GC matrix (P x P) from a single multi-output model:
    #       - model(input_seq_local) -> outs: (1, T, P)
    #       - For each output j in [0,P-1]:
    #           s_j = outs[0, :, j].sum()   # scalar aggregated over time
    #           grads = grad(s_j, inp) -> shape (1, T, P)
    #           GC[j, i] = aggregate_t |grads[0,t,i]|  (agg = mean or max)
    #     Args:
    #       model: single multi-output model
    #       input_seq_local: tensor (1, T, P)
    #       create_graph: whether to create grad graph (for backprop through GC)
    #       lag_agg_local: 'mean' or 'max'
    #     Returns:
    #       GCs: tensor shape (P, P) on same device
    #     """
    #     device_local = input_seq_local.device
    #     P_local = input_seq_local.shape[2]
    #     T_local = input_seq_local.shape[1]
    #
    #     # optionally normalize input to reduce scale effects
    #     if mean_cols is not None and std_cols is not None:
    #         inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
    #     else:
    #         inp = input_seq_local
    #
    #     # ensure a fresh tensor that requires grad
    #     inp = inp.detach().clone().requires_grad_(True)
    #
    #     outs = model(inp).squeeze(0)  # (T, P)
    #
    #     GCs_local = torch.zeros((P_local, P_local), device=device_local)
    #
    #     # compute per-output gradients
    #     # Note: this loop does P backward/grad calls; for modest P this is fine and precise.
    #     for j in range(P_local):
    #         out_j = outs[:, j]        # (T,)
    #         if weights_t is not None:
    #             # 使用变量 j 对应的权重列 (T,) 进行加权
    #             weight_j = weights_t[:, j].to(device_local)
    #             s_j = (out_j * weight_j).sum()
    #         else:
    #             # 原始逻辑：简单求和
    #             s_j = out_j.sum()
    #         grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
    #         # aggregate across time dim: mean or max
    #         if lag_agg_local == 'mean':
    #             gc_row = grads.abs().mean(dim=1).squeeze(0)   # (P,)
    #         elif lag_agg_local == 'max':
    #             gc_row = grads.abs().max(dim=1)[0].squeeze(0) # (P,)
    #         else:
    #             # default to mean
    #             gc_row = grads.abs().mean(dim=1).squeeze(0)
    #
    #         GCs_local[j, :] = gc_row
    #
    #     return GCs_local  # shape (P, P)

    # ---------------------------------------------------------------

    # training loop
    def compute_gradient_gc_freq(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                 use_amp_only=True):
        """
        Frequency-Domain Gradient-Based Granger Causality
        - Apply FFT to input sequence before gradient calculation.
        """

        device_local = input_seq_local.device
        P_local = input_seq_local.shape[2]
        T_local = input_seq_local.shape[1]

        # normalize option
        inp = input_seq_local.detach().clone()

        # --- Frequency Transform (FFT) ---
        freq = torch.fft.rfft(inp, dim=1)  # (1, T', P)
        if use_amp_only:
            inp = torch.abs(freq)  # amplitude only
        else:
            inp = torch.view_as_real(freq)  # keep real/imag
            inp = inp.norm(dim=-1)  # magnitude

        # ensure grad
        inp = inp.detach().clone().requires_grad_(True)

        # forward
        outs = model(inp).squeeze(0)  # (T or T', P)

        GCs_local = torch.zeros((P_local, P_local), device=device_local)

        # compute gradients per output
        for j in range(P_local):
            out_j = outs[:, j]  # (T')
            s_j = out_j.sum()  # scalar

            grads = torch.autograd.grad(
                s_j, inp,
                create_graph=create_graph,
                retain_graph=True
            )[0]  # (1, T', P)

            grads_abs = grads.abs()

            # time aggregation
            if lag_agg_local == 'mean':
                gc_row = grads_abs.mean(dim=1).squeeze(0)
            elif lag_agg_local == 'max':
                gc_row = grads_abs.max(dim=1)[0].squeeze(0)
            else:
                gc_row = grads_abs.mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local


    # 2 0.6891  0.13183    1   0.7859 0.22466  0015
    def compute_gradient_gc_smooth(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                   smooth=True, grad_clip=3.0):
        """
        Gradient Smoothing + Clip for More Stable GC
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
                # reshape to (batch=1, channels=P, T)
                g = grads_abs.permute(0, 2, 1)  # (1, P, T)
                g = torch.nn.functional.pad(g, (1, 1), mode='reflect')  # pad time dim
                g = torch.nn.functional.conv1d(g, kernel.repeat(P_local, 1, 1), groups=P_local)
                grads_abs = g.permute(0, 2, 1)  # back to (1, T, P)

            # --- clip outliers (reduce GC noise spikes) ---
            if grad_clip is not None:
                grads_abs = grads_abs.clamp(max=grad_clip)

            # aggregate
            if lag_agg_local == 'mean':
                gc_row = grads_abs.mean(dim=1).squeeze(0)
            elif lag_agg_local == 'max':
                gc_row = grads_abs.max(dim=1)[0].squeeze(0)
            else:
                gc_row = grads_abs.mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local

    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # optionally normalize inputs for forward/backward pass (same scheme as GC)
        if mean_cols is not None and std_cols is not None:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        # forward predictions
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)   # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)   # (T, P)

        # compute per-target losses (same as before)
        losses_fwd = []
        loss_t_fwd = (outs_fwd_all - target_seq) ** 2
        loss_t_rev = (outs_rev_all - reversed_target_seq) ** 2
        for j in range(P):
            losses_fwd.append(loss_fn(outs_fwd_all[:, j], target_seq[:, j]))
        losses_rev = []
        for j in range(P):
            losses_rev.append(loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)
        W_fwd_normalized = loss_t_fwd / (loss_t_fwd.sum(dim=0, keepdim=True) + 1e-8)
        W_rev_normalized = loss_t_rev / (loss_t_rev.sum(dim=0, keepdim=True) + 1e-8)

        W_fwd_detached = W_fwd_normalized.detach()
        W_rev_detached = W_rev_normalized.detach()
        # --- compute GC matrices using gradient method ---
        if (i % gc_every) == 0:

            GCs = compute_gradient_gc_smooth(model_fwd, input_seq, create_graph=gc_create_graph,
                                                lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_smooth(model_rev, reversed_input_seq, create_graph=gc_create_graph,
                                                 lag_agg_local=lag_agg)
        else:
            # monitoring only (no graph)
            with torch.no_grad():

                GCs = compute_gradient_gc_smooth(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_smooth(model_rev, reversed_input_seq, create_graph=False,
                                                     lag_agg_local=lag_agg)

        # --- compute fused alphas and fused GC tensor as before ---
        # Note: you used outs_fwd, outs_rev lists previously; adapt to new shape
        # We provide outs_fwd_list and outs_rev_list (list of per-target tensors) to existing compute_edge_features
        outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           None, losses_fwd, losses_rev)  # models->None as not needed, adapt compute_edge_features if required
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s


        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # --- GC sparsity loss: ℓ1 on gradient-based GC (optionally backpropagatable) ---
        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse_base * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)
            Lsparse_fused = torch.tensor(0.0, device=device)  # 确保有定义

        # combine loss
        loss = (predict_loss1 + predict_loss2  -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2+Lsparse_fused)

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

            # --- 2. 计算 AUPRC ---
            # ❗ 新增 AUPRC 计算
            auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)
            # if best_score < score_fusion:
            if best_auprc < auprc_fusion:
                # best_score = score_fusion
                # best_score1 = score1
                # best_score2 = score2

                # ❗ 记录最佳 AUPRC
                best_auprc = auprc_fusion
                best_auprc1 = auprc1
                best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()
            if best_score < score_fusion:
            # if best_auprc < auprc_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

                # ❗ 记录最佳 AUPRC
                # best_auprc = auprc_fusion
                # best_auprc1 = auprc1
                # best_auprc2 = auprc2

                best_GCs = GCs.detach().clone()
                best_GC2s = GC2s.detach().clone()
                best_fused = fused_GC_tensor.detach().clone()

        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f},Lsparse_fused: {Lsparse_fused.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    # plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)

    return best_score,best_auprc
    # return best_auprc


# --- 辅助函数：平滑 + 通用 GC 计算 ---
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

# ---------------- GC computation helpers (gradient-based) ----------------
def compute_gradient_gc_smooth_1213(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                               smooth=True, grad_clip=3.0):
    device_l = input_seq_local.device
    P_l = input_seq_local.shape[2]

    # (1) 确保 inp 具有梯度追踪，并克隆/分离原始输入以防意外修改
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

        # ❗ 核心修改：计算 Attribution = Gradient * Input
        # (1, T, P) * (1, T, P) -> (1, T, P)
        # inp 已经是 requires_grad_(True) 的副本，可以直接使用
        attribution_scores = grads * inp

        # ❗ 将计算 Attribution scores 的绝对值，作为新的因果强度指标
        grads_abs = attribution_scores.abs()  # (1, T, P)

        # ----------------- 以下平滑和聚合逻辑不变 -----------------
        if smooth:
            # 3-point moving average along time axis
            # 注意：平滑操作现在是对 Attribution Score 进行的
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device_l).view(1, 1, 3)
            g = grads_abs.permute(0, 2, 1)  # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel.repeat(P_l, 1, 1), groups=P_l)
            grads_abs = g.permute(0, 2, 1)

        if grad_clip is not None:
            # 裁剪现在是针对 Attribution scores 的绝对值
            grads_abs = grads_abs.clamp(max=grad_clip)

        if lag_agg_local == 'mean':
            # 时间轴上的平均 Attribtion score
            gc_row = grads_abs.mean(dim=1).squeeze(0)
        elif lag_agg_local == 'max':
            # 时间轴上的最大 Attribtion score
            gc_row = grads_abs.max(dim=1)[0].squeeze(0)
        else:
            gc_row = grads_abs.mean(dim=1).squeeze(0)

        GCs_local[j, :] = gc_row

    return GCs_local


import torch.nn.functional as F


# ---------------- GC computation helpers (Integrated Gradients) ----------------
def compute_gradient_gc_smooth_ig_1213(model, input_seq_local, create_graph=False, lag_agg_local='mean',
                                  smooth=True, grad_clip=3.0,
                                  # ❗ 新增 IG 参数
                                  IG_STEPS=20,
                                  baseline=None):
    device_l = input_seq_local.device
    P_l = input_seq_local.shape[2]

    # 确保 input_seq_local 是 (1, T, P)
    inp = input_seq_local.detach().clone()
    T = inp.shape[1]

    # --- 1. 设置基线 ---
    if baseline is None:
        # 默认使用全零基线 (对于归一化数据常见且合理)
        baseline_tensor = torch.zeros_like(inp, device=device_l)
    else:
        # 确保 baseline 是与 inp 形状兼容的 (1, 1, P) 或 (1, T, P)
        baseline_tensor = baseline.to(device_l)

    # 计算差值 (Input - Baseline)
    input_diff = inp - baseline_tensor  # (1, T, P)

    # 初始化 IG 累加器 (将存储最终的 Integrated Gradients Scores)
    IG_scores_abs = torch.zeros((1, T, P_l), device=device_l)

    # --- 2. 迭代每个输出变量 j ---
    for j in range(P_l):
        # 初始化当前输出变量 j 的累积梯度 (用于 IG 积分)
        sum_of_grads = torch.zeros_like(inp, device=device_l)  # (1, T, P)

        # --- 3. 黎曼近似和迭代 (IG 积分) ---
        for k in range(1, IG_STEPS + 1):
            alpha = k / IG_STEPS

            # 计算插值点 x_k = baseline + alpha * (input - baseline)
            # ❗ 必须在此处 requires_grad_，因为每次迭代 x_k 都不同
            x_k = (baseline_tensor + alpha * input_diff).requires_grad_(True)

            # 前向传播并计算梯度
            outs_k = model(x_k).squeeze(0)  # (T, P)
            out_j_k = outs_k[:, j]
            s_j_k = out_j_k.sum()

            grads_k = torch.autograd.grad(
                s_j_k, x_k,
                create_graph=create_graph,
                retain_graph=True
            )[0]  # (1, T, P)

            # 累加梯度
            sum_of_grads = sum_of_grads + grads_k

        # --- 4. 计算最终 IG Score ---
        # IG = (Input - Baseline) * (Average Gradient)
        # Average Gradient = (1 / IG_STEPS) * sum_of_grads
        average_grads = sum_of_grads / IG_STEPS

        # 最终 Attribution Score (IG Score)
        # (1, T, P) * (1, T, P) -> (1, T, P)
        attribution_scores = input_diff * average_grads

        # 将 IG scores 的绝对值累加到全局 IG 矩阵
        current_ig_abs = attribution_scores.abs()

        # ----------------- 5. 平滑、裁剪和聚合 -----------------
        # GCs_local 矩阵用于存储最终结果
        if j == 0:
            GCs_local = torch.zeros((P_l, P_l), device=device_l)

        if smooth:
            # 3-point moving average along time axis
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device_l).view(1, 1, 3)
            g = current_ig_abs.permute(0, 2, 1)  # (1, P, T)
            g = F.pad(g, (1, 1), mode='reflect')
            g = F.conv1d(g, kernel.repeat(P_l, 1, 1), groups=P_l)
            current_ig_abs = g.permute(0, 2, 1)

        if grad_clip is not None:
            # 裁剪现在是针对 IG scores 的绝对值
            current_ig_abs = current_ig_abs.clamp(max=grad_clip)

        # 聚合（均值或最大值）
        if lag_agg_local == 'mean':
            gc_row = current_ig_abs.mean(dim=1).squeeze(0)
        elif lag_agg_local == 'max':
            gc_row = current_ig_abs.max(dim=1)[0].squeeze(0)
        else:
            gc_row = current_ig_abs.mean(dim=1).squeeze(0)

        GCs_local[j, :] = gc_row

    return GCs_local
# compute_batched_gc 函数无需修改，因为它只负责分块和平均。

def compute_gradient_gc_smooth_ig_1213(
    model,
    input_seq,
    create_graph=False,
    lag_agg_local='mean',
    return_grad=False
):
    """
    返回:
    - GC: (P, P)  梯度 Granger 因果矩阵
    - grad_abs: (P, P) 输入-输出梯度强度（用于 GRNGC 正则）
    """
    inp = input_seq.detach().clone().requires_grad_(True)
    outs = model(inp).squeeze(0)  # (T, P)

    T, P = outs.shape
    device = inp.device

    GC = torch.zeros(P, P, device=device)
    grad_abs_all = torch.zeros(P, P, device=device)

    for j in range(P):
        s_j = outs[:, j].sum()

        grads = torch.autograd.grad(
            s_j,
            inp,
            create_graph=create_graph,
            retain_graph=True
        )[0]  # (1, T, P)

        g_abs = grads.abs()  # (1, T, P)

        if lag_agg_local == 'mean':
            g_agg = g_abs.mean(dim=1).squeeze(0)   # (P,)
        elif lag_agg_local == 'max':
            g_agg = g_abs.max(dim=1)[0].squeeze(0)
        else:
            raise ValueError(f'Unknown lag_agg_local: {lag_agg_local}')

        GC[j] = g_agg
        grad_abs_all[j] = g_agg

    if return_grad:
        return GC, grad_abs_all
    else:
        return GC


import torch
import numpy as np

import torch
import torch.fft


def compute_gradient_gc_spectral_1218(model, input_seq, cutoff_ratio=0.2, create_graph=True, lag_agg_local='mean'):
    """
    在梯度计算中集成频域低通滤波，抑制高频噪声。
    """
    input_seq.requires_grad = True
    # 前向传播 (1, T, P)
    out = model(input_seq)

    T, P = input_seq.shape[1], input_seq.shape[2]
    gc_matrix = torch.zeros(P, P).to(input_seq.device)

    # 遍历每个输出维度 j
    for j in range(P):
        # 计算第 j 个变量对所有输入的总梯度
        # grad_out shape: (1, T, P)
        grad_out = torch.autograd.grad(out[0, :, j].sum(), input_seq,
                                       create_graph=create_graph, retain_graph=True)[0]

        for i in range(P):
            if i == j: continue

            # 提取变量 i 对输出 j 的时间序列梯度 (T,)
            grad_seq = grad_out[0, :, i]

            # --- FFT 滤波逻辑 ---
            # 转到频域
            ffted = torch.fft.rfft(grad_seq)
            n_freq = ffted.size(0)

            # 构建低通掩码 (保留低频趋势，消除高频抖动)
            mask = torch.zeros_like(ffted)
            keep_idx = max(1, int(n_freq * cutoff_ratio))
            mask[:keep_idx] = 1.0

            # 逆 FFT 回到时域
            filtered_grad = torch.fft.irfft(ffted * mask, n=T)

            # 聚合
            if lag_agg_local == 'mean':
                gc_matrix[i, j] = torch.abs(filtered_grad).mean()
            else:
                gc_matrix[i, j] = torch.abs(filtered_grad).max()

    return gc_matrix


import torch


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


def compute_gradient_gc_smooth_universal_v2(model, input_seq_local, create_graph=True,
                                            lag_agg_local='mean', freq_denoise=False, cutoff_ratio=0.2):
    """
    集成频域去噪的 GC 计算函数
    """
    device_local = input_seq_local.device
    P_local = input_seq_local.shape[2]
    T_local = input_seq_local.shape[1]

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

        grads_abs = grads.abs()

        # --- 核心改进：频域去噪代替或辅助滑动平均 ---
        if freq_denoise:
            # 直接对时间轴上的梯度分布进行 FFT 平滑
            grads_abs = frequency_domain_denoise(grads_abs, cutoff_ratio=cutoff_ratio)
        else:
            # 传统的 3 点滑动平均
            kernel = torch.tensor([0.25, 0.5, 0.25], device=device_local).view(1, 1, 3)
            g = grads_abs.permute(0, 2, 1)
            g = torch.nn.functional.pad(g, (1, 1), mode='reflect')
            g = torch.nn.functional.conv1d(g, kernel.repeat(P_local, 1, 1), groups=P_local)
            grads_abs = g.permute(0, 2, 1)

        # --- 聚合 ---
        if lag_agg_local == 'mean':
            gc_row = grads_abs.mean(dim=1).squeeze(0)
        elif lag_agg_local == 'max':
            gc_row = grads_abs.max(dim=1)[0].squeeze(0)
        else:
            gc_row = grads_abs.mean(dim=1).squeeze(0)

        GCs_local[j, :] = gc_row

    return GCs_local



def compute_gradient_gc_smooth_universal_v3(model, input_seq_local, create_graph=True,
                                            lag_agg_local='mean', freq_denoise=True, cutoff_ratio=0.2):
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


def compute_gradient_gc_v3_optimized(model, input_seq, create_graph=True, eps=1e-8):
    """
    深度优化的 GC 计算函数
    1. 支持 Batch 梯度并行化
    2. 引入 L2 归一化增强稳健性
    3. 优化计算图管理
    """
    device = input_seq.device
    B, T, P = input_seq.shape  # Batch, Time, Predictors

    # 1. 准备输入
    # 建议使用 clone 以免干扰原始数据流
    inp = input_seq.detach().clone().requires_grad_(True)

    # 2. 前向传播
    # 假设模型输出为 (B, T, P)
    outs = model(inp)

    # 初始化 GC 矩阵: (B, P, P) -> [目标变量, 来源变量]
    gc_matrix = torch.zeros((B, P, P), device=device)

    # 3. 预先定义平滑卷积核 (3点高斯平滑)
    # 使用深度卷积 (groups=P) 一次性处理
    kernel = torch.tensor([0.2, 0.6, 0.2], device=device).view(1, 1, 3).repeat(P, 1, 1)

    # 4. 遍历目标变量 (因果受体)
    # 虽然可以用 vmap，但 grad 对于多输出的 Jacobian 有时更稳健
    for j in range(P):
        # 取出所有 Batch 中第 j 个变量的输出总和
        # 使用 sum() 是为了获得对 inp 的标量梯度
        target_output = outs[:, :, j].sum()

        # 计算梯度: d(Output_j) / d(All_Inputs)
        # shape: (B, T, P)
        grads = torch.autograd.grad(
            target_output, inp,
            create_graph=create_graph,
            retain_graph=True  # 因为循环还要用 outs 的图
        )[0]

        # --- 核心改进 A: 梯度平滑与清理 ---
        grads_abs = grads.abs()

        # 维度转换: (B, T, P) -> (B, P, T) 适配 Conv1d
        g = grads_abs.permute(0, 2, 1)
        g = F.pad(g, (1, 1), mode='reflect')
        g = F.conv1d(g, kernel, groups=P)  # 每个变量的时间轴独立平滑

        # --- 核心改进 B: 归一化处理 (至关重要) ---
        # 梯度受模型缩放影响大，计算每个来源变量对目标 j 的贡献占比
        # 计算在时间维度的均值
        # shape: (B, P)
        contribution = g.mean(dim=-1)

        # L2 归一化：让每一行的因果强度具有可比性，消除特定时间点梯度爆发的影响
        row_norm = torch.norm(contribution, p=2, dim=1, keepdim=True) + eps
        gc_row = contribution / row_norm

        gc_matrix[:, j, :] = gc_row

    # 5. 最终清理
    if not create_graph:
        outs.detach_()
        del outs

    return gc_matrix

def infer_Grangercausalityv4_inge_plus_try_tosparse_1218(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False,
                                  # --- 新增/调整的超参数 ---
                                  lambda_confidence=0.01,  # 置信度损失的权重
                                  grad_clip_quantile=0.99  # 动态裁剪的分位数
                                  ,cutoff_ratio=0.2):
    """
    说明: 1211 优化版本。
    - 集成动态梯度裁剪 (基于分位数)。
    - 集成基于预测损失的 Alpha 置信度正则化 (L_confidence)。
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import pandas as pd

    # ---------------- 准备工作 (略去重复的初始化和数据加载) ----------------
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    # 假设 read_dream4, KAN, FusionEdge, compute_roc, compute_auprc,
    # off_diagonal, compute_edge_features_1205 已在 scope 内或已导入
    best_score = 0
    best_auprc = 0.0

    # 模拟数据加载（使用您的原始逻辑）
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)
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

    mean_cols, std_cols = None, None
    if normalize_input:
        mean_cols = input_seq.mean(dim=1, keepdim=True)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8

    # 模型构建 (KAN, FusionEdge)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()
    # ---------------- 训练循环 ----------------

    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # 1. 前向预测 (预测损失用于反向传播)
        if normalize_input:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)

        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]

        predict_loss1 = sum(losses_fwd)  # L_fwd
        predict_loss2 = sum(losses_rev)  # L_rev
        # --- 核心改进：频域梯度提取 ---
        # GCs = compute_gradient_gc_spectral_1218(model_fwd, input_seq, cutoff_ratio=cutoff_ratio)
        # GC2s = compute_gradient_gc_spectral_1218(model_rev, reversed_input_seq, cutoff_ratio=cutoff_ratio)
        # 2. GC 矩阵计算 (使用通用的平滑函数)
        if (i % gc_every) == 0:
            GCs_raw = compute_gradient_gc_spectral_1218(model_fwd, input_seq,
                                                        cutoff_ratio=cutoff_ratio,
                                                        create_graph=gc_create_graph)
            GC2s_raw = compute_gradient_gc_spectral_1218(model_rev, reversed_input_seq,
                                                         cutoff_ratio=cutoff_ratio,
                                                         create_graph=gc_create_graph)

            # 动态梯度裁剪 (你的 1211 逻辑)
            with torch.no_grad():
                combined = torch.cat([GCs_raw.view(-1), GC2s_raw.view(-1)])
                nonzero = combined[combined > 1e-6]
                clip_threshold = torch.quantile(nonzero, grad_clip_quantile) if nonzero.numel() > 0 else 1.0

            GCs = GCs_raw.clamp(max=clip_threshold)
            GC2s = GC2s_raw.clamp(max=clip_threshold)
        else:
            with torch.no_grad():
                GCs = compute_gradient_gc_spectral_1218(model_fwd, input_seq, cutoff_ratio=cutoff_ratio,
                                                        create_graph=False)
                GC2s = compute_gradient_gc_spectral_1218(model_rev, reversed_input_seq, cutoff_ratio=cutoff_ratio,
                                                         create_graph=False)

            # 3. 融合与置信度正则
        feat_edges = compute_edge_features_1205(GCs, GC2s, [o for o in outs_fwd_all.T], [o for o in outs_rev_all.T],
                                                target_seq, reversed_target_seq, None, losses_fwd, losses_rev)
        alphas = fusion_edge(feat_edges).view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        L_confidence = lambda_confidence * ((1.0 - alphas) * predict_loss1 + alphas * predict_loss2).mean()

        # 稀疏损失
        Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()

        loss = predict_loss1 + predict_loss2 + Lsparse_fused + L_confidence  # - lambda_alpha_reg * alpha_reg

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        # 6. 评估和记录
        # ... (评估和记录逻辑与原代码保持一致)
        # 评估和记录部分逻辑与原代码保持一致
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

            auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)

            # Update best by AUPRC fusion primarily, then ROC fusion
            if best_auprc < auprc_fusion:
                best_auprc = auprc_fusion
                # ... save best GCs/GC2s/fused ...
            if best_score < score_fusion:
                best_score = score_fusion
                # ... save best GCs/GC2s/fused ...

        print(
            f'Epoch [{i + 1}/{epoch}], L_pred1: {predict_loss1.item():.4f}, L_pred2: {predict_loss2.item():.4f}, '
            f'L_conf: {L_confidence.item():.4f}, L_sparse_fused: {Lsparse_fused.item():.4f}, '
            f'ROC_fwd: {score1:.4f}, ROC_rev: {score2:.4f}, ROC_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    return best_score, best_auprc

from sklearn import metrics

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
def compute_gradient_gc_simple(model, input_seq_local):
    """
    方法一的梯度 GC，返回 (P, P)
    input_seq_local: (1, T, P)
    """
    device_local = input_seq_local.device
    P_local = input_seq_local.shape[2]
    inp = input_seq_local.squeeze(0).detach().clone().requires_grad_(True)  # (T, P)
    outs = model(inp.unsqueeze(0)).squeeze(0)  # (T, P)

    GC_local = torch.zeros((P_local, P_local), device=device_local)

    for j in range(P_local):
        model.zero_grad()
        out_j = outs[:, j].sum()
        grad = torch.autograd.grad(out_j, inp, retain_graph=True, create_graph=False)[0]  # (T, P)
        GC_local[j, :] = grad.abs().mean(dim=0)

    return GC_local

# 1219yizhizaiyong
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False,
                                  # --- 新增/调整的超参数 ---
                                  cutoff_ratio=0.2,
                                  grad_clip_quantile=0.99  # 动态裁剪的分位数
                                  ):
    """
    说明: 1211 优化版本。
    - 集成动态梯度裁剪 (基于分位数)。
    - 集成基于预测损失的 Alpha 置信度正则化 (L_confidence)。
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import pandas as pd

    # ---------------- 准备工作 (略去重复的初始化和数据加载) ----------------
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    # 假设 read_dream4, KAN, FusionEdge, compute_roc, compute_auprc,
    # off_diagonal, compute_edge_features_1205 已在 scope 内或已导入
    best_score = 0
    best_auprc = 0.0

    # 模拟数据加载（使用您的原始逻辑）
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)
    length = X.shape[0]

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    mean_cols, std_cols = None, None
    if normalize_input:
        mean_cols = input_seq.mean(dim=1, keepdim=True)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8

    # 模型构建 (KAN, FusionEdge)
    model_fwd = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()
    # ---------------- 训练循环 ----------------

    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()


        inp_fwd = input_seq
        inp_rev = reversed_input_seq

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)

        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]

        predict_loss1 = sum(losses_fwd)  # L_fwd
        predict_loss2 = sum(losses_rev)  # L_rev

        # 2. GC 矩阵计算 (使用通用的平滑函数)
        if (i % gc_every) == 0:
            # ❗ 步骤 2.1: 计算 GCs_raw (平滑后，未裁剪)
            GCs_raw = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq,cutoff_ratio=cutoff_ratio,
                                                           create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s_raw = compute_gradient_gc_smooth_universal_v2(model_rev, reversed_input_seq,cutoff_ratio=cutoff_ratio,
                                                            create_graph=gc_create_graph, lag_agg_local=lag_agg)
            # GCs_raw = compute_gradient_gc_v3_optimized(model_fwd, input_seq,
            #                                                create_graph=gc_create_graph)
            # GC2s_raw = compute_gradient_gc_v3_optimized(model_rev, reversed_input_seq,
            #                                                 create_graph=gc_create_graph)
            # GCs_raw = compute_gradient_gc_simple(model_fwd, input_seq)
            # GC2s_raw = compute_gradient_gc_simple(model_rev, reversed_input_seq)

            # ❗ 步骤 2.2: 动态裁剪
            # 裁剪阈值取两个 GC 矩阵的梯度绝对值之和的 Q_99
            with torch.no_grad():
                combined_gc_values = torch.cat([GCs_raw.view(-1), GC2s_raw.view(-1)])
                # 忽略 0 值，防止稀疏矩阵对分位数的干扰
                nonzero_combined = combined_gc_values[combined_gc_values > 1e-6]
                if nonzero_combined.numel() > 0:
                    clip_threshold = torch.quantile(nonzero_combined, grad_clip_quantile)
                else:
                    clip_threshold = 1.0  # 默认值

            # 应用裁剪
            GCs = GCs_raw.clamp(max=clip_threshold)
            GC2s = GC2s_raw.clamp(max=clip_threshold)

            # 由于裁剪是在训练图上进行的，Lsparse 仍然可以反传
        else:
            # Monitoring only (no graph)
            print('no use')
            with torch.no_grad():
                GCs = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq, create_graph=False,
                                                           lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_smooth_universal_v2(model_rev, reversed_input_seq, create_graph=False,
                                                            lag_agg_local=lag_agg)
                # Note: 此时不对 GCs, GC2s 进行裁剪，仅用于评估/展示

        # 3. 融合和正则化
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        # 熵正则项
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # ❗ 创新点 2: 基于预测损失的 Alpha 置信度正则化 L_confidence
        # L_confidence = lambda_conf * ((1-alpha) * L_fwd + alpha * L_rev).mean()
        # 由于 L_fwd/L_rev 是标量，需要广播
        # L_confidence = lambda_confidence * (
        #         (1.0 - alphas) * predict_loss1 + alphas * predict_loss2
        # ).mean()

        # GC 稀疏化损失
        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse_base * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)
            Lsparse_fused = torch.tensor(0.0, device=device)

        # 4. 组合总损失
        loss = (predict_loss1 + predict_loss2
                - lambda_alpha_reg * alpha_reg
                + Lsparse1 + Lsparse2 + Lsparse_fused
                )  # ❗ 新增置信度损失项

        # 5. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # 6. 评估和记录
        # ... (评估和记录逻辑与原代码保持一致)
        # 评估和记录部分逻辑与原代码保持一致
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            # score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            # score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            # score_fusion = compute_roc(GC, off_diagonal(fused_np), False)
            score1,auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            score2,auprc2 = compute_roc_1219(GC, off_diagonal(GC2s_np), False)
            score_fusion,auprc_fusion = compute_roc_1219(GC, off_diagonal(fused_np), False)

            # auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            # auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            # auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)

            # Update best by AUPRC fusion primarily, then ROC fusion
            if best_auprc < auprc_fusion:
                best_auprc = auprc_fusion
                # ... save best GCs/GC2s/fused ...
            if best_score < score_fusion:
                best_score = score_fusion
                # ... save best GCs/GC2s/fused ...

        print(
            f'Epoch [{i + 1}/{epoch}], L_pred1: {predict_loss1.item():.4f}, L_pred2: {predict_loss2.item():.4f}, '
            f' L_sparse_fused: {Lsparse_fused.item():.4f}, '
            f'ROC_fwd: {score1:.4f}, ROC_rev: {score2:.4f}, ROC_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    return best_score, best_auprc
class CausalEnricher(nn.Module):
    """
    使用 GC 矩阵作为注意力权重来构造 time-step 基础的 context，然后 x' = x + gamma * context
    - input_seq: (1, T, P)
    - gc_matrix: (P, P)  row j 表示权重分配到各输入 i 用于目标 j
    Returns enriched_seq: (1, T, P)
    """
    def __init__(self, P, init_gamma: float = 0.5):
        super().__init__()
        self.P = P
        # gamma 可学习，初始化为 init_gamma（可正可负）
        self.gamma = nn.Parameter(torch.tensor(init_gamma, dtype=torch.float32))

    def forward(self, input_seq: torch.Tensor, gc_matrix: torch.Tensor):
        # input_seq: (1, T, P)
        # gc_matrix: (P, P)
        device = input_seq.device
        # safe cast
        gc = gc_matrix.to(device)
        # softmax 每行（每个目标 j 对输入 i 的权重）
        att = F.softmax(gc, dim=1)            # (P, P)
        x = input_seq.squeeze(0)              # (T, P)
        # context_t_j = sum_i att[j,i] * x_t[i]  -> for all t: context = x @ att.T
        context = x @ att.t()                 # (T, P)
        enriched = x + self.gamma * context   # (T, P)
        return enriched.unsqueeze(0)          # (1, T, P)
def infer_Grangercausalityv4_inge_plus_try2(P, type, epoch, hidden_size,  learning_rate,
                                  lambda_alpha_reg,  lambda_gc_sparse, gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False):
    """
    改写后：
    - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
    - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
    新增可选参数（非必要改变签名，只是内部可选）:
      - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
      - normalize_input: 是否先对 input_seq 做按变量（col）标准化
    其它行为与原先保持一致（fusion, alpha, sparse 等）。
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
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # Optional: compute column-wise mean/std for normalization (to make gradient magnitudes comparable)
    if normalize_input:
        # compute on training input (forward); use same stats for reversed for simplicity
        mean_cols = input_seq.mean(dim=1, keepdim=True)  # (1,1,P)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8
    else:
        mean_cols = None
        std_cols = None

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)

    fusion_edge = FusionEdge(in_dim=6, hidden=64).to(device)

    causal_enricher_fwd = CausalEnricher(P).to(device)
    causal_enricher_rev = CausalEnricher(P).to(device)

    # 将它们的参数加入 optimizer（在你已有 params 列表中追加）
    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters()) + \
             list(causal_enricher_fwd.parameters()) + list(causal_enricher_rev.parameters())

    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1},
        {'params': list(causal_enricher_fwd.parameters()), 'lr': learning_rate },
        {'params': list(causal_enricher_rev.parameters()), 'lr': learning_rate },
    ])

    loss_fn = nn.MSELoss()

    # ---- helper: compute gradient-based GC for a multi-output model ----
    def compute_gradient_gc_for_model(model, input_seq_local, create_graph=False, lag_agg_local='mean'):
        """
        Compute GC matrix (P x P) from a single multi-output model:
          - model(input_seq_local) -> outs: (1, T, P)
          - For each output j in [0,P-1]:
              s_j = outs[0, :, j].sum()   # scalar aggregated over time
              grads = grad(s_j, inp) -> shape (1, T, P)
              GC[j, i] = aggregate_t |grads[0,t,i]|  (agg = mean or max)
        Args:
          model: single multi-output model
          input_seq_local: tensor (1, T, P)
          create_graph: whether to create grad graph (for backprop through GC)
          lag_agg_local: 'mean' or 'max'
        Returns:
          GCs: tensor shape (P, P) on same device
        """
        device_local = input_seq_local.device
        P_local = input_seq_local.shape[2]
        T_local = input_seq_local.shape[1]

        # optionally normalize input to reduce scale effects
        if mean_cols is not None and std_cols is not None:
            inp = (input_seq_local - mean_cols.to(device_local)) / std_cols.to(device_local)
        else:
            inp = input_seq_local

        # ensure a fresh tensor that requires grad
        inp = inp.detach().clone().requires_grad_(True)

        outs = model(inp).squeeze(0)  # (T, P)

        GCs_local = torch.zeros((P_local, P_local), device=device_local)

        # compute per-output gradients
        # Note: this loop does P backward/grad calls; for modest P this is fine and precise.
        for j in range(P_local):
            out_j = outs[:, j]        # (T,)
            s_j = out_j.sum()         # scalar
            grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # aggregate across time dim: mean or max
            if lag_agg_local == 'mean':
                gc_row = grads.abs().mean(dim=1).squeeze(0)   # (P,)
            elif lag_agg_local == 'max':
                gc_row = grads.abs().max(dim=1)[0].squeeze(0) # (P,)
            else:
                # default to mean
                gc_row = grads.abs().mean(dim=1).squeeze(0)

            GCs_local[j, :] = gc_row

        return GCs_local  # shape (P, P)

    # ---------------------------------------------------------------

    # training loop
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        causal_enricher_fwd.train()
        causal_enricher_rev.train()
        optimizer.zero_grad()

        # optionally normalize inputs for forward/backward pass (same scheme as GC)
        if mean_cols is not None and std_cols is not None:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        # --- 1) initial pass to compute GCs (use compute_gradient_gc_for_model which itself runs model on a fresh requires_grad input) ---
        # We call compute_gradient_gc_for_model with create_graph according to gc_create_graph flag (that's your original design)
        if (i % gc_every) == 0:
            # compute with graph if requested
            GCs = compute_gradient_gc_for_model(model_fwd, inp_fwd, create_graph=gc_create_graph,
                                                lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_for_model(model_rev, inp_rev, create_graph=gc_create_graph,
                                                 lag_agg_local=lag_agg)
        else:
            with torch.no_grad():
                GCs = compute_gradient_gc_for_model(model_fwd, inp_fwd, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_for_model(model_rev, inp_rev, create_graph=False, lag_agg_local=lag_agg)

        # detach the GC matrices before using them to enrich inputs (avoid second-order / expensive bpy)
        GCs_det = GCs.detach()
        GC2s_det = GC2s.detach()

        # --- 2) create enriched inputs using causal attention/enricher ---
        enriched_inp_fwd = causal_enricher_fwd(inp_fwd, GCs_det)  # (1, T, P)
        enriched_inp_rev = causal_enricher_rev(inp_rev, GC2s_det)  # (1, T, P)

        # --- 3) forward with enriched inputs and compute predictions/losses ---
        outs_fwd_all = model_fwd(enriched_inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(enriched_inp_rev).squeeze(0)  # (T, P)

        # per-target losses (same as before)
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # --- compute fused alphas and fused GC tensor (和你原先逻辑一致) ---
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                           target_seq, reversed_target_seq,
                                           None, losses_fwd, losses_rev)
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

        loss = (predict_loss1 + predict_loss2
                - lambda_alpha_reg * alpha_reg
                + Lsparse1 + Lsparse2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # EMA update for alphas 等你原本逻辑保持不变...
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        # monitor / tracking (与你之前的打印一致)
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

    # 最后画图（与你原有 plot_best_gc_heatmap 一致）
    plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused,threshold=0.1)
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
def plot_best_gc_heatmap_new(GC_true, GCs_best, GC2s_best, fused_best, save_path=None):
    """
    绘制 ROC 最优时的 GC 热力图（2x2 排布）。
    自动将矩阵归一化到 0–1，避免二值化的视觉效果。
    """

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # --- 转 numpy ---
    GC_true = to_np(GC_true)
    GCs_best = to_np(GCs_best)
    GC2s_best = to_np(GC2s_best)
    fused_best = to_np(fused_best)

    # --- 标准化，使热力图连续 ---
    def normalize(m):
        m = m.copy()
        mn, mx = m.min(), m.max()
        if mx - mn < 1e-12:
            return np.zeros_like(m)
        return (m - mn) / (mx - mn)

    GCs_best = normalize(GCs_best)
    GC2s_best = normalize(GC2s_best)
    fused_best = normalize(fused_best)

    # --- 画图 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    titles = ["Ground Truth", "Forward GC", "Reverse GC", "Fused GC"]
    matrices = [GC_true, GCs_best, GC2s_best, fused_best]

    for ax, mat, title in zip(axes, matrices, titles):
        sns.heatmap(mat, ax=ax, cmap="YlGnBu", square=True, cbar=True, vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Cause variable (X_i)", fontsize=10)
        ax.set_ylabel("Effect variable (Y_j)", fontsize=10)
        ax.tick_params(axis='x', labelrotation=90, labelsize=8)
        ax.tick_params(axis='y', labelrotation=0, labelsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
def plot_best_gc_heatmap_thre(GC_true, GCs_best, GC2s_best, fused_best,
                         save_path=None, threshold=0.6):
    """
    绘制 ROC 最优时的 GC 热力图（2x2 排布）
    如果 threshold 不为 None，则执行二值化，大于阈值为 1，小于等于为 0。
    """

    def to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # --- 转 numpy ---
    GC_true = to_np(GC_true)
    GCs_best = to_np(GCs_best)
    GC2s_best = to_np(GC2s_best)
    fused_best = to_np(fused_best)

    # --- 阈值二值化（如果设置了 threshold） ---
    if threshold is not None:
        print(f"Applying threshold={threshold} to GC matrices...")
        GCs_best = (GCs_best > threshold).astype(float)
        GC2s_best = (GC2s_best > threshold).astype(float)
        fused_best = (fused_best > threshold).astype(float)
        # Ground truth 不做 threshold（一般是 0/1）
        # GC_true = (GC_true > threshold).astype(float)

    # --- 画图 ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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

from sklearn.preprocessing import StandardScaler
# def infer_Grangercausality_1219(P, type, epoch, hidden_size, learning_rate, lam):
#
#     global_seed = 1
#     torch.manual_seed(global_seed)
#     torch.cuda.manual_seed(global_seed)
#     np.random.seed(global_seed)
#
#     GC, X = read_dream4(P, type)
#     GC = off_diagonal(GC)
#
#     length = X.shape[0]
#
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#
#     test_x = X[:length-1, :]
#     test_y = X[1:length, :]
#
#     input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda(1)
#     target_seq = torch.tensor(test_y, dtype=torch.float32).cuda(1)
#
#     X2 = np.ascontiguousarray(X[::-1, :])
#     reversed_x = X2[:length - 1, :]
#     reversed_y = X2[1:length, :]
#     reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda(1)
#     reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda(1)
#
#     # model = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
#     model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
#     model_rev = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
#     fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)
#     loss_fn = nn.MSELoss()
#     # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     optimizer = torch.optim.Adam([
#         {'params': list(model_fwd.parameters()), 'lr': learning_rate},
#         {'params': list(model_rev.parameters()), 'lr': learning_rate},
#         {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
#     ])
#
#     for i in range(epoch):
#         losses = []
#         input_current = input_seq.squeeze(0).detach().clone().requires_grad_().to(device)
#         output_current = model(input_current.unsqueeze(0)).squeeze(0)
#         predict_loss = loss_fn(output_current, target_seq)
#
#         grad_input_matrix = torch.zeros((P, P)).to(device)
#         for output_idx in range(P):
#             model.zero_grad()
#             out = output_current[:, output_idx].sum()
#             grad = torch.autograd.grad(out, input_current, retain_graph=True, create_graph=True)[0]
#             grad_input = grad.abs().mean(dim=0)
#             loss_i = grad_input.abs().sum()
#             losses.append(loss_i)
#             grad_input_matrix[output_idx] = grad_input
#
#         L1_loss = lam * sum(losses)
#
#         GC_est = grad_input_matrix.detach().cpu().numpy()
#
#         GC_est = off_diagonal(GC_est)
#
#         score_gi, aupr_gi = compute_roc_1219(GC, GC_est, False)
#
#         total_loss = predict_loss + L1_loss
#         optimizer.zero_grad()
#         total_loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 1 == 0:
#             print(f"Epoch {i+1}/{epoch} | Loss: {total_loss:.4f} | L1_loss:{L1_loss:.4f} | "
#                   f" AUROC: {score_gi:.4f} | AUPRC: {aupr_gi:.4f}")

# no good
def infer_Grangercausalityv4_inge_plus_try_tosparse_1219gradtry(
        P, type, epoch, hidden_size, learning_rate,
        lambda_alpha_reg,
        lambda_gc_sparse_base, lambda_gc_sparse_fusion,
        gc_create_graph=True, gc_every=1,
        lag_agg='mean', normalize_input=False,
        # --- 新增/调整的超参数 ---
        cutoff_ratio=0.2,
        grad_clip_quantile=0.99,  # 动态裁剪的分位数
        lambda_gc_grad=1e-4,      # 新增：梯度正则的权重（method-1）
        use_grad_freq_denoise=False # 新增：对梯度做频域去噪（可选）
):
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import pandas as pd

    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_auprc = 0.0

    # ---------------- 数据准备（与你原逻辑一致） ----------------
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)
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

    mean_cols, std_cols = None, None
    if normalize_input:
        mean_cols = input_seq.mean(dim=1, keepdim=True)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8

    model_fwd = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()

    # ---------- 内部辅助函数：频域去噪（与前面你已有实现语义一致） ----------
    def frequency_domain_denoise_local(grads_abs, cutoff_ratio_local=0.2):
        # grads_abs: (1, T, P)
        # 返回 (1, T, P)
        g = grads_abs.transpose(1, 2)  # (1, P, T)
        g_fft = torch.fft.rfft(g, dim=-1)
        n_freq = g_fft.shape[-1]
        cutoff = max(1, int(n_freq * cutoff_ratio_local))
        mask = torch.zeros_like(g_fft)
        mask[..., :cutoff] = 1.0
        g_fft_filtered = g_fft * mask
        g_ifft = torch.fft.irfft(g_fft_filtered, n=g.shape[-1], dim=-1)
        return g_ifft.transpose(1, 2).abs()  # (1, T, P)

    # ---------- 内部辅助函数：计算方法一的梯度正则 L_gc_grad ----------
    def compute_gradient_reg_term(model_local, input_seq_local,
                                  use_freq=use_grad_freq_denoise,
                                  cutoff_ratio_local=cutoff_ratio):
        """
        返回: L_grad (scalar tensor, 可反向传播), grad_matrix (P x P) for logging
        说明：这里使用 create_graph=True 来保证 L_grad 能反传到 model 参数
        """
        device_local = input_seq_local.device
        # 复制输入，使其 requires_grad=True（只会对这个临时副本求导）
        inp = input_seq_local.detach().clone().requires_grad_(True)  # (1, T, P)
        outs = model_local(inp).squeeze(0)  # (T, P)
        P_local = outs.shape[1]
        T_local = outs.shape[0]

        grad_matrix = torch.zeros((P_local, P_local), device=device_local)
        losses_list = []

        # 对每个输出 j 求关于整个输入序列 inp 的梯度（s_j 是标量）
        for j in range(P_local):
            s_j = outs[:, j].sum()  # aggregate over time -> scalar
            grads = torch.autograd.grad(
                s_j, inp, create_graph=True, retain_graph=True
            )[0]  # (1, T, P)

            grads_abs = grads.abs()  # (1, T, P)

            if use_freq:
                grads_abs = frequency_domain_denoise_local(grads_abs, cutoff_ratio_local=cutoff_ratio_local)
            else:
                # 简单 3 点滑动平均（沿时间轴）
                kernel = torch.tensor([0.25, 0.5, 0.25], device=device_local).view(1, 1, 3)
                g = grads_abs.permute(0, 2, 1)  # (1, P, T)
                g = torch.nn.functional.pad(g, (1, 1), mode='reflect')
                g = torch.nn.functional.conv1d(g, kernel.repeat(P_local, 1, 1), groups=P_local)
                grads_abs = g.permute(0, 2, 1)  # (1, T, P)

            # 聚合时间维度 -> 对每个输入变量 i 得到平均敏感度
            grad_input_i = grads_abs.mean(dim=1).squeeze(0)  # (P,)
            # 归一化（避免不同输出尺度影响），这里用 max 归一化
            denom = grad_input_i.max().clamp_min(1e-8)
            grad_input_i_norm = grad_input_i / denom

            # 保存到矩阵（用于日志/裁剪判断）
            grad_matrix[j, :] = grad_input_i.detach().clone()

            # 聚合成一个标量损失（L1）, 也可以尝试 L2: (grad_input_i_norm ** 2).sum()
            losses_list.append(grad_input_i_norm.abs().sum())

        # 合并所有输出的损失，乘以权重
        L_grad = lambda_gc_grad * sum(losses_list)
        return L_grad, grad_matrix.detach()

    # -------------------- 训练循环 --------------------
    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()

        if normalize_input:
            inp_fwd = (input_seq - mean_cols.to(device)) / std_cols.to(device)
            inp_rev = (reversed_input_seq - mean_cols.to(device)) / std_cols.to(device)
        else:
            inp_fwd = input_seq
            inp_rev = reversed_input_seq

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)

        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]

        predict_loss1 = sum(losses_fwd)  # L_fwd
        predict_loss2 = sum(losses_rev)  # L_rev

        # 2. GC 矩阵计算 (平滑 + 裁剪) -- 保持你原来的推断式实现
        if (i % gc_every) == 0:
            GCs_raw = compute_gradient_gc_smooth_universal_v2(
                model_fwd, input_seq, cutoff_ratio=cutoff_ratio,
                create_graph=gc_create_graph, lag_agg_local=lag_agg
            )
            GC2s_raw = compute_gradient_gc_smooth_universal_v2(
                model_rev, reversed_input_seq, cutoff_ratio=cutoff_ratio,
                create_graph=gc_create_graph, lag_agg_local=lag_agg
            )

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
            with torch.no_grad():
                GCs = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq, create_graph=False,
                                                              lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_smooth_universal_v2(model_rev, reversed_input_seq, create_graph=False,
                                                               lag_agg_local=lag_agg)

        # ------------------ 新增：基于梯度的训练期正则（method-1） ------------------
        # 只有在 gc_every 步上计算以节省计算/内存（你也可以改成每 step 计算）
        if (i % gc_every) == 0 and lambda_gc_grad > 0.0:
            # 注意：compute_gradient_reg_term 内部使用 create_graph=True -> L_grad 可反传
            L_grad_fwd, grad_mat_fwd = compute_gradient_reg_term(model_fwd, input_seq,
                                                                 use_freq=use_grad_freq_denoise,
                                                                 cutoff_ratio_local=cutoff_ratio)
            L_grad_rev, grad_mat_rev = compute_gradient_reg_term(model_rev, reversed_input_seq,
                                                                 use_freq=use_grad_freq_denoise,
                                                                 cutoff_ratio_local=cutoff_ratio)
        else:
            # 仅用于监控，不参与反传
            with torch.no_grad():
                if lambda_gc_grad > 0.0:
                    _, grad_mat_fwd = compute_gradient_reg_term(model_fwd, input_seq,
                                                                use_freq=False, cutoff_ratio_local=cutoff_ratio)
                    _, grad_mat_rev = compute_gradient_reg_term(model_rev, reversed_input_seq,
                                                                use_freq=False, cutoff_ratio_local=cutoff_ratio)
                L_grad_fwd = torch.tensor(0.0, device=device)
                L_grad_rev = torch.tensor(0.0, device=device)

        # ---------------- 融合、熵正则、稀疏正则 ----------------
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        if lambda_gc_sparse_fusion > 0.0 and ((i % gc_every) == 0):
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse_fused = torch.tensor(0.0, device=device)

        # ----------------- 总损失合并（包含梯度正则项） -----------------
        loss = (predict_loss1 + predict_loss2
                - lambda_alpha_reg * alpha_reg
                 + Lsparse_fused
                + L_grad_fwd + L_grad_rev
                )

        # 反向传播与优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # 评估与记录（与你原来逻辑一致，输出融合评估）
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1, auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            score2, auprc2 = compute_roc_1219(GC, off_diagonal(GC2s_np), False)
            score_fusion, auprc_fusion = compute_roc_1219(GC, off_diagonal(fused_np), False)

            if best_auprc < auprc_fusion:
                best_auprc = auprc_fusion
            if best_score < score_fusion:
                best_score = score_fusion

        print(
            f'Epoch [{i + 1}/{epoch}], L_pred1: {predict_loss1.item():.4f}, L_pred2: {predict_loss2.item():.4f}, '
            f' L_sparse_fused: {Lsparse_fused.item():.4f}, L_grad_fwd: {float(L_grad_fwd.detach().cpu().numpy()):.6f}, '
            f'ROC_fwd: {score1:.4f}, ROC_rev: {score2:.4f}, ROC_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    return best_score, best_auprc
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_1220(P, type, epoch, hidden_size, learning_rate,
                                  lambda_alpha_reg,
                                  lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean', normalize_input=False,
                                  # --- 新增/调整的超参数 ---
                                  cutoff_ratio=0.2,
                                  grad_clip_quantile=0.99  # 动态裁剪的分位数
                                  ):
    """
    说明: 1211 优化版本。
    - 集成动态梯度裁剪 (基于分位数)。
    - 集成基于预测损失的 Alpha 置信度正则化 (L_confidence)。
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import pandas as pd

    # ---------------- 准备工作 (略去重复的初始化和数据加载) ----------------
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    # 假设 read_dream4, KAN, FusionEdge, compute_roc, compute_auprc,
    # off_diagonal, compute_edge_features_1205 已在 scope 内或已导入
    best_score = 0
    best_auprc = 0.0

    # 模拟数据加载（使用您的原始逻辑）
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)
    length = X.shape[0]

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    mean_cols, std_cols = None, None
    if normalize_input:
        mean_cols = input_seq.mean(dim=1, keepdim=True)
        std_cols = input_seq.std(dim=1, keepdim=True) + 1e-8

    # 模型构建 (KAN, FusionEdge)
    model_fwd = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    model_rev = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)
    fusion_edge = FusionEdge(in_dim=10, hidden=16).to(device)

    params = list(model_fwd.parameters()) + list(model_rev.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
        {'params': list(model_rev.parameters()), 'lr': learning_rate},
        {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
    ])

    loss_fn = nn.MSELoss()
    # ---------------- 训练循环 ----------------

    for i in range(epoch):
        model_fwd.train()
        model_rev.train()
        fusion_edge.train()
        optimizer.zero_grad()


        inp_fwd = input_seq
        inp_rev = reversed_input_seq

        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)

        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        losses_rev = [loss_fn(outs_rev_all[:, j], reversed_target_seq[:, j]) for j in range(P)]

        predict_loss1 = sum(losses_fwd)  # L_fwd
        predict_loss2 = sum(losses_rev)  # L_rev

        # 2. GC 矩阵计算 (使用通用的平滑函数)
        if (i % gc_every) == 0:
            # ❗ 步骤 2.1: 计算 GCs_raw (平滑后，未裁剪)
            GCs_raw = compute_gradient_gc_smooth_universal_v3(model_fwd, input_seq,cutoff_ratio=cutoff_ratio,
                                                           create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s_raw = compute_gradient_gc_smooth_universal_v3(model_rev, reversed_input_seq,cutoff_ratio=cutoff_ratio,
                                                            create_graph=gc_create_graph, lag_agg_local=lag_agg)
            # GCs_raw = compute_gradient_gc_simple(model_fwd, input_seq)
            # GC2s_raw = compute_gradient_gc_simple(model_rev, reversed_input_seq)

            # ❗ 步骤 2.2: 动态裁剪
            # 裁剪阈值取两个 GC 矩阵的梯度绝对值之和的 Q_99
            with torch.no_grad():
                combined_gc_values = torch.cat([GCs_raw.view(-1), GC2s_raw.view(-1)])
                # 忽略 0 值，防止稀疏矩阵对分位数的干扰
                nonzero_combined = combined_gc_values[combined_gc_values > 1e-6]
                if nonzero_combined.numel() > 0:
                    clip_threshold = torch.quantile(nonzero_combined, grad_clip_quantile)
                else:
                    clip_threshold = 1.0  # 默认值

            # 应用裁剪
            GCs = GCs_raw.clamp(max=clip_threshold)
            GC2s = GC2s_raw.clamp(max=clip_threshold)

            # 由于裁剪是在训练图上进行的，Lsparse 仍然可以反传
        else:
            # Monitoring only (no graph)
            print('no use')
            with torch.no_grad():
                GCs = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq, create_graph=False,
                                                           lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_smooth_universal_v2(model_rev, reversed_input_seq, create_graph=False,
                                                            lag_agg_local=lag_agg)
                # Note: 此时不对 GCs, GC2s 进行裁剪，仅用于评估/展示

        # 3. 融合和正则化
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev_list = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd_list, outs_rev_list,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd, losses_rev)
        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        eps = 1e-8
        # 熵正则项
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        # ❗ 创新点 2: 基于预测损失的 Alpha 置信度正则化 L_confidence
        # L_confidence = lambda_conf * ((1-alpha) * L_fwd + alpha * L_rev).mean()
        # 由于 L_fwd/L_rev 是标量，需要广播
        # L_confidence = lambda_confidence * (
        #         (1.0 - alphas) * predict_loss1 + alphas * predict_loss2
        # ).mean()

        # GC 稀疏化损失
        if lambda_gc_sparse_base > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse_base * torch.abs(GC2s).sum()
            Lsparse_fused = lambda_gc_sparse_fusion * torch.abs(fused_GC_tensor).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)
            Lsparse_fused = torch.tensor(0.0, device=device)

        # 4. 组合总损失
        loss = (predict_loss1 + predict_loss2
                - lambda_alpha_reg * alpha_reg
                + Lsparse1 + Lsparse2 + Lsparse_fused
                )  # ❗ 新增置信度损失项

        # 5. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # 6. 评估和记录
        # ... (评估和记录逻辑与原代码保持一致)
        # 评估和记录部分逻辑与原代码保持一致
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()

        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            # score1 = compute_roc(GC, off_diagonal(GCs_np), False)
            # score2 = compute_roc(GC, off_diagonal(GC2s_np), False)
            # score_fusion = compute_roc(GC, off_diagonal(fused_np), False)
            score1,auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            score2,auprc2 = compute_roc_1219(GC, off_diagonal(GC2s_np), False)
            score_fusion,auprc_fusion = compute_roc_1219(GC, off_diagonal(fused_np), False)

            # auprc1 = compute_auprc(GC, off_diagonal(GCs_np), False)
            # auprc2 = compute_auprc(GC, off_diagonal(GC2s_np), False)
            # auprc_fusion = compute_auprc(GC, off_diagonal(fused_np), False)

            # Update best by AUPRC fusion primarily, then ROC fusion
            if best_auprc < auprc_fusion:
                best_auprc = auprc_fusion
                # ... save best GCs/GC2s/fused ...
            if best_score < score_fusion:
                best_score = score_fusion
                # ... save best GCs/GC2s/fused ...

        print(
            f'Epoch [{i + 1}/{epoch}], L_pred1: {predict_loss1.item():.4f}, L_pred2: {predict_loss2.item():.4f}, '
            f' L_sparse_fused: {Lsparse_fused.item():.4f}, '
            f'ROC_fwd: {score1:.4f}, ROC_rev: {score2:.4f}, ROC_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    return best_score, best_auprc




















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
    use_first_order_prior=False,
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
            print('no use fre')
            pass
            # g = grads_abs.permute(0, 2, 1)
            # g = F.pad(g, (1, 1), mode='reflect')
            # g = F.conv1d(g, kernel, groups=P)
            # grads_abs = g.permute(0, 2, 1)

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
def infer_Grangercausalityv4_inge_plus_try_tosparse_1228(P, type, epoch, hidden_size, learning_rate,
                                                         lambda_gc_sparse_base,
                                                         gc_create_graph=True, gc_every=1,
                                                         lag_agg='mean',
                                                         cutoff_ratio=0.2,
                                                         tau=10.0,  # 新增参数，用于 CSDRF 细化
                                                         adj_threshold=0.05,
                                                         C_plus=1.2,  # 根据曲率平衡需求调整
                                                         grad_clip_quantile=0.99,  # 动态裁剪的分位数
                                                         ):
    """
    改写后：
    - 使用两个 multi-output 模型（正序 model_fwd 和 反序 model_rev），每个模型输出 P 个序列。
    - compute_gradient_gc_for_model 计算单个 multi-output 模型的 (P, P) 梯度 GC 矩阵。
    新增可选参数（非必要改变签名，只是内部可选）:
      - lag_agg: 'mean' 或 'max'（时间维度聚合方法）
      - normalize_input: 是否先对 input_seq 做按变量（col）标准化
    其它行为与原先保持一致（fusion, alpha, sparse 等）。
    """
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    best_auprc = 0.0
    best_auprc1 = 0.0
    best_auprc2 = 0.0
    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)     # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)                # (T, P)

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    total_params = sum(p.numel() for p in model_fwd.parameters())
    trainable_params = sum(p.numel() for p in model_fwd.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
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

        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model_fwd,
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
        score1, auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
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
    # for i in [1,2,3,4,5]:
    for params in param_list:
        print(f"Training with params: {params}")

        # avg_score = infer_Grangercausalityv4(100,1, 300, hidden_size=params['hidden_size'], lam=params['lam'],
        #                                    lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
        #                                    , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                     lambda_consistency=params['lambda_consistency'])
        # avg_score = infer_Grangercausalityv4_inge(100,1, 150, hidden_size=params['hidden_size'], lam=params['lam'],
        #                                    lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
        #                                    , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                     lambda_consistency=params['lambda_consistency'])




        # 2025 12 8  muqianzuihao danshi queshaochuangxin
        # avg_score = infer_Grangercausalityv4_inge_plus_try(100, 5, 150, hidden_size=params['hidden_size'], learning_rate=params['learning_rate']
        #                                      , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                     lambda_gc_sparse=params['lambda_gc_sparse'])



        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1228(100, 1, 300, hidden_size=params['hidden_size'],
                                                           learning_rate=params['learning_rate'],
                                                lambda_gc_sparse_base = params['lambda_gc_sparse_base'], lag_agg = params['lag_agg'])
        # avg_score = infer_Grangercausality(100, 1, 300, hidden_size=params['hidden_size'], lam=params['lam'],
        #                                      lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate'])



        # avg_score = infer_Grangercausality_1219(100, 2, 300, hidden_size=params['hidden_size'],
        #                                                                  learning_rate=params['learning_rate']
        #                                                                  , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                                                  lambda_gc_sparse_fusion=params['lambda_gc_sparse_fusion'],lam=params['lam'])
        # infer_Grangercausality_1219(
        #     P=100,
        #     type=1,
        #     epoch=200,
        #     hidden_size=128,
        #     learning_rate=0.01,
        #     lam=0.008,
        #     lambda_alpha_reg=0.01,
        #     lambda_gc_sparse_fusion=0.008,
        # )

        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))

    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc


if __name__ == '__main__':
    # param_grid = {
    #     'hidden_size': [256],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.005]
    # }  ###  0.734,0.658,0.555,0.555,0.521

    # param_grid = {
    #     'hidden_size': [256],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.001]
    # } ###  0.734,0.673,0.59,0.53,0.566
  #v GC ronghe   infer_Grangercausalityv4_inge 5 Best params: {'hidden_size': 128, 'lam': 0.001, 'lam_ridge': 1.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.1, 'learning_rate': 0.001} with avg score: 0.6321675880500744

    # param_grid = {
    #     'hidden_size': [256],  ##128   orgin256
    #     'lam': [0.01],
    #     'lam_ridge': [5.0],
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [5.0],
    #     'lambda_consistency': [0.05]
    # }
   #111 try   256 0.01 5 0.001   10   0.05   0.005

    #2025 12 05 feature1205 1 0.79  2 0.659 3 0.637  4 0.67  5 0.667   5 0.07768  4 0.0808  3 0.1238   2  0.0885  1  0.1608
    # 2025 12 08 add fusion_spare 约束  1 0.78884 0.2299  2  0.6415  0.0993  3   0.64503  0.14483  4  0.65983  0.0873  5  0.66553 0.077279
    # """ fusion sparse canshu
    # 'hidden_size': [512],  ##128   orgin256
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.01],
    #     'lambda_consistency': [0.05],
    #     # 'lambda_gc_sparse': [0.005],
    # """ 有一说一 好像分层稀疏约束真有说法 但是我不知道是什么说法  好像是我之前的gc_sparse太大了。。。  2   0.6751  0.1034
    # param_grid = {
    #     'hidden_size': [512],  ##128   orgin256
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.001],
    #     'lambda_consistency': [0.05],
    #     # 'lambda_gc_sparse': [0.005],
    #     'lambda_gc_sparse_base': [0.001],
    #     'lambda_gc_sparse_fusion': [0.001]
    # }


    #  我先注释掉 试一下信给我的代码实现  2025 12 19
    # param_grid = {
    #     'hidden_size': [512],  ##128   orgin256
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.01],
    #     'lambda_consistency': [0.05],
    #     'cutoff_ratio': [0.4],
    #     # 'lambda_gc_sparse': [0.005],
    #     'lambda_gc_sparse_base': [0.001],# 0.0012
    #     'lambda_gc_sparse_fusion': [0.0008] #  0.0008
    # }
    # 12-18 test Training with params: {'cutoff_ratio': 0.2, 'hidden_size': 256, 'lambda_alpha_reg': 0.01,
    # 'lambda_consistency': 0.05, 'lambda_gc_sparse_base': 0.0012, 'lambda_gc_sparse_fusion': 0.0008, 'learning_rate': 0.001}

    param_grid = {
        'hidden_size': [512],  ##128   orgin256
        'learning_rate': [0.001],  # 0.005 0.001
        'lambda_gc_sparse_base': [0.008],  #
        'cutoff_ratio': [0.6],
        'lag_agg': ['softmax'],
    }
   # 1: Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.8004217540480909
    # Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.2945597332827147
    # 2:Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.6891214219638891
    # Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.15555267738931208
    # 3:Best params: {'cutoff_ratio': 0.8, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.6792427905256344
    # Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.2068616688587608
    #4:Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.7015524029546382
    #Best params: {'cutoff_ratio': 0.8, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.001} with avg score: 0.16665523976716018
    #5:Best params: {'cutoff_ratio': 0.8, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.001} with avg score: 0.7228158089002594
    #Best params: {'cutoff_ratio': 0.4, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.002, 'learning_rate': 0.001} with avg score: 0.16196828429436952



   # v1
    # 1:Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.01} with avg score: 0.8021156650835795
    # Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.277769046513611
    # 2:  Best params: {'cutoff_ratio': 0.6, 'hidden_size': 256, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.01} with avg score: 0.7263849720714795
    # Best params: {'cutoff_ratio': 0.6, 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.01} with avg score: 0.15070850202105857
    #3:Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.01} with avg score: 0.6979690616784899
    #  Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.19362228300963633
     #4:Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.7018375751267256
     #    Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.01} with avg score: 0.1403577663174371
     #5:Best params: {'cutoff_ratio': 0.8, 'hidden_size': 512, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.001, 'learning_rate': 0.01} with avg score: 0.7246274388815079    0.12730693182307742
     #Best params: {'cutoff_ratio': 0.6, 'hidden_size': 512, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.008, 'learning_rate': 0.001} with avg score: 0.1318506830679595



















    # 2025.11.15 try  256 0.01   5   0.001   5   0.05   0.005   1： 0.78or0.76  2：0.76 3： 0.63  4： 0.67  5： 0.66

    # 111 try 2 256 0.01 5 0.001   10   0.05   0.005
    # param_grid = {
    #     'hidden_size': [256,128],  ##128   orgin256
    #     'learning_rate': [0.01,0.001]  # 0.005
    #     , 'lambda_alpha_reg': [5.0,10],
    #     'lambda_gc_sparse': [0.005,0.001]
    # }





    #    yakebi  infer_Grangercausalityv4_inge   1   Best params: {'hidden_size': 256, 'lam': 0.001, 'lam_ridge': 10, 'lambda_alpha_reg': 5.0, 'lambda_consistency': 0.01, 'learning_rate': 0.01} with avg score: 0.7739648386372986
    # param_grid = {
    #     'hidden_size': [128, 256],  ##128   orgin256
    #     'lam': [0.01, 0.001],
    #     'lam_ridge': [5.0, 1.0],
    #     'learning_rate': [0.001, 0.01]  # 0.005
    #     , 'lambda_alpha_reg': [10.0, 5.0, 1.0],
    #     'lambda_consistency': [0.05, 0.1]
    # }
    # Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 10.0, 'lambda_alpha_reg': 3.0, 'lambda_consistency': 0.01, 'learning_rate': 0.01} with avg score: 0.7614454395871508  1
   #  Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 5.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.5, 'learning_rate': 0.001} with avg score: 0.5801024011079028  2
    # Best params: {'hidden_size': 128, 'lam': 0.01, 'lam_ridge': 5.0, 'lambda_alpha_reg': 3.0, 'lambda_consistency': 0.05, 'learning_rate': 0.001} with avg score: 0.6111060912297389   3
    # Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 10.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.5, 'learning_rate': 0.001} with avg score: 0.5984091012478605  4

    # Best  5   infer_Grangercausalityv4   0.6886366390153786  Integrated Gradients
    # params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 10, 'lambda_alpha_reg': 3.0, 'lambda_consistency': 0.01,
    #          'learning_rate': 0.01}
    # with avg score: 0.6886366390153786
    # Best   4    infer_Grangercausalityv4   0.6886366390153786  Integrated Gradients
    # params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 10, 'lambda_alpha_reg': 3.0, 'lambda_consistency': 0.01,
    #          'learning_rate': 0.01}
    # with avg score: 0.6791416855680869

    #  3   infer_Grangercausalityv4
    # score_fusion: 0.6655

    # 2  infer_Grangercausalityv4
    # 0.6624745796989637

    # 1 infer_Grangercausalityv4
    # 0.806441302681276
    # param_grid = {
    #     'hidden_size': [600],
    #     'lam': [0.0001],
    #     'lam_ridge': [0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,16,20],
    #     'learning_rate': [0.01]
    # }

    # param_grid = {
    #     'hidden_size': [256],  ##128   orgin256
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.01],
    #     'lambda_consistency': [0.05],
    #     'lam': [0.08],
    #     'lambda_gc_sparse_fusion': [0.08] #  0.0008
    # }
    best_params = grid_search(param_grid)
    # infer_Grangercausality_1219(
    #     P=100,
    #     type=1,
    #     epoch=200,
    #     hidden_size=128,
    #     learning_rate=0.01,
    #     lam=0.008,
    #     lambda_alpha_reg=0.01,
    #     lambda_gc_sparse_fusion=0.008,
    # )
