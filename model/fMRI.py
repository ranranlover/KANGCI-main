import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import fmri_read
import time

# Set seed for random number generation (for reproducibility of results)

global_seed = 1
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
np.random.seed(global_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ridge_regularize(model, lam_ridge):
    total_weight_sum = 0
    for layer in model.layers[1:]:
        weight_squared_sum = torch.sum(layer.base_weight ** 2)
        total_weight_sum += weight_squared_sum
    result = lam_ridge * total_weight_sum
    return result


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
    elif penalty == 'row+column':
        return lam * (torch.sum(torch.norm(x, dim=0))
                      + torch.sum(torch.norm(x, dim=1)))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def infer_Grangercausality(simulation, subject, hidden_size, epoch, lam, lam_ridge, learning_rate):
    best_score = 0
    best_score1 = 0
    best_score2 = 0

    X, GC, length = fmri_read(simulation, subject)
    P = X.shape[1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    # reversed data
    X2 = X[::-1, :]
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda()
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda()

    # component-wise generate p models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, 600, 64, 32, 1], base_activation=nn.Identity).to(device)
        networks.append(network)

    models = nn.ModuleList(networks)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)

    for i in range(epoch):
        start_time = time.time()
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

        loss = predict_loss1 + predict_loss2 + regularize_loss1 + regularize_loss2 + ridge_loss1 + ridge_loss2

        GCs = []
        GC2s = []
        for k in range(P):
            GCs.append(models[k].GC().detach().cpu().numpy())
        GCs = np.array(GCs)

        for k in range(P, 2 * P):
            GC2s.append(models[k].GC().detach().cpu().numpy())
        GC2s = np.array(GC2s)

        if predict_loss1 < predict_loss2 and regularize_loss1 < regularize_loss2:
            result = GCs
        elif predict_loss1 > predict_loss2 and regularize_loss1 > regularize_loss2:
            result = GC2s
        else:
            result = np.where(
                np.abs(GCs - GC2s) < 0.05,
                (GCs + GC2s) / 2,
                np.maximum(GCs, GC2s)
            )

        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time

        if best_score < score_fusion:
            best_score = score_fusion
            best_score1 = score1
            best_score2 = score2
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}'
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}')

    return best_score, best_score1, best_score2
class FusionEdge(nn.Module):
    """
    Per-edge fusion network: map per-edge features -> alpha_ij in (0,1).
    Vectorized: accepts (P*P, feat_dim) and outputs (P*P,1) which we reshape to (P,P).
    """
    def __init__(self, in_dim=5, hidden=64):
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
def infer_Grangercausalityv1(simulation, subject, epoch, hidden_size, lam, lam_ridge, learning_rate):
    # reproducibility & device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    score = 0
    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    X, GC ,length= fmri_read(simulation, subject)
    P = X.shape[1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    test_x = X[:length - 1,:]
    test_y = X[1:length,:]
    # use device variable defined at top of your file
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
        network = KAN([P, 600, 64, 32, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    # instantiate per-edge fusion net
    fusion_edge = FusionEdge(in_dim=5, hidden=64).to(device)

    # optimizer includes fusion params
    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
    {'params': list(models.parameters()), 'lr': learning_rate},
    {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.01}
])


    loss_fn = nn.MSELoss()


    for i in range(epoch):
        start_time = time.time()
        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # forward pass: predictions for each single-target model (forward & reverse)
        outs_fwd = []
        losses_fwd = []
        for j in range(0, P):
            out_j = models[j](input_seq).view(-1)
            outs_fwd.append(out_j)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))

        outs_rev = []
        losses_rev = []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            outs_rev.append(out_j)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # regularizers (same as before)
        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # --- compute GCs as tensors (shape (P,P)) ---
        # assume model.GC() returns a torch tensor on the model's device
        GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)       # (P,P)
        GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0) # (P,P)

        # --- build per-edge features for fusion_edge (vectorized) ---
        # features: [g_fwd, g_rev, absdiff, row_mean_absdiff, pred_loss_ratio]
        # compute scalar per-row diff and pred loss ratio
        row_absdiff = torch.mean(torch.abs(GCs - GC2s), dim=1)  # (P,)
        # prepare per-edge tensors
        # Expand to (P,P) then flatten to (P*P,)
        g_fwd = GCs.view(-1)            # (P*P,)
        g_rev = GC2s.view(-1)
        absdiff = torch.abs(g_fwd - g_rev)
        # row_mean_absdiff mapped to each edge in that row
        row_mean_rep = row_absdiff.unsqueeze(1).repeat(1, P).view(-1)
        # pred loss ratio per row (use detach to avoid loops)
        loss_fwd_vec = torch.stack([l.detach() for l in losses_fwd])  # (P,)
        loss_rev_vec = torch.stack([l.detach() for l in losses_rev])
        pred_ratio = (loss_fwd_vec / (loss_rev_vec + 1e-12)).unsqueeze(1).repeat(1, P).view(-1)

        # stack features to (P*P, feat_dim)
        feat_edges = torch.stack([g_fwd.detach(), g_rev.detach(), absdiff.detach(), row_mean_rep.detach(), pred_ratio.detach()], dim=1)
        # numeric stability / scaling
        feat_edges = torch.log1p(torch.abs(feat_edges))
        feat_mean = feat_edges.mean(dim=0, keepdim=True)
        feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
        feat_edges = (feat_edges - feat_mean) / feat_std

        feat_edges = feat_edges.detach()
        # pass through fusion net -> alphas per edge (P*P,1) -> reshape (P,P)
        alphas_flat = fusion_edge(feat_edges)   # (P*P,1)
        alphas = alphas_flat.view(P, P)         # (P,P)

        # compute fused GC tensor per-edge:
        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s  # (P,P)

        # --- use row-level aggregated alpha for prediction fusion training signal ---
        alpha_row = torch.mean(alphas, dim=1, keepdim=True)  # (P,1)
        # compute fused predictions using alpha_row per target (align reverse outputs)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]           # (T-1,)
            out_r = outs_rev[j]           # (T-1,) reverse-time
            out_r_aligned = torch.flip(out_r, dims=[0])  # align time
            a = alpha_row[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss = fused_predict_loss + loss_fn(fused_pred, target_seq[:, j])

        # consistency loss (encourage two branches to not disagree wildly)
        consistency_loss = torch.norm(GCs - GC2s, p=2)

        # alpha regularization (prevent collapse)

        alpha_reg = torch.mean(alphas * (1.0 - alphas))
        # alpha_reg = torch.mean((alphas - 0.5) ** 2)
        # alpha_reg = - torch.mean(alphas * torch.log(alphas + 1e-8) + (1 - alphas) * torch.log(1 - alphas + 1e-8))

        # total loss: keep original prediction + regularizer terms and add fused_predict_loss + regs
        # lambda_fused = 1.0
        # lambda_alpha_reg = 1e-3

        lambda_fused = 1.0
        lambda_alpha_reg = 5e-3

        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                0.01 * consistency_loss +
                lambda_fused * fused_predict_loss +
                lambda_alpha_reg * alpha_reg)

        # backward + step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        # 放到训练循环里、backward之后、评估之前
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, fused_eval.cpu().numpy(), False)

        # --- evaluate metrics for logging (convert to numpy) ---
        with torch.no_grad():
            fused_np = fused_GC_tensor.detach().cpu().numpy()
            GCs_np = GCs.detach().cpu().numpy()
            GC2s_np = GC2s.detach().cpu().numpy()

            score1 = compute_roc(GC, GCs_np, False)
            score2 = compute_roc(GC, GC2s_np, False)
            score_fusion = compute_roc(GC, fused_np, False)

            if best_score < score_fusion:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

        epoch_time = time.time() - start_time
        # print log (include alpha statistics)
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score, best_score1, best_score2
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
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    score = 0
    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    X, GC, length = fmri_read(simulation, subject)
    P = X.shape[1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)

        # combine loss
        loss = (predict_loss1 + predict_loss2  -
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
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
    plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)


    return best_score



import torch
import torch.nn.functional as F
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
def infer_Grangercausality_single116(simulation, subject, epoch, hidden_size, learning_rate, lambda_gc_sparse_base,lag_agg='mean',
                                  cutoff_ratio=0.2,tau=10.0):

    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    best_auroc = 0
    best_aupr = 0

    X, GC, length = fmri_read(simulation, subject)
    P = X.shape[1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    model = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    loss_fn = nn.MSELoss()
    params = list(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # --- 训练循环 ---
    smooth_flag = True  # 默认启用平滑
    best_score = -1e9
    best_auprc = -1e9
    # ------------------- 早停参数 -------------------
    patience = 10  # 容忍多少轮没有提升
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

        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model,
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
import numpy as np
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
# if __name__ == '__main__':
#     array = np.zeros((28, 50), dtype=float)
#     array1 = np.zeros((28, 50), dtype=float)
#     array2 = np.zeros((28, 50), dtype=float)
#     for i in range(1, 29):
#         for j in range(0, 50):
#             best_score, best_score1, best_score2 = infer_Grangercausalityv116(i, j, 128, 250, 0.01, 5, 0.001)
#             array[i - 1, j] = best_score
#             array1[i - 1, j] = best_score1
#             array2[i - 1, j] = best_score2
#             print(f'{best_score, best_score1, best_score2}')
#             print(f'node {i} done',f'node{j} done')
#
#     np.savetxt(f"../FMRI_fusion.txt", array, fmt='%.5f')
#     np.savetxt(f"../FMRI_origin.txt", array1, fmt='%.5f')
#     np.savetxt(f"../FMRI_reverse.txt", array2, fmt='%.5f')
def grid_search(param_grid):
    results_roc = []
    results_prc = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        avg_score=infer_Grangercausality_single116(3, 4, 600, hidden_size=params['hidden_size'],  lambda_gc_sparse_base=params['lambda_gc_sparse_base'],cutoff_ratio=params['cutoff_ratio'],
                                                           tau=params['tau'],learning_rate=params['learning_rate'],lag_agg=params['lag_agg']
                                           )
        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))
    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc
if __name__ == '__main__':

    # infer_Grangercausality(
    #     simulation=4,             # 输入维度
    #     subject=0,
    #     epoch=300,        # 训练轮数
    #     hidden_size=512,  # KAN隐藏层大小（未使用，但可以保留以扩展）
    #     learning_rate=0.0001,
    #     lam = 0.001,Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7989267676767677
    # Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.6748934687327441
    # )


    param_grid = {
        'hidden_size': [128,256],
        'lambda_gc_sparse_base': [0.002],
        'learning_rate': [0.001],
        'cutoff_ratio': [0.5,0.7,0.9],
        'tau': [0.1],
        'lag_agg': ['softmax']
    } ###  0.734,0.673,0.59,0.53,0.566

    best_params = grid_search(param_grid)
    # 3 0 Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.80697601010101
    # Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.6943861600334654

   # 3 1 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 256, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8767361111111112
    # Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7628287641882281

   # 3 2 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7989267676767677
# Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.6748934687327441

# 3 3 Best params: {'cutoff_ratio': 0.7, 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.0001, 'tau': 0.1} with avg score: 0.8303345959595959
# Best params: {'cutoff_ratio': 0.5, 'hidden_size': 256, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.6803642846471141

#3 4 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.0005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8880997474747475
#Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7454862960381881


# 1-23:new test result
# 0 Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.851010101010101
# Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.68177747407828

# 1 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.9122474747474748
# Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8305566271792997

# 2 Best params: {'cutoff_ratio': 0.9, 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8791035353535355
# Best params: {'cutoff_ratio': 0.9, 'hidden_size': 256, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7226432055610814

# 3 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8825757575757577
# Best params: {'cutoff_ratio': 0.9, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 1, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.746520994024479

# 4 Best params: {'cutoff_ratio': 0.5, 'hidden_size': 128, 'lag_agg': 'mean', 'lambda_gc_sparse_base': 0.0005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.8880997474747475
# #Best params: {'cutoff_ratio': 0.7, 'hidden_size': 128, 'lag_agg': 'softmax', 'lambda_gc_sparse_base': 0.0005, 'learning_rate': 0.001, 'tau': 0.1} with avg score: 0.7454862960381881
