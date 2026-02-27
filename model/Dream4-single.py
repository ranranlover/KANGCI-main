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







def infer_Grangercausalityv4_inge_plus_try_tosparse(P, type, epoch, hidden_size, learning_rate,
                                                    lambda_alpha_reg,
                                                    lambda_gc_sparse_base, lambda_gc_sparse_fusion,
                                                    gc_create_graph=True, gc_every=1,
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
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)

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
            out_j = outs[:, j]  # (T,)
            s_j = out_j.sum()  # scalar
            grads = torch.autograd.grad(s_j, inp, create_graph=create_graph, retain_graph=True)[0]  # (1, T, P)
            # aggregate across time dim: mean or max
            if lag_agg_local == 'mean':
                gc_row = grads.abs().mean(dim=1).squeeze(0)  # (P,)
            elif lag_agg_local == 'max':
                gc_row = grads.abs().max(dim=1)[0].squeeze(0)  # (P,)
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
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        outs_rev_all = model_rev(inp_rev).squeeze(0)  # (T, P)

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
            GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=gc_create_graph,
                                                lag_agg_local=lag_agg)
            GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=gc_create_graph,
                                                 lag_agg_local=lag_agg)
        else:
            # monitoring only (no graph)
            with torch.no_grad():
                GCs = compute_gradient_gc_for_model(model_fwd, input_seq, create_graph=False, lag_agg_local=lag_agg)
                GC2s = compute_gradient_gc_for_model(model_rev, reversed_input_seq, create_graph=False,
                                                     lag_agg_local=lag_agg)

        # --- compute fused alphas and fused GC tensor as before ---
        # Note: you used outs_fwd, outs_rev lists previously; adapt to new shape
        # We provide outs_fwd_list and outs_rev_list (list of per-target tensors) to existing compute_edge_features
        outs_fwd = [outs_fwd_all[:, j] for j in range(P)]
        outs_rev = [outs_rev_all[:, j] for j in range(P)]

        feat_edges = compute_edge_features_1205(GCs, GC2s, outs_fwd, outs_rev,
                                                target_seq, reversed_target_seq,
                                                None, losses_fwd,
                                                losses_rev)  # models->None as not needed, adapt compute_edge_features if required
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
        loss = (predict_loss1 + predict_loss2 -
                lambda_alpha_reg * alpha_reg +
                Lsparse1 + Lsparse2 + Lsparse_fused)

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
            f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'Lsparse1: {Lsparse1.item():.4f}, Lsparse2: {Lsparse2.item():.4f},Lsparse_fused: {Lsparse_fused.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f},'
            f'AUPRC_fwd: {auprc1:.4f}, AUPRC_rev: {auprc2:.4f}, AUPRC_fusion: {auprc_fusion:.4f}'
        )

    # plot best GC heatmap (unchanged)
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    # plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)

    return best_score, best_auprc
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
                                                         , cutoff_ratio=0.2):
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
            GCs_raw = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq, cutoff_ratio=cutoff_ratio,
                                                              create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s_raw = compute_gradient_gc_smooth_universal_v2(model_rev, reversed_input_seq, cutoff_ratio=cutoff_ratio,
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
            score1, auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            score2, auprc2 = compute_roc_1219(GC, off_diagonal(GC2s_np), False)
            score_fusion, auprc_fusion = compute_roc_1219(GC, off_diagonal(fused_np), False)

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
            GCs_raw = compute_gradient_gc_smooth_universal_v3(model_fwd, input_seq, cutoff_ratio=cutoff_ratio,
                                                              create_graph=gc_create_graph, lag_agg_local=lag_agg)
            GC2s_raw = compute_gradient_gc_smooth_universal_v3(model_rev, reversed_input_seq, cutoff_ratio=cutoff_ratio,
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
            score1, auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            score2, auprc2 = compute_roc_1219(GC, off_diagonal(GC2s_np), False)
            score_fusion, auprc_fusion = compute_roc_1219(GC, off_diagonal(fused_np), False)

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


def infer_Grangercausalityv4_inge_plus_try_tosparse_1208_single(P, type, epoch, hidden_size, learning_rate,
                                                         lambda_gc_sparse_base,
                                                         gc_create_graph=True, gc_every=1,
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


    best_auprc = 0.0

    # simulate and preprocess
    GC, X = read_dream4(P, type)
    GC = off_diagonal(GC)

    length = X.shape[0]

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)  # (T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)
    params = list(model_fwd.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate}
    ])
    loss_fn = nn.MSELoss()

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
        optimizer.zero_grad()
        inp_fwd = input_seq
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        losses_fwd = []
        loss_t_fwd = (outs_fwd_all - target_seq) ** 2
        for j in range(P):
            losses_fwd.append(loss_fn(outs_fwd_all[:, j], target_seq[:, j]))
        predict_loss1 = sum(losses_fwd)
        GCs = compute_gradient_gc_smooth_universal_v2(model_fwd, input_seq, create_graph=gc_create_graph,
                                         lag_agg_local=lag_agg)
        Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()
        # combine loss
        loss = (predict_loss1 +Lsparse1 )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            GCs_np = GCs.detach().cpu().numpy()
            score1,auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)
            if best_auprc < auprc1:
                best_auprc = auprc1
            if best_score < score1:
                best_score = score1
        print(
            f'Epoch [{i + 1}/{epoch}] | predict_loss1: {predict_loss1.item():.4f} | '
            f'Lsparse1: {Lsparse1.item():.4f} | '
            f'score1: {score1:.4f} | '
            f'AUPRC_fwd: {auprc1:.4f} '
        )
    return best_score, best_auprc
    # return best_auprc
def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_1220_single(P, type, epoch, hidden_size, learning_rate,
                                                              lambda_gc_sparse_base,
                                                              gc_create_graph=True, gc_every=1,
                                                              lag_agg='mean',cutoff_ratio=0.2,
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


    # 模型构建 (KAN, FusionEdge)
    model_fwd = KAN_1219([P, hidden_size, P], base_activation=nn.Identity).to(device)

    params = list(model_fwd.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate},
    ])

    loss_fn = nn.MSELoss()
    # ---------------- 训练循环 ----------------

    for i in range(epoch):
        model_fwd.train()
        optimizer.zero_grad()
        inp_fwd = input_seq
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        # 预测损失
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)  # L_fwd

        GCs_raw = compute_gradient_gc_smooth_universal_v3(model_fwd, input_seq, cutoff_ratio=cutoff_ratio,
                                                          create_graph=gc_create_graph, lag_agg_local=lag_agg)
        with torch.no_grad():
            combined_gc_values = torch.cat([GCs_raw.view(-1)])
            # 忽略 0 值，防止稀疏矩阵对分位数的干扰
            nonzero_combined = combined_gc_values[combined_gc_values > 1e-6]
            if nonzero_combined.numel() > 0:
                clip_threshold = torch.quantile(nonzero_combined, grad_clip_quantile)
            else:
                clip_threshold = 1.0  # 默认值
        # 应用裁剪
        GCs = GCs_raw.clamp(max=clip_threshold)
        # 3. 融合和正则化
        outs_fwd_list = [outs_fwd_all[:, j] for j in range(P)]

        Lsparse1 = lambda_gc_sparse_base * torch.abs(GCs).sum()

        # 4. 组合总损失
        loss = (predict_loss1 + Lsparse1)

        # 5. 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            GCs_np = GCs.detach().cpu().numpy()

            score1, auprc1 = compute_roc_1219(GC, off_diagonal(GCs_np), False)

            if best_auprc < auprc1:
                best_auprc = auprc1
            if best_score < score1:
                best_score = score1
        print(
            f'Epoch [{i + 1}/{epoch}] | L_pred1: {predict_loss1.item():.4f} | '
            f'ROC_fwd: {score1:.4f} | '
            f'AUPRC_fwd: {auprc1:.4f}'
        )

    return best_score, best_auprc
def grid_search(param_grid):
    results_roc = []
    results_prc = []
    param_list = list(ParameterGrid(param_grid))
    # for i in [1,2,3,4,5]:
    for params in param_list:
        print(f"Training with params: {params}")



        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1211_1220_single(100, 2, 300, hidden_size=params['hidden_size'],
                                                                         learning_rate=params['learning_rate'],
                                                                         lambda_gc_sparse_base=params[
                                                                             'lambda_gc_sparse_base'],
                                                                         )


        results_roc.append((params, avg_score[0]))
        results_prc.append((params, avg_score[1]))

    best_params_roc = max(results_roc, key=lambda x: x[1])
    best_params_prc = max(results_prc, key=lambda x: x[1])
    print(f"Best params: {best_params_roc[0]} with avg score: {best_params_roc[1]}")
    print(f"Best params: {best_params_prc[0]} with avg score: {best_params_prc[1]}")
    return best_params_roc


if __name__ == '__main__':

    param_grid = {
        'hidden_size': [512],  ##128   orgin256
        'learning_rate': [0.001],  # 0.005
        'lambda_gc_sparse_base': [0.001]
    }

    best_params = grid_search(param_grid)

