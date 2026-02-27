import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
# from src.efficient_kan.kan01 import KAN
from tool import dream_read_label, dream_read_data
import time
import torch.nn.functional as F
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def off_diagonal(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr

def read_dream3(size, type):
    name_list = ["Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3"]
    GC = dream_read_label(
        r"/home/user/wcj/KANGCI-main/DREAM3 in silico challenge"
        r"/DREAM3 gold standards/DREAM3GoldStandard_InSilicoSize" + str(size) + "_" + name_list[type - 1] + ".txt",
        size)
    data = dream_read_data(
        r"/home/user/wcj/KANGCI-main/DREAM3 in silico challenge"
        r"/Size" + str(size) + "/DREAM3 data/InSilicoSize" + str(size) + "-" + name_list[
            type - 1] + "-trajectories.tsv")
    return GC, data


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)
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

    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda(1)
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda(1)

    X2 = X[::-1, :]  # reverse data
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]

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

        GCs = off_diagonal(GCs)
        GC2s = off_diagonal(GC2s)

        result = off_diagonal(result)

        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time

        if best_score < score_fusion:
            best_score = score_fusion
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}'
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}')
    print('Score:' + str(best_score))
    return score

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


# --- 修改后的 infer_Grangercausality 函数（替换原函数实现） ---
def infer_Grangercausalityv1(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate):
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)
    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:965 ,:]
    test_y = X[1:966, :]
    # use device variable defined at top of your file
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)
    models = nn.ModuleList(networks)

    # instantiate per-edge fusion net
    fusion_edge = FusionEdge(in_dim=5, hidden=64).to(device)

    # optimizer includes fusion params
    params = list(models.parameters()) + list(fusion_edge.parameters())
    optimizer = torch.optim.Adam([
    {'params': list(models.parameters()), 'lr': learning_rate},
    {'params': list(fusion_edge.parameters()), 'lr': learning_rate * 0.1}
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
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        # --- evaluate metrics for logging (convert to numpy) ---
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

        epoch_time = time.time() - start_time
        # print log (include alpha statistics)
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )
    print('Score:' + str(best_score))
    return score



def compute_grad_similarity(models_fwd, models_rev, losses_fwd, losses_rev, P):
    """
    输入:
        models_fwd, models_rev: P 个 forward/backward 模型
        losses_fwd, losses_rev: 每个模型对应的 MSE loss
        P: 节点数
    输出:
        grad_sim: (P*P,) 每条边的梯度方向相似度
    """
    sims = []
    for j, (m_f, m_r) in enumerate(zip(models_fwd, models_rev)):
        g1 = torch.autograd.grad(losses_fwd[j], m_f.layers[0].base_weight, retain_graph=True)[0].flatten()
        g2 = torch.autograd.grad(losses_rev[j], m_r.layers[0].base_weight, retain_graph=True)[0].flatten()
        sims.append(F.cosine_similarity(g1, g2, dim=0))
    sims = torch.stack(sims)  # (P,)
    return sims.unsqueeze(1).repeat(1, P).view(-1)  # (P*P,)
def compute_edge_features_v1(GCs, GC2s, outs_fwd, outs_rev,
                          target_seq, reversed_target_seq,
                          models, losses_fwd, losses_rev):
    P = GCs.shape[0]

    # --- 基础分数 ---
    g_fwd = GCs.view(-1)
    g_rev = GC2s.view(-1)
    absdiff = torch.abs(g_fwd - g_rev)

    # --- 残差动态特征 ---
    dyn_fwd = residual_dynamic_features(outs_fwd, target_seq)      # (P,3)
    dyn_rev = residual_dynamic_features(outs_rev, reversed_target_seq)  # (P,3)
    # 差值作为每条边的特征
    dyn_diff = (dyn_fwd - dyn_rev).unsqueeze(0).repeat(P, 1, 1)  # (P,P,3)
    dyn_diff = dyn_diff.view(-1, 3)  # (P*P,3)

    # --- 梯度方向相似度 (per-edge) ---
    grad_sim = compute_grad_similarity_per_edge(models[:P], models[P:], losses_fwd, losses_rev, P)  # (P*P,)

    # --- in/out degree ---
    in_degree = GCs.sum(0).view(-1)
    out_degree = GCs.sum(1).view(-1).repeat(P)
    in_degree_rep = in_degree.unsqueeze(0).repeat(P, 1).view(-1)

    # --- 拼接 ---
    feat_edges = torch.cat([
        g_fwd.detach().unsqueeze(1),
        g_rev.detach().unsqueeze(1),
        absdiff.detach().unsqueeze(1),
        grad_sim.detach().unsqueeze(1),
        in_degree_rep.detach().unsqueeze(1),
        out_degree.detach().unsqueeze(1),
        dyn_diff.detach()
    ], dim=1)

    # --- 标准化 ---
    feat_edges = torch.log1p(torch.abs(feat_edges))
    feat_mean = feat_edges.mean(dim=0, keepdim=True)
    feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
    feat_edges = (feat_edges - feat_mean) / feat_std

    return feat_edges
def residual_dynamic_features(outs, target_seq):
    """
    提取动态 residual 特征: 方差, lag-1 自相关, 偏度
    输入:
        outs: list of length P, 每个是 (T-1,) 预测
        target_seq: (T-1,P) 真值
    输出:
        feats: (P,3) 每个节点的 residual 动态特征
    """
    feats = []
    for j in range(len(outs)):
        res = (outs[j] - target_seq[:, j]).detach().cpu().numpy()  # (T-1,)
        var = np.var(res)
        if len(res) > 1:
            autocorr = np.corrcoef(res[:-1], res[1:])[0, 1]
        else:
            autocorr = 0.0
        skew = ((res - res.mean())**3).mean() / (res.std()**3 + 1e-8)
        feats.append([var, autocorr, skew])
    return torch.tensor(feats, dtype=torch.float32, device=target_seq.device)  # (P,3)
def compute_grad_similarity_per_edge(models_fwd, models_rev, losses_fwd, losses_rev, P):
    """
    计算每条边的梯度方向相似度 (P*P,)

    输入:
        models_fwd, models_rev: forward/backward 模型 (各 P 个)
        losses_fwd, losses_rev: 对应的 MSE loss
        P: 节点数
    输出:
        grad_sims: (P*P,) 每条边的梯度相似度
    """
    grad_sims = []
    for j, (m_f, m_r) in enumerate(zip(models_fwd, models_rev)):
        # 分别取前向/后向的梯度
        g1 = torch.autograd.grad(losses_fwd[j], m_f.layers[0].base_weight, retain_graph=True)[0]  # (P,)
        g2 = torch.autograd.grad(losses_rev[j], m_r.layers[0].base_weight, retain_graph=True)[0]  # (P,)

        # 对每个输入维度 i，算边 (i->j) 的梯度相似度
        sim_edges = []
        for i in range(P):
            # sim = F.cosine_similarity(g1[i].view(-1), g2[i].view(-1), dim=0)
            sim = F.cosine_similarity(g1[:, i].view(-1), g2[:, i].view(-1), dim=0)
            sim_edges.append(sim)
        grad_sims.append(torch.stack(sim_edges))  # (P,)

    grad_sims = torch.stack(grad_sims, dim=0)  # (P,P)
    return grad_sims.flatten()  # (P*P,)
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
def compute_edge_features_spare(GCs, GC2s, outs_fwd, outs_rev,
                          target_seq, reversed_target_seq,
                          models, losses_fwd, losses_rev):
    """
    输入:
        GCs, GC2s: (P,P) forward/backward 的 GC 矩阵
        outs_fwd, outs_rev: list of (T-1,) 或 (T-1,1) 预测
        target_seq, reversed_target_seq: (T-1,P) 序列
        models: ModuleList of 2P models
        losses_fwd, losses_rev: 每个模型的 loss
    输出:
        feat_edges: (P*P, feat_dim)
    """
    P = GCs.shape[0]

    # --- 基础分数 ---
    g_fwd = GCs.view(-1)   # (P*P,)
    g_rev = GC2s.view(-1)  # (P*P,)
    absdiff = torch.abs(g_fwd - g_rev)

    # --- 残差方差比 ---
    res_var_fwd = torch.stack([
        torch.var((outs_fwd[j].view(-1) - target_seq[:, j]).detach())
        for j in range(P)
    ])  # (P,)

    res_var_rev = torch.stack([
        torch.var((outs_rev[j].view(-1) - reversed_target_seq[:, j]).detach())
        for j in range(P)
    ])  # (P,)

    # 广播到 (P,P)
    res_ratio_matrix = (res_var_fwd[:, None] / (res_var_rev[None, :] + 1e-12))  # (P,P)
    res_ratio = res_ratio_matrix.reshape(-1)  # (P*P,)

    # --- in/out degree ---
    in_degree = GCs.sum(0).view(-1)              # (P,)
    out_degree = GCs.sum(1).view(-1).repeat(P)   # (P*P,)
    in_degree_rep = in_degree.unsqueeze(0).repeat(P, 1).view(-1)  # (P*P,)

    # print("g_fwd:", g_fwd.shape)
    # print("g_rev:", g_rev.shape)
    # print("absdiff:", absdiff.shape)
    # print("res_ratio:", res_ratio.shape)
    # print("in_degree_rep:", in_degree_rep.shape)
    # print("out_degree:", out_degree.shape)

    # --- 拼接 ---
    feat_edges = torch.stack([
        g_fwd.detach(),
        g_rev.detach(),
        absdiff.detach(),
        res_ratio.detach(),
        in_degree_rep.detach(),
        out_degree.detach()
    ], dim=1)  # (P*P, 6)

    # --- 标准化 ---
    feat_edges = torch.log1p(torch.abs(feat_edges))
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    test_x = X[:965,:]
    test_y = X[1:966,:]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
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

    for i in range(epoch):
        start_time = time.time()
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
        # GCs = torch.stack([models[k].GC(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)
        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)
        # learned alpha from fusion network (shape: (P*P,1) -> (P,P))
        alphas_flat = fusion_edge(feat_edges)  # (P*P, 1)
        alpha_learned = alphas_flat.view(-1).clamp(1e-6, 1 - 1e-6)  # (P*P,)
        alphas_learned_mat = alpha_learned.view(P, P)

        # ---- performance-derived confidences ----
        # use prediction losses to build confidences: cfwd_j = exp(-loss_fwd_j), crev_j = exp(-loss_rev_j)
        # (you may replace with 1/(var+eps) if prefer)
        eps = 1e-8
        cfwd = torch.stack([torch.exp(-l.detach()) for l in losses_fwd]).to(device)  # (P,)
        crev = torch.stack([torch.exp(-l.detach()) for l in losses_rev]).to(device)  # (P,)

        # broadcast to edges (matching in_degree_rep / view order used earlier)
        # for edge (i->j) we want target j confidence => repeat as rows P x P where each row contains cfwd (per column j)
        cfwd_rep = cfwd.unsqueeze(0).repeat(P, 1).reshape(-1)  # (P*P,)
        crev_rep = crev.unsqueeze(0).repeat(P, 1).reshape(-1)  # (P*P,)

        # performance-derived alpha per edge (probability): prefer forward if cfwd larger
        alpha_perf = (cfwd_rep / (cfwd_rep + crev_rep + eps)).clamp(1e-6, 1 - 1e-6)  # (P*P,)

        # ---- combine learned alpha and perf-alpha in logit domain ----
        def safe_logit(x):
            return torch.log(x) - torch.log(1.0 - x)

        gamma = 1.0  # hyperparameter: >0 favors performance signal more; tune (e.g., 0.5~2.0)
        logit_learned = safe_logit(alpha_learned)  # (P*P,)
        logit_perf = safe_logit(alpha_perf)  # (P*P,)
        logit_combined = logit_learned + gamma * logit_perf
        alpha_combined_flat = torch.sigmoid(logit_combined)  # (P*P,)
        alphas = alpha_combined_flat.view(P, P)  # (P,P)
        # alphas_flat = fusion_edge(feat_edges)
        # alphas = alphas_flat.view(P, P)

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

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score
def infer_Grangercausalityv4_edgeup(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,lambda_alpha_reg,lambda_consistency):
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    test_x = X[:965,:]
    test_y = X[1:966,:]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
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

    for i in range(epoch):
        start_time = time.time()
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
        # GCs = torch.stack([models[k].GC(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)
        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)
        # learned alpha from fusion network (shape: (P*P,1) -> (P,P))
        alphas_flat = fusion_edge(feat_edges)  # (P*P, 1)
        alpha_learned = alphas_flat.view(-1).clamp(1e-6, 1 - 1e-6)  # (P*P,)
        alphas_learned_mat = alpha_learned.view(P, P)

        # ---- performance-derived confidences ----
        # use prediction losses to build confidences: cfwd_j = exp(-loss_fwd_j), crev_j = exp(-loss_rev_j)
        # (you may replace with 1/(var+eps) if prefer)
        eps = 1e-8
        cfwd = torch.stack([torch.exp(-l.detach()) for l in losses_fwd]).to(device)  # (P,)
        crev = torch.stack([torch.exp(-l.detach()) for l in losses_rev]).to(device)  # (P,)

        # broadcast to edges (matching in_degree_rep / view order used earlier)
        # for edge (i->j) we want target j confidence => repeat as rows P x P where each row contains cfwd (per column j)
        cfwd_rep = cfwd.unsqueeze(0).repeat(P, 1).reshape(-1)  # (P*P,)
        crev_rep = crev.unsqueeze(0).repeat(P, 1).reshape(-1)  # (P*P,)

        # performance-derived alpha per edge (probability): prefer forward if cfwd larger
        alpha_perf = (cfwd_rep / (cfwd_rep + crev_rep + eps)).clamp(1e-6, 1 - 1e-6)  # (P*P,)

        # ---- combine learned alpha and perf-alpha in logit domain ----
        def safe_logit(x):
            return torch.log(x) - torch.log(1.0 - x)

        gamma = 1.0  # hyperparameter: >0 favors performance signal more; tune (e.g., 0.5~2.0)
        logit_learned = safe_logit(alpha_learned)  # (P*P,)
        logit_perf = safe_logit(alpha_perf)  # (P*P,)
        logit_combined = logit_learned + gamma * logit_perf
        alpha_combined_flat = torch.sigmoid(logit_combined)  # (P*P,)
        alphas = alpha_combined_flat.view(P, P)  # (P,P)
        # alphas_flat = fusion_edge(feat_edges)
        # alphas = alphas_flat.view(P, P)

        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        # alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        # fused_predict_loss = torch.tensor(0.0, device=device)
        # for j in range(P):
        #     out_f = outs_fwd[j]
        #     out_r = outs_rev[j]
        #     out_r_aligned = torch.flip(out_r, dims=[0])
        #     a = alpha_row[j].view(1)
        #     fused_pred = a * out_f + (1.0 - a) * out_r_aligned
        #     fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        # --- Method 1: weight by absolute forward-GC (per-target row weighting) ---
        eps = 1e-8
        # alphas: (P, P)  (row j -> alphas[j, :] are sources -> target j)
        # GCs:    (P, P)  (same layout assumed)

        # 1) 权重 = |GCs|，按每行（target）归一化到和为 1
        weights = torch.abs(GCs)  # (P, P)
        weights = weights / (weights.sum(dim=1, keepdim=True) + eps)  # normalize per row

        # 2) 用权重对每行的 alphas 做加权求和 -> per-target scalar
        alpha_learned_target = (alphas * weights).sum(dim=1)  # (P,)

        # 3) clip 保证数值稳定（可选）
        alpha_learned_target = alpha_learned_target.clamp(eps, 1.0 - eps)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]
            out_r = outs_rev[j]
            out_r_aligned = torch.flip(out_r, dims=[0])
            a = alpha_learned_target[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])
        # haixing  gen  yuanlunwenjiabuduo
        # eps = 1e-8
        # # 1) learned per-target alpha: 对每个 target j，取入边 alpha 的均值
        # alpha_learned_target = alphas.mean(dim=1).clamp(eps, 1.0 - eps)  # shape (P,)
        #
        # # 2) performance-derived per-target alpha: 用 cfwd, crev（已在你代码中计算过）
        # # cfwd, crev are shape (P,) where cfwd[j] = exp(-loss_fwd_j) etc.
        # alpha_perf_target = (cfwd / (cfwd + crev + eps)).clamp(eps, 1.0 - eps)  # shape (P,)
        #
        # # 3) combine them simply (weighted average). w controls how much to trust performance signal.
        # w = 0.5  # 推荐 0.3~0.7 范围；w=0.5 表示同等信任 learned 和 perf
        # alpha_target = (1.0 - w) * alpha_learned_target + w * alpha_perf_target  # (P,)
        #
        # # reshape to (P,1) to broadcast with time-series preds
        # alpha_target = alpha_target.view(P, 1)
        #
        # # 4) compute fused_predict_loss using per-target scalar (same as your original but with improved alpha)
        # fused_predict_loss = torch.tensor(0.0, device=device)
        # detach_preds = False  # 若内存吃紧，设 True 可节省大量显存（但 fused_predict_loss 不会反向影响 models）
        # for j in range(P):
        #     out_f = outs_fwd[j].view(-1)  # (T,)
        #     out_r = outs_rev[j].view(-1)
        #     out_r_aligned = torch.flip(out_r, dims=[0])
        #
        #     if detach_preds:
        #         out_f = out_f.detach()
        #         out_r_aligned = out_r_aligned.detach()
        #
        #     a = alpha_target[j].view(1)  # scalar tensor
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

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score
def infer_Grangercausalityv5(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,lambda_alpha_reg,lambda_consistency):
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    test_x = X[:965,:]
    test_y = X[1:966,:]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # build 2P models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1],
                  base_activation=nn.Identity,
                  use_sparse_proj=True,
                  proj_out_dim=P  # 或者 <P 做降维
                 ).to(device)
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
        start_time = time.time()
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
        proj_loss1 = sum([m.regularization_loss(lambda_proj=lam) for m in models[:P]])
        proj_loss2 = sum([m.regularization_loss(lambda_proj=lam) for m in models[P:2 * P]])

        GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)
        # GCs = torch.stack([models[k].GC(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)
        # --- 提取 edge 特征 ---
        feat_edges = compute_edge_features_spare(GCs, GC2s, outs_fwd, outs_rev,
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

        lambda_fused = 1.0
        # lambda_alpha_reg = 5.0
        # lambda_consistency = 0.05
        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                proj_loss1 + proj_loss2 +
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

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'proj_loss1: {proj_loss1.item():.4f}, proj_loss2: {proj_loss2.item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score
import numpy as np

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

def infer_Grangercausalityv5_lag(P, L, type, epoch, hidden_size, lam, lam_ridge,
                             learning_rate, lambda_alpha_reg, lambda_consistency):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # -------------------------
    # 构造 3 阶滞后数据
    # -------------------------
    lag = 3
    T = X.shape[0]

    # 输入: (N, lag*P), 输出: (N, P)
    input_list = []
    target_list = []
    for t in range(lag, T):
        input_list.append(X[t - lag:t, :].reshape(-1))  # 拼接 lag 个时刻
        target_list.append(X[t, :])  # 当前时刻

    input_array = np.stack(input_list)  # (N, lag*P)
    target_array = np.stack(target_list)  # (N, P)

    # 转成 torch 格式
    input_seq = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).to(device)  # (1, N, lag*P)
    target_seq = torch.tensor(target_array, dtype=torch.float32).to(device)  # (N, P)

    # -------------------------
    # 反向数据 (time reversal)
    # -------------------------
    X2 = np.ascontiguousarray(X[::-1, :])

    input_list_r = []
    target_list_r = []
    for t in range(lag, T):
        input_list_r.append(X2[t - lag:t, :].reshape(-1))
        target_list_r.append(X2[t, :])

    reversed_input_array = np.stack(input_list_r)
    reversed_target_array = np.stack(target_list_r)

    reversed_input_seq = torch.tensor(reversed_input_array, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_target_array, dtype=torch.float32).to(device)

    # -------------------------
    # build 2P models
    # -------------------------
    networks = []
    for _ in range(2 * P):
        network = KAN(num_series=P,  # 变量数
                      lag=lag,  # 滞后长度
                      hidden_size=hidden_size,
                      base_activation=nn.Identity,
                      use_projection=True  # ✅ 让 KAN 内部用 Projection
                      ).to(device)
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
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
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




def infer_Grangercausalityv6(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,lambda_alpha_reg,lambda_consistency):
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    test_x = X[:965,:]
    test_y = X[1:966,:]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
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

    for i in range(epoch):
        start_time = time.time()
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

        # GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        # GCs = torch.stack([models[k].GC(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
        #                    dim=0)
        GCs = torch.stack([models[k].pte_gc_nolag(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        GC2s = torch.stack([models[k].pte_gc_nolag(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],
                           dim=0)
        # --- 提取 edge 特征 ---
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

        epoch_time = time.time() - start_time
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score


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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
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
        # regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        # regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize_se(m, lam, "GL", learning_rate) for m in models[P:2 * P]])
        # GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)

        # GCs = torch.stack([models[k].GC_integrated_stable(input_seq) for k in range(P)], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated_stable(reversed_input_seq) for k in range(P, 2 * P)], dim=0)
        # --- 加上梯度正则化（Jacobian 稳定项） ---
        # jacobian_reg1 = sum([models[m].jacobian_regularization_loss(input_seq, lam=1e-3, mode='l1') for m in range(P)])
        # jacobian_reg2 = sum([models[m].jacobian_regularization_loss(reversed_input_seq, lam=1e-3, mode='l1') for m in range(P, 2 * P)])

        GCs = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P)], dim=0)
        GC2s = torch.stack([models[k].GC_weighted_by_sensitivity(input_seq) for k in range(P, 2 * P)], dim=0)
        # GCs = torch.stack([models[k].GC_integrated_v1(input_seq, target_idx=j) for j, k in enumerate(range(P))], dim=0)
        # GC2s = torch.stack([models[k].GC_integrated_v1(reversed_input_seq, target_idx=j) for j, k in enumerate(range(P, 2 * P))],dim=0)
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


        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
           
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}'
        )

    return best_score
def infer_Grangercausalityv4_gc_X_IG(P, type, epoch, hidden_size, lam, lam_ridge,
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # --- 构建 2P 个 KAN 模型 ---
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

    # ================================================================
    # =================== Training Loop ==============================
    # ================================================================
    for i in range(epoch):
        start_time = time.time()
        models.train()
        fusion_edge.train()
        optimizer.zero_grad()

        # === Forward predictions ===
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

        # === Regularizations ===
        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        # def get_effective_GC(model):
        #     if hasattr(model, 'GC_refined'):
        #         return model.update_gc_with_integrated.to(device)
        #     else:
        #         return model.GC().to(device)
        def get_effective_GC(model):
            # 始终返回当前模型的GC（已经可能被积分修正过）
            return model.GC().to(device)

        GCs = torch.stack([get_effective_GC(models[k]) for k in range(P)], dim=0)  # (P, P)
        GC2s = torch.stack([get_effective_GC(models[k]) for k in range(P, 2 * P)], dim=0)  # (P, P)

        # === Compute Edge features ===
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq,
                                           models, losses_fwd, losses_rev)

        alphas_flat = fusion_edge(feat_edges)
        alphas = alphas_flat.view(P, P)

        fused_GC_tensor = alphas * GCs + (1.0 - alphas) * GC2s

        # === Prediction fusion ===
        alpha_row = torch.mean(alphas, dim=1, keepdim=True)
        fused_predict_loss = torch.tensor(0.0, device=device)
        for j in range(P):
            out_f = outs_fwd[j]
            out_r = outs_rev[j]
            out_r_aligned = torch.flip(out_r, dims=[0])
            a = alpha_row[j].view(1)
            fused_pred = a * out_f + (1.0 - a) * out_r_aligned
            fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])

        # === Consistency and Entropy Regularization ===
        consistency_loss = torch.norm(GCs - GC2s, p=2)
        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)

        lambda_fused = 1.0
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

        # === EMA 更新 ===
        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, off_diagonal(fused_eval.cpu().numpy()), False)

        # === 性能评估 ===
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

        if (i + 1) % 10 == 0:
            print(f"\n[Epoch {i + 1}] Updating GC with Integrated Gradients ...")
            models_state = [m.training for m in models]
            for m in models:
                m.eval()

            for j in range(P):
                try:
                    refined = models[j].update_gc_with_integrated(input_seq, lambda_scale=0.1, steps=50)
                    print(f" Model {j} GC refined mean: {refined.mean():.6f}")
                except Exception as e:
                    print(f" ERROR updating model {j}: {e}")
                    continue

            for m, s in zip(models, models_state):
                if s:
                    m.train()
                else:
                    m.eval()
            print("GC correction completed.\n")

        # === 打印日志 ===
        epoch_time = time.time() - start_time
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
           
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score
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
def infer_Grangercausalityv4_inge_plus_try(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate,
                                  lambda_alpha_reg, lambda_consistency,
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

    # --- 数据读取 ---
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)
    #
    # global device
    # global_seed = 1
    # torch.manual_seed(global_seed)
    # torch.cuda.manual_seed_all(global_seed)
    # np.random.seed(global_seed)
    #
    # best_score = 0
    # best_score1 = 0
    # best_score2 = 0
    # GC, data = read_dream3(P, type)
    # GC = off_diagonal(GC)
    #
    # X = data.reshape(966, 100)
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    #
    # test_x = X[:965, :]
    # test_y = X[1:966, :]
    # input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    # target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)
    #
    # X2 = np.ascontiguousarray(X[::-1, :])
    # reversed_x = X2[:965, :]
    # reversed_y = X2[1:966, :]
    # # simulate and preprocess
    #
    #
    # reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    # reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

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

    fusion_edge = FusionEdge(in_dim=10, hidden=64).to(device)

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
    plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)

    return best_score
def infer_Grangercausalityv4_inge_plus_try_newfusion(P, type, epoch, hidden_size, learning_rate,
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
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]
    # simulate and preprocess


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
        if lambda_gc_sparse > 0.0 and ((i % gc_every) == 0):
            Lsparse1 = lambda_gc_sparse * torch.abs(GCs).sum()
            Lsparse2 = lambda_gc_sparse * torch.abs(GC2s).sum()
        else:
            Lsparse1 = torch.tensor(0.0, device=device)
            Lsparse2 = torch.tensor(0.0, device=device)

        # combine loss
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
    # plot_best_gc_heatmap(GC, best_GCs, best_GC2s, best_fused)
    plot_best_gc_heatmap_thre(GC, best_GCs, best_GC2s, best_fused, threshold=0.15)

    return best_score
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
    # --- 数据读取 ---
    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)

    # build two multi-output models (each outputs P targets for all time steps)
    # Architecture: KAN([P, hidden_size, P])  -> outputs (1, T, P)
    model_fwd = KAN([P, hidden_size, P], base_activation=nn.Identity).to(device)

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
    patience = 10  # 容忍多少轮没有提升
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
def grid_search(param_grid):
    # results = []
    results_roc = []
    results_prc = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        # avg_score = infer_Grangercausalityv4(100, 5, 300, hidden_size=params['hidden_size'], lam=params['lam'],
        #                                    lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate'], lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                      lambda_consistency=params['lambda_consistency']
        #                                    )
        # avg_score = infer_Grangercausalityv4_gc_X_IG(100, 2, 300, hidden_size=params['hidden_size'], lam=params['lam'],
        # avg_score = infer_Grangercausalityv4_inge(100, 1, 300, hidden_size=params['hidden_size'], lam=params['lam'],
        #                                      lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
        #                                      , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                      lambda_consistency=params['lambda_consistency'])
        # avg_score = infer_Grangercausalityv4_inge_plus_try(100, 1, 150, hidden_size=params['hidden_size'],
        #                                                    lam=params['lam'],
        #                                                    lam_ridge=params['lam_ridge'],
        #                                                    learning_rate=params['learning_rate']
        #                                                    , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                                    lambda_consistency=params['lambda_consistency'],
        #                                                    lambda_gc_sparse=params['lambda_gc_sparse'])
        # avg_score = infer_Grangercausalityv4_inge_plus_try_newfusion(100, 1, 150, hidden_size=params['hidden_size'],
        #                                                    learning_rate=params['learning_rate']
        #                                                    , lambda_alpha_reg=params['lambda_alpha_reg'],
        #                                                    lambda_gc_sparse=params['lambda_gc_sparse'])

        avg_score = infer_Grangercausalityv4_inge_plus_try_tosparse_1228(100, 1, 300, hidden_size=params['hidden_size'],
                                                                         learning_rate=params['learning_rate'],
                                                                         lambda_gc_sparse_base=params[
                                                                             'lambda_gc_sparse_base'],
                                                                         lag_agg=params['lag_agg'])
    #     results.append((params, avg_score))
    #
    # best_params = max(results, key=lambda x: x[1])
    # print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    # return best_params
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
   # v4 bushitidu
   #  Training with params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 3.0, 'lambda_alpha_reg': 3.0, 'lambda_consistency': 0.5, 'learning_rate': 0.001}  0.7493  1
    # Best params: {'hidden_size': 128, 'lam': 0.01, 'lam_ridge': 5.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.5, 'learning_rate': 0.001} with avg score: 0.7003210649355336
    # Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 5.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.5, 'learning_rate': 0.001} with avg score: 0.6344319129816987   3
    # Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 10.0, 'lambda_alpha_reg': 5.0, 'lambda_consistency': 0.5, 'learning_rate': 0.01} with avg score: 0.5897994177490061     4
    #  Best params: {'hidden_size': 128, 'lam': 0.01, 'lam_ridge': 5.0, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.05, 'learning_rate': 0.01} with avg score: 0.5925014253686303  5


    #111
    # param_grid = {
    #     'hidden_size': [256,128], ##128   orgin256
    #     'lam': [0.01,0.1,3],
    #     'lam_ridge': [0.5],# 5
    #     'learning_rate': [0.001]  # 0.005
    #     ,'lambda_alpha_reg':[10,5,3],
    #     'lambda_consistency':[0.05]  # 0.05
    # } ###  0.734,0.673,0.59,0.53,0.566
    # mingan   0.761613094629156  1   0.6823  2        3 buhao 0.61   4 Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 5, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.05, 'learning_rate': 0.001} with avg score: 0.5830204993325276     5  0.56
    # 1 Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 1, 'lambda_alpha_reg': 10.0, 'lambda_consistency': 0.05, 'learning_rate': 0.001} with avg score: 0.7449583631713556

    ## yakebi   infer_Grangercausalityv4_inge  1   Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 1.0, 'lambda_alpha_reg': 1.0, 'lambda_consistency': 0.1, 'learning_rate': 0.001} with avg score: 0.7664474680306905Best params: {'hidden_size': 256, 'lam': 0.01, 'lam_ridge': 1.0, 'lambda_alpha_reg': 1.0, 'lambda_consistency': 0.1, 'learning_rate': 0.001} with avg score: 0.7664474680306905
    # param_grid = {
    #     'hidden_size': [128],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.01]
    # }


    #1111  try  1205 feature 1205  5 0.533  4  0.542  3  0.5770  2 0.589
    # param_grid = {
    #     'hidden_size': [128,64],  ##128   orgin256
    #     'lam': [0.01],
    #     'lam_ridge': [5.0],
    #     'learning_rate': [0.01,0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.5,5],# 5
    #     'lambda_consistency': [0.05],
    #     'lambda_gc_sparse': [0.005,0.05] # 0.05
    # }

    #  try_fusion 2025 12 05
    # param_grid = {
    #     'hidden_size': [512],  ##128   orgin256
    #     'learning_rate': [0.001]  # 0.005
    #     , 'lambda_alpha_reg': [0.01],
    #     'lambda_consistency': [0.05],
    #     'lambda_gc_sparse': [0.005]
    # }
    # 2025.11.15 try  256 0.01   5   0.001   5   0.05   0.005   1： 0.78or0.76  2：0.76 3： 0.63  4： 0.67  5： 0.66




   # v1
    param_grid = {
        'hidden_size': [256,512],  ##128   orgin256
        'learning_rate': [0.001],  # 0.005 0.001
        'lambda_gc_sparse_base': [0.1,0.0008],  #
        'cutoff_ratio': [1],
        'lag_agg': ['softmax','mean'],
    }

    best_params = grid_search(param_grid)
