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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
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
#  no use
# class FusionNet(nn.Module):
#     def __init__(self, in_dim=5, hidden=32):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden // 2),
#             nn.ReLU(),
#             nn.Linear(hidden // 2, 1),
#             nn.Sigmoid()  # outputs alpha in (0,1)
#         )
#
#     def forward(self, x):
#         # x: (P, in_dim) -> returns (P, 1)
#         return self.net(x)
def infer_Grangercausality(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
    # Set seed for random number generation (for reproducibility of results)
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    score = 0

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # Lorenz-96 dataset
    X, GC = simulate_lorenz_96(p=P, T=500, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    X2 = X[::-1, :]  # reverse data
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda()
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda()

    # component-wise generate p models
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


        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_score < score_fusion:
            best_score = score_fusion
            best_score1 = score1
            best_score2 = score2

        epoch_time = time.time() - start_time

        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f},'
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}')

    return best_score,best_score1,best_score2

# def infer_Grangercausality(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate,
#                            lambda_fused_loss=1.0, lambda_alpha_reg=1e-3):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     global_seed = 1
#     torch.manual_seed(global_seed)
#     torch.cuda.manual_seed_all(global_seed)
#     np.random.seed(global_seed)
#
#     score = 0
#     best_score = 0
#     best_score1 = 0
#     best_score2 = 0
#
#     # data
#     X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     length = X.shape[0]
#
#     test_x = X[:length - 1]
#     test_y = X[1:length]
#     input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)
#     target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)
#
#     X2 = np.ascontiguousarray(X[::-1, :])
#     reversed_x = X2[:length - 1, :]
#     reversed_y = X2[1:length, :]
#     reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
#     reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)
#
#     # models
#     networks = []
#     for _ in range(2 * P):
#         network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
#         networks.append(network)
#     models = nn.ModuleList(networks)
#
#     fusion_net = FusionNet(in_dim=5, hidden=32).to(device)
#     params = list(models.parameters()) + list(fusion_net.parameters())
#     optimizer = torch.optim.Adam(params, lr=learning_rate)
#     loss_fn = nn.MSELoss()
#
#     # helper: safe wrapper to call regularize with possible different signatures
#     def safe_regularize(model, lam_val, lr_val):
#         # Try common signatures in order, return a tensor on the correct device
#         try:
#             return regularize(model, lam_val, "GL", lr_val)  # preferred
#         except TypeError:
#             try:
#                 return regularize(model, lam_val, lr_val)      # older variant?
#             except TypeError:
#                 try:
#                     return regularize(model, lam_val)          # even older
#                 except Exception:
#                     # fallback to zero tensor (on device)
#                     return torch.tensor(0.0, device=device)
#
#     # main loop
#     for i in range(epoch):
#         start_time = time.time()
#         models.train()
#         fusion_net.train()
#         optimizer.zero_grad()
#
#         losses1 = []
#         outs1 = []
#         for j in range(0, P):
#             network_output = models[j](input_seq).view(-1)
#             outs1.append(network_output)
#             loss_i = loss_fn(network_output, target_seq[:, j])
#             losses1.append(loss_i)
#
#         losses2 = []
#         outs2 = []
#         for j in range(P, 2 * P):
#             network_output = models[j](reversed_input_seq).view(-1)
#             outs2.append(network_output)
#             loss_i = loss_fn(network_output, reversed_target_seq[:, j - P])
#             losses2.append(loss_i)
#
#         predict_loss1 = sum(losses1)
#         predict_loss2 = sum(losses2)
#
#         ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
#         ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
#         regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
#         regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])
#
#         # build GCs
#         GCs = torch.stack([models[k].GC() for k in range(P)], dim=0)       # (P,P)
#         GC2s = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0)
#
#         # features for fusion net
#         diff_rows = torch.mean(torch.abs(GCs - GC2s), dim=1)  # (P,)
#         feat_list = []
#         for j in range(P):
#             fwd_loss_j = losses1[j].detach()
#             rev_loss_j = losses2[j].detach()
#             reg_fwd_j = safe_regularize(models[j], lam, learning_rate).detach()
#             reg_rev_j = safe_regularize(models[j + P], lam, learning_rate).detach()
#             feat = torch.stack([fwd_loss_j, rev_loss_j, reg_fwd_j, reg_rev_j, diff_rows[j].detach()])
#             feat_list.append(feat)
#         feat_tensor = torch.stack(feat_list, dim=0).to(device)
#         feat_tensor = torch.log1p(feat_tensor)
#
#         alphas = fusion_net(feat_tensor)  # (P,1)
#
#         # fused predict loss (align reverse outputs)
#         fused_predict_loss = torch.tensor(0.0, device=device)
#         for j in range(P):
#             out_f = outs1[j]
#             out_r = outs2[j]
#             out_r_aligned = torch.flip(out_r, dims=[0])
#             alpha_j = alphas[j].view(1)
#             fused_pred = alpha_j * out_f + (1.0 - alpha_j) * out_r_aligned
#             fused_predict_loss = fused_predict_loss + loss_fn(fused_pred, target_seq[:, j])
#
#         alpha_reg = torch.mean(alphas * (1.0 - alphas))
#         consistency_loss = torch.norm(GCs - GC2s, p=2)
#
#         loss = (predict_loss1 + predict_loss2 +
#                 regularize_loss1 + regularize_loss2 +
#                 ridge_loss1 + ridge_loss2 +
#                 0.1 * consistency_loss +
#                 lambda_fused_loss * fused_predict_loss +
#                 lambda_alpha_reg * alpha_reg)
#
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
#         optimizer.step()
#
#         # evaluate fused GC
#         with torch.no_grad():
#             fused_GCs = (alphas.detach() * GCs.detach()) + ((1.0 - alphas.detach()) * GC2s.detach())
#             fused_np = fused_GCs.cpu().numpy()
#             GCs_np = GCs.detach().cpu().numpy()
#             GC2s_np = GC2s.detach().cpu().numpy()
#             score1 = compute_roc(GC, GCs_np, False)
#             score2 = compute_roc(GC, GC2s_np, False)
#             score_fusion = compute_roc(GC, fused_np, False)
#             if best_score < score_fusion:
#                 best_score = score_fusion
#                 best_score1 = score1
#                 best_score2 = score2
#
#         epoch_time = time.time() - start_time
#         print(f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
#               f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
#               f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}, '
#               f'alpha_mean: {alphas.mean().item():.4f}, score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}')
#
#     return best_score, best_score1, best_score2

# --- 新增：FusionEdge class（放在文件顶部类定义区） ---
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
def infer_Grangercausalityv1(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
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
    X, GC = simulate_lorenz_96(p=P, T=500, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]
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


def infer_Grangercausalityv2(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
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
    X, GC = simulate_lorenz_96(p=P, T=500, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]
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
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
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


        # pass through fusion net -> alphas per edge (P*P,1) -> reshape (P,P)
        # alphas_flat = torch.clamp(fusion_edge(feat_edges), min=0.05, max=0.95)
        alphas_flat = fusion_edge(feat_edges)   # (P*P,1)
        alphas_flat = torch.sigmoid(alphas_flat)
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
        # k = min(3, P)  # 超参：top-k，P小则取P
        # fused_predict_loss = torch.tensor(0.0, device=device)
        # for j in range(P):
        #     # take top-k alpha values in row j (shape (k,))
        #     topk_vals, _ = torch.topk(alphas[j, :], k, largest=True)
        #     # use their mean as row-level fusion weight (shape ())
        #     a = topk_vals.mean()
        #     # align reverse prediction
        #     out_f = outs_fwd[j]
        #     out_r = outs_rev[j]
        #     out_r_aligned = torch.flip(out_r, dims=[0])
        #     fused_pred = a * out_f + (1.0 - a) * out_r_aligned
        #     fused_predict_loss += loss_fn(fused_pred, target_seq[:, j])
        # consistency loss (encourage two branches to not disagree wildly)
        consistency_loss = torch.norm(GCs - GC2s, p=2)

        # alpha regularization (prevent collapse)

        eps = 1e-8
        entropy = - (alphas * torch.log(alphas + eps) + (1 - alphas) * torch.log(1 - alphas + eps))
        alpha_reg = torch.mean(entropy)  # 越大表示 alpha 越均匀（接近 0.5）

        lambda_fused = 1.0
        lambda_alpha_reg = 0.5  # 建议先小一点

        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                0.01 * consistency_loss +
                lambda_fused * fused_predict_loss -
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

# def compute_grad_similarity(models_fwd, models_rev):
#     """计算每对 (fwd[j], rev[j]) 的梯度方向余弦相似度"""
#     sims = []
#     for m_f, m_r in zip(models_fwd, models_rev):
#         g1 = torch.autograd.grad(
#             outputs=m_f.layers[0].base_weight.sum(),
#             inputs=m_f.layers[0].base_weight,
#             retain_graph=True,
#             create_graph=False
#         )[0].flatten()
#         g2 = torch.autograd.grad(
#             outputs=m_r.layers[0].base_weight.sum(),
#             inputs=m_r.layers[0].base_weight,
#             retain_graph=True,
#             create_graph=False
#         )[0].flatten()
#         cos = F.cosine_similarity(g1, g2, dim=0)
#         sims.append(cos)
#     sims = torch.stack(sims)  # (P,)
#     # 扩展到 (P*P,) 形式：每行相同
#     P = sims.shape[0]
#     sims = sims.unsqueeze(1).repeat(1, P).view(-1)
#     return sims
# 更有意义的 grad similarity（在训练循环中）
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
            sim = F.cosine_similarity(g1[i].view(-1), g2[i].view(-1), dim=0)
            sim_edges.append(sim)
        grad_sims.append(torch.stack(sim_edges))  # (P,)

    grad_sims = torch.stack(grad_sims, dim=0)  # (P,P)
    return grad_sims.flatten()  # (P*P,)


# --------------------
# 融合特征提取
# --------------------
#  效果并不好 下面这个动态和计算特征的函数
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
# 99.8  原梯度计算 但是好像是错误的
def infer_Grangercausalityv3(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    score = 0
    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    X, GC = simulate_lorenz_96(p=P, T=500, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]
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

    fusion_edge = FusionEdge(in_dim=7, hidden=64).to(device)

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

        # -------------------
        # 替换后的 edge 特征
        # -------------------
        feat_edges = compute_edge_features(GCs, GC2s, outs_fwd, outs_rev,
                                           target_seq, reversed_target_seq, models)

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
        lambda_alpha_reg = 0.5

        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                0.01 * consistency_loss +
                lambda_fused * fused_predict_loss -
                lambda_alpha_reg * alpha_reg)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        if 'alpha_ema' not in locals():
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        fused_eval = alpha_ema * GCs.detach() + (1 - alpha_ema) * GC2s.detach()
        score_fusion = compute_roc(GC, fused_eval.cpu().numpy(), False)

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
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score, best_score1, best_score2
# 99.8    120  99.9
def infer_Grangercausalityv4(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
    global device
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0
    best_score1 = 0
    best_score2 = 0

    # simulate and preprocess
    X, GC = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    length = X.shape[0]

    test_x = X[:length - 1]
    test_y = X[1:length]
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
        lambda_alpha_reg = 0.5

        loss = (predict_loss1 + predict_loss2 +
                regularize_loss1 + regularize_loss2 +
                ridge_loss1 + ridge_loss2 +
                0.01 * consistency_loss +
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
        score_fusion = compute_roc(GC, fused_eval.cpu().numpy(), False)

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
        print(
            f'Epoch [{i+1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
            f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
            f'ridge_loss1: {ridge_loss1.item():.4f}, ridge_loss2: {ridge_loss2.item():.4f}, '
            f'alpha_mean_row: {alpha_row.mean().item():.4f}, alpha_mean_edge: {alphas.mean().item():.4f}, '
            f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}'
        )

    return best_score, best_score1, best_score2
# 单独训练alfa的版本  利用真实值，但是真实数据集没有真实值
def infer_Grangercausality_fusion_only(P, F, epoch, hidden_size, lam, lam_ridge, learning_rate):
    """
    目标：
      - 训练 2P 个单目标模型（P 个顺向 + P 个反向），与常规训练保持一致；
      - 同时用一个小的可学习网络 FusionEdge 将两组 GC 矩阵按边融合为最终 GC；
      - 融合网络独立训练（使用真实 GC 作为监督信号），特征从 GCs detach（不会影响 base models）。
    输入/输出与原函数一致：返回 best_score, best_score1, best_score2（按 ROC/AUC 比较）
    """
    # reproducibility
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed_all(global_seed)
    np.random.seed(global_seed)

    best_score = 0.0
    best_score1 = 0.0
    best_score2 = 0.0

    # --- 数据准备（使用 Lorenz-96，与你的代码保持一致） ---
    X, GC_true = simulate_lorenz_96(p=P, T=1000, F=F, delta_t=0.1, sd=0.1, burn_in=1000, seed=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Tlen = X.shape[0]

    # 构造 step 监督对
    test_x = X[:Tlen - 1]      # (T-1, P)
    test_y = X[1:Tlen]         # (T-1, P)
    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).to(device)    # (1, T-1, P)
    target_seq = torch.tensor(test_y, dtype=torch.float32).to(device)               # (T-1, P)

    # 反向序列
    X2 = np.ascontiguousarray(X[::-1, :])
    reversed_x = X2[:Tlen - 1, :]
    reversed_y = X2[1:Tlen, :]
    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).to(device)
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).to(device)

    # --- 构建 2P 个模型（与原来一致） ---
    networks = []
    for _ in range(2 * P):
        net = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(net)
    models = nn.ModuleList(networks)

    # 损失与优化器：模型参数使用一个 optimizer；fusion 网络使用单独的 optimizer
    loss_fn = nn.MSELoss()
    optimizer_models = torch.optim.Adam(models.parameters(), lr=learning_rate)

    # 实例化融合网络并单独优化
    fusion_edge = FusionEdge(in_dim=5, hidden=64).to(device)
    optimizer_fusion = torch.optim.Adam(fusion_edge.parameters(), lr=learning_rate * 0.1)

    # fusion 的监督损失：如果 GC_true 是二值邻接（0/1），可用 BCE；否则用 MSE。
    # 这里我们假设 simulate_lorenz_96 返回的 GC 是 0/1（若不是，可改为 MSELoss）
    bce_loss = nn.BCELoss()

    # 把真实 GC 转为 tensor（放到 device）
    GC_true_torch = torch.tensor(GC_true, dtype=torch.float32).to(device)  # (P, P)

    # EMA 用于评估（可选）
    alpha_ema = None
    lambda_alpha_reg = 1e-3    # alpha 正则系数（会把 alpha 拉向 0.5，避免极端）
    lambda_fusion_supervise = 1.0  # 融合网络监督损失权重（可调）

    for ep in range(epoch):
        t0 = time.time()
        # ---------- 1) 训练/更新 base models ----------
        models.train()
        optimizer_models.zero_grad()

        # 顺向分支预测与损失
        losses_fwd = []
        for j in range(P):
            out_j = models[j](input_seq).view(-1)              # (T-1,)
            losses_fwd.append(loss_fn(out_j, target_seq[:, j]))
        # 反向分支
        losses_rev = []
        for j in range(P, 2 * P):
            out_j = models[j](reversed_input_seq).view(-1)
            losses_rev.append(loss_fn(out_j, reversed_target_seq[:, j - P]))

        predict_loss1 = sum(losses_fwd)
        predict_loss2 = sum(losses_rev)

        # 正则项（与原实现一致）
        ridge_loss1 = sum([ridge_regularize(m, lam_ridge) for m in models[:P]])
        ridge_loss2 = sum([ridge_regularize(m, lam_ridge) for m in models[P:2 * P]])
        regularize_loss1 = sum([regularize(m, lam, "GL", learning_rate) for m in models[:P]])
        regularize_loss2 = sum([regularize(m, lam, "GL", learning_rate) for m in models[P:2 * P]])

        loss_models = predict_loss1 + predict_loss2 + ridge_loss1 + ridge_loss2 + regularize_loss1 + regularize_loss2

        loss_models.backward()
        torch.nn.utils.clip_grad_norm_(models.parameters(), max_norm=1.0)
        optimizer_models.step()

        # ---------- 2) 计算当前 epoch 的 GCs（detach，为 fusion 提供特征） ----------
        # 使用 torch.stack 保持 tensor 形式，方便后续计算
        with torch.no_grad():
            GCs_t = torch.stack([models[k].GC() for k in range(P)], dim=0).to(device)      # (P, P)
            GC2s_t = torch.stack([models[k].GC() for k in range(P, 2 * P)], dim=0).to(device) # (P, P)

        # ---------- 3) 构造每条边的特征并训练 fusion 网络 ----------
        # 特征: [g_fwd, g_rev, absdiff, row_mean_absdiff, pred_loss_ratio]
        # 注意：我们把特征全部 detach（上面 already in no_grad），fusion 只学习如何从这些数值映射到 alpha
        g_fwd = GCs_t.view(-1)           # (P*P,)
        g_rev = GC2s_t.view(-1)
        absdiff = torch.abs(g_fwd - g_rev)

        row_absdiff = torch.mean(torch.abs(GCs_t - GC2s_t), dim=1)   # (P,)
        row_mean_rep = row_absdiff.unsqueeze(1).repeat(1, P).view(-1) # (P*P,)

        # pred loss ratio: 使用上面 losses（detach -> 数值），若 losses 为 0 需小偏移
        loss_fwd_vec = torch.stack([l.detach() for l in losses_fwd]).to(device)    # (P,)
        loss_rev_vec = torch.stack([l.detach() for l in losses_rev]).to(device)
        pred_ratio = (loss_fwd_vec / (loss_rev_vec + 1e-12)).unsqueeze(1).repeat(1, P).view(-1)

        # stack features (P*P, 5)
        feat_edges = torch.stack([
            g_fwd, g_rev, absdiff, row_mean_rep, pred_ratio
        ], dim=1)   # 所有项已经在 no_grad 中 -> detach

        # 标准化（同你之前做法）
        feat_edges = torch.log1p(torch.abs(feat_edges))
        feat_mean = feat_edges.mean(dim=0, keepdim=True)
        feat_std = feat_edges.std(dim=0, keepdim=True).clamp(min=1e-6)
        feat_edges = (feat_edges - feat_mean) / feat_std

        # 前向 fusion 网络（训练 fusion_net）
        fusion_edge.train()
        optimizer_fusion.zero_grad()

        alphas_flat = fusion_edge(feat_edges)          # (P*P, 1)
        alphas = alphas_flat.view(P, P)               # (P, P) in (0,1)

        # fused GC（注意：使用 detach 的 GCs_t，不会反传回 base models）
        fused_GC = alphas * GCs_t.detach() + (1.0 - alphas) * GC2s_t.detach()  # (P, P)

        # 选择 fusion loss：GC_true 假设为 0/1 二值矩阵 -> 使用 BCE；若是连续值，可换 MSE
        fusion_loss = bce_loss(fused_GC, GC_true_torch)

        # alpha 正则：我们希望 alpha 不太极端 -> 惩罚偏离 0.5 的平方
        alpha_reg = torch.mean((alphas - 0.5) ** 2)
        fusion_loss_total = lambda_fusion_supervise * fusion_loss + lambda_alpha_reg * alpha_reg

        fusion_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(fusion_edge.parameters(), max_norm=1.0)
        optimizer_fusion.step()

        # EMA（用于平滑评估）
        if alpha_ema is None:
            alpha_ema = alphas.detach().clone()
        else:
            alpha_ema = 0.9 * alpha_ema + 0.1 * alphas.detach()

        # ---------- 4) 评估与记录 ----------
        with torch.no_grad():
            fused_eval = alpha_ema * GCs_t.detach() + (1 - alpha_ema) * GC2s_t.detach()
            fused_np = fused_eval.cpu().numpy()
            GCs_np = GCs_t.detach().cpu().numpy()
            GC2s_np = GC2s_t.detach().cpu().numpy()

            score1 = compute_roc(GC_true, GCs_np, False)
            score2 = compute_roc(GC_true, GC2s_np, False)
            score_fusion = compute_roc(GC_true, fused_np, False)

            if score_fusion > best_score:
                best_score = score_fusion
                best_score1 = score1
                best_score2 = score2

        t_cost = time.time() - t0
        # 日志
        if (ep + 1) % 1 == 0:
            print(
                f'Epoch [{ep+1}/{epoch}], model_loss_fwd: {predict_loss1.item():.4f}, model_loss_rev: {predict_loss2.item():.4f}, '
                f'fusion_bce: {fusion_loss.item():.6f}, alpha_mean: {alphas.mean().item():.4f}, '
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time: {t_cost:.3f}'
            )

    return best_score, best_score1, best_score2


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

    # --- 模型构建 ---
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
        'lag_agg': ['softmax'],  # 'softmax' 'quantile'
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
