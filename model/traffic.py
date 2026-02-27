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

def infer_Grangercausalityv4_inge_plus_try_tosparse_1211_single(P, type, epoch, hidden_size, learning_rate,
                                  lambda_gc_sparse_base,
                                  gc_create_graph=True, gc_every=1,
                                  lag_agg='mean',
                                  cutoff_ratio=0.2,
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
    device = device_local 
    try:
        input_seq, target_seq, reversed_input_seq, reversed_target_seq, GC, X_raw = \
            load_causaltime_seq(data_root, device_local, scaler_type='minmax', n_max=P)
        # print('1111')
        if isinstance(GC, np.ndarray):
            GC = GC.astype(float)
    except NameError:
        pass
    P_effective = input_seq.shape[2]
    model_fwd = KAN([P_effective, hidden_size, P_effective], base_activation=nn.Identity).to(device)
    params = list(model_fwd.parameters())
    optimizer = torch.optim.Adam([
        {'params': list(model_fwd.parameters()), 'lr': learning_rate}
    ])
    loss_fn = nn.MSELoss()
    smooth_flag = True  
    best_score = -1e9
    best_auprc = -1e9
    patience = 20  
    min_delta = 1e-5  
    counter = 0  
    previous_auroc = -1e9  
    for i in range(epoch):
        model_fwd.train()
        optimizer.zero_grad()
        inp_fwd = input_seq
        outs_fwd_all = model_fwd(inp_fwd).squeeze(0)  # (T, P)
        losses_fwd = [loss_fn(outs_fwd_all[:, j], target_seq[:, j]) for j in range(P)]
        predict_loss1 = sum(losses_fwd)  # L_fwd
        GCs_raw, L_gc = compute_gradient_gc_smooth_universal_v3_1220_103v1(
            model_fwd,
            input_seq,
            create_graph=True,
            lag_agg_local=lag_agg,
            freq_denoise=True,  
            cutoff_ratio=cutoff_ratio,
            lambda_gc=lambda_gc_sparse_base,
            tau=tau,
            tau_prior=tau_prior,
            beta_prior=beta_prior
        )
        loss = predict_loss1 + L_gc
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
            counter += 1  
        else:
            counter = 0 
        previous_auroc = score1

        print(f"Epoch [{i + 1}/{epoch}] loss: {loss.item():.6f} | predict_loss1: {predict_loss1.item():.6f} | "
              f"Lsparse_fwd: {L_gc.item():.6f} | "
              f"score1: {score1:.4f}  | "
              f"AUPRC_fwd: {auprc1:.4f} | ")
        if counter > 0:
            print(f"  --- No improvement counter: {counter} / {patience} ---")

        if counter >= patience:
            print(
                f" Early stopping triggered after {i + 1} epochs! AUPRC did not improve for {patience} rounds. ")
            break 
    # End training
    # return best_score, best_auprc
    return best_score, best_auprc
def grid_search(param_grid):
    results_roc = []
    results_prc = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")
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
