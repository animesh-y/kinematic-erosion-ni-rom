import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

import os
import json
import logging
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Architectures
# ==========================================
class SemiCircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return self.conv(x)

class MultiChannelCNN(nn.Module):
    def __init__(self, in_channels=23, latent_dim=517):
        super().__init__()
        act = nn.GELU
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            SemiCircularConv2d(in_channels, 32, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(32, 64, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(64, 128, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(128, 256, 3, 1), act(), nn.MaxPool2d(2, 2)
        )
        self.flatten_size = 256 * 32 * 16
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2), act(),
            nn.ConvTranspose2d(128, 64, 2, 2), act(),
            nn.ConvTranspose2d(64, 32, 2, 2), act(),
            nn.ConvTranspose2d(32, in_channels, 2, 2)
        )

    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.fc_encode(h)
        h_dec = self.fc_decode(z).view(z.size(0), 256, 32, 16)
        return self.decoder(h_dec)

class BlockMultiChannelCNN(nn.Module):
    def __init__(self, in_channels, latent_dim=16):
        super().__init__()
        act = nn.GELU
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            SemiCircularConv2d(in_channels, 32, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(32, 64, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(64, 128, 3, 1), act(), nn.MaxPool2d(2, 2),
            SemiCircularConv2d(128, 256, 3, 1), act(), nn.MaxPool2d(2, 2)
        )
        self.flatten_size = 256 * 32 * 16
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2), act(),
            nn.ConvTranspose2d(128, 64, 2, 2), act(),
            nn.ConvTranspose2d(64, 32, 2, 2), act(),
            nn.ConvTranspose2d(32, in_channels, 2, 2)
        )
    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        z = self.fc_encode(h)
        h_dec = self.fc_decode(z).view(z.size(0), 256, 32, 16)
        return self.decoder(h_dec)



class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_models, num_features=6):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_models]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=num_features, batch_shape=torch.Size([num_models])),
            batch_shape=torch.Size([num_models])
        )
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def apply_power_law_transform(X, power=0.5):
    return np.sign(X) * (np.abs(X) ** power)

def inverse_power_law_transform(Y, power=0.5):
    return np.sign(Y) * (np.abs(Y) ** (1.0 / power))

def train_batch_gpr(train_x, train_y, num_features=4, lr=0.1, training_iter=100):
    num_models = train_y.shape[0]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([num_models]))
    model = BatchGPModel(train_x, train_y, likelihood, num_models, num_features=num_features)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(training_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y).sum()
        loss.backward()
        optimizer.step()
    return model, likelihood

def get_grid(grid_s=582, grid_theta=221):
    s_1d = np.linspace(0.0, 2.0, grid_s)
    theta_1d = np.linspace(-np.pi, np.pi, grid_theta)
    s_grid, theta_grid = np.meshgrid(s_1d, theta_1d, indexing='ij')
    return s_grid, theta_grid, s_1d, theta_1d

def predict_batch_gpr(model, likelihood, test_x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval(); likelihood.eval()
    num_models = model.mean_module.batch_shape[0]
    if not isinstance(test_x, torch.Tensor):
        test_x = torch.from_numpy(test_x).float()
    test_x_batch = test_x.unsqueeze(0).repeat(num_models, 1, 1).to(device)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_mean = likelihood(model(test_x_batch)).mean
    return pred_mean.cpu().numpy()

def compute_pod_snapshot_method(Z_train_centered, Z_test_centered, k):
    K_gram = Z_train_centered @ Z_train_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(K_gram)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    k_actual = min(k, int(np.sum(eigenvalues[idx] > 1e-12)))
    inv_sqrt_val = 1.0 / np.sqrt(eigenvalues[idx][:k_actual])
    Vt_k = (eigenvectors[:, :k_actual].T * inv_sqrt_val[:, None]) @ Z_train_centered
    pod_coeffs_train = Z_train_centered @ Vt_k.T
    pod_coeffs_test = Z_test_centered @ Vt_k.T
    Z_test_pred_matrix = pod_coeffs_test @ Vt_k
    return Z_test_pred_matrix, Vt_k, pod_coeffs_train, pod_coeffs_test

def compute_hosvd_snapshot(Z_train, Z_test, C, k):
    M = Z_train.shape[1] // C
    N_tr = Z_train.shape[0]
    Z_tr_unfolded = Z_train.reshape(N_tr, C, M).transpose(0, 1, 2).reshape(N_tr * C, M)
    mu_h = np.mean(Z_tr_unfolded, axis=0, keepdims=True)
    Z_tr_cen = Z_tr_unfolded - mu_h
    K_gram = Z_tr_cen @ Z_tr_cen.T
    evals, evecs = np.linalg.eigh(K_gram)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    k_actual = min(k, int(np.sum(evals[idx] > 1e-12)))
    inv_sqrt_val = 1.0 / np.sqrt(evals[idx][:k_actual])
    Vt_k = (evecs[:, :k_actual].T * inv_sqrt_val[:, None]) @ Z_tr_cen
    
    Z_test_pred = np.zeros_like(Z_test)
    for c in range(C):
        Z_te_cen_c = Z_test.reshape(-1, C, M)[:, c, :] - mu_h
        coeffs = Z_te_cen_c @ Vt_k.T
        Z_test_pred[:, c * M : (c+1) * M] = (coeffs @ Vt_k) + mu_h
    return Z_test_pred

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = os.path.join(_REPO_ROOT, 'temp_hybrid_cache/data')
    if not os.path.exists(cache_dir): cache_dir = os.path.join(_REPO_ROOT, 'temp_hybrid_cache/v_mean_norm')
    
    with open(os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study/block_config.json'), 'r') as f:
        blocks_config = json.load(f)
    # with open(os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study/stainless_steel_interpolated_expansion_results.json'), 'r') as f:
    #     expansion_coeffs = json.load(f)
        
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(cache_dir) if f.endswith('.npy')])
    N = len(case_names)
    
    # ---------------------------------------------------------
    # MANUSCRIPT ALIGNMENT: 75 Permanently Locked Test Cases
    # ---------------------------------------------------------
    # We reproduce the exact 75 cases that were sequestered by taking the test split of the 
    # original random KFold(n_splits=5, random_state=42) which effectively separated 297 dev / 75 test.
    from sklearn.model_selection import KFold
    kf_initial = KFold(n_splits=5, shuffle=True, random_state=42)
    dev_indices, locked_test_indices = next(kf_initial.split(np.arange(N)))
    
    train_indices = dev_indices
    test_indices = locked_test_indices
    
    # Load parameters for GPR and Stokes filter
    import pandas as pd
    df = pd.read_csv(os.path.join(_REPO_ROOT, 'unnormalized_erosion_vs_stokes_data.csv'))
    df["case_id"] = df["file_name"].apply(lambda x: x.split(".")[0])
    df_sorted = df.set_index("case_id").loc[case_names]
    
    # Filter test cases by St > 1
    valid_test_idx = []
    for i in test_indices:
        if df_sorted.iloc[i]['stokes'] > 1:
            valid_test_idx.append(i)
            
    # Pick first 10 valid test cases
    test_idx = valid_test_idx
    test_cases = [case_names[i] for i in test_idx]
    logger.info(f"Selected {len(test_cases)} test cases with St > 1: {test_cases}")
    
    X_raw_params = df_sorted[["bend_ratio", "velocity", "density", "size"]].values
    pod_param_path = os.path.join(_REPO_ROOT, 'pod_individual_999/X_Parameters.npy')
    
    pi_groups = np.zeros((N, 4))
    rho_f, mu_f, D = 998.2, 0.001003, 0.0508
    pi_groups[:, 0] = rho_f * X_raw_params[:, 1] * D / mu_f  # Re
    pi_groups[:, 1] = X_raw_params[:, 2] / rho_f             # Density ratio
    pi_groups[:, 2] = (X_raw_params[:, 3] * 1e-6) / D        # Size ratio
    pi_groups[:, 3] = X_raw_params[:, 0]                     # Bend ratio
    
    train_pi = pi_groups[train_indices]
    test_pi = pi_groups[test_idx]
    
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_x_gpr = torch.from_numpy(scaler.fit_transform(train_pi)).float()
    test_x_gpr = torch.from_numpy(scaler.transform(test_pi)).float()
    
    all_vars = []
    for b_info in sorted(blocks_config.values(), key=lambda x: list(blocks_config.keys())[list(blocks_config.values()).index(x)]):
        all_vars.extend(b_info['variables'])
        
    predictions_all = {
        'Actual': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32),
        'POD': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32),
        'HOSVD': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32),
        'Block-CNN': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32),
        'All-CNN': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32),
        'GPR': np.zeros((len(test_idx), len(all_vars), 582, 221), dtype=np.float32)
    }
    
    # ---------------------------------------------------------
    # PROCESS BLOCKS FOR POD, HOSVD, Block-CNN, GPR
    # ---------------------------------------------------------
    var_idx_offset = 0
    for block_id, b_info in sorted(blocks_config.items()):
        variables = b_info['variables']
        C = len(variables)
        L_sum = b_info['summed_latent_size']
        logger.info(f"Processing {block_id} for {C} variables...")
        
        X_raw = np.zeros((N, C, 582, 221), dtype=np.float32)
        for c_idx, var in enumerate(variables):
            var_path = os.path.join(os.path.join(_REPO_ROOT, 'temp_hybrid_cache'), var)
            if not os.path.exists(var_path): var_path = os.path.join(os.path.join(_REPO_ROOT, 'temp_hybrid_cache/data'), var)
            if not os.path.exists(var_path): var_path = os.path.join(os.path.join(_REPO_ROOT, 'temp_hybrid_cache'), var) # Fallback for old cache structure
            for i, c_name in enumerate(case_names):
                X_raw[i, c_idx] = np.load(os.path.join(var_path, f"{c_name}.npy")).reshape(582, 221)
                
        Y_power = apply_power_law_transform(X_raw, 0.5)
        M_factors = np.max(np.abs(Y_power.reshape(N, C, -1)), axis=-1) + 1e-8
        Z_norm = Y_power / M_factors[:, :, None, None]
        
        Z_train = Z_norm[train_indices].reshape(len(train_indices), -1)
        Z_test_N = Z_norm[test_idx].reshape(len(test_idx), -1)
        
        mu = np.mean(Z_train, axis=0, keepdims=True)
        Z_train_c = Z_train - mu
        Z_test_c = Z_test_N - mu
        
        # --- POD ---
        Z_pred_pod, Vt_k, c_train, c_test = compute_pod_snapshot_method(Z_train_c, Z_test_c, L_sum)
        Z_pred_pod += mu
        
        # --- HOSVD ---
        Z_pred_hosvd = compute_hosvd_snapshot(Z_train, Z_test_N, C, L_sum)
        
        # --- GPR ---
        
        # --- Block-CNN ---
        cnn = BlockMultiChannelCNN(in_channels=C, latent_dim=L_sum).to(device)
        model_path = os.path.join(_REPO_ROOT, f'block_pod_powerlaw_study/models/freq_weighted_cnn_block_{block_id}_fold_0.pt')
        if os.path.exists(model_path):
            cnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        else:
            model_path = os.path.join(_REPO_ROOT, f'block_pod_powerlaw_study/models/standard_cnn_block_{block_id}_fold_0.pt')
            cnn.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        cnn.eval()
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        Z_test_tensor = torch.from_numpy(Z_norm[test_idx]).float()
        Z_test_512 = F.interpolate(Z_test_tensor, size=(512, 256), mode='bilinear', align_corners=True).to(device)
        with torch.no_grad():
            Z_preds = []
            for b_i in range(0, Z_test_512.shape[0], 5):
                pred_batch = cnn(Z_test_512[b_i:b_i+5])
                pred_interp = F.interpolate(pred_batch, size=(582, 221), mode='bilinear', align_corners=True)
                Z_preds.append(pred_interp.cpu().numpy().reshape(pred_interp.shape[0], -1))
            Z_pred_cnn = np.concatenate(Z_preds, axis=0)
        
        # Inverse transform
        M_N = M_factors[test_idx]
        for c_idx in range(C):
            global_idx = var_idx_offset + c_idx
            # Actual
            predictions_all['Actual'][:, global_idx] = X_raw[test_idx, c_idx]
            # POD
            predictions_all['POD'][:, global_idx] = inverse_power_law_transform(Z_pred_pod.reshape(len(test_idx), C, 582, 221)[:, c_idx] * M_N[:, c_idx, None, None], 0.5)
            # HOSVD
            predictions_all['HOSVD'][:, global_idx] = inverse_power_law_transform(Z_pred_hosvd.reshape(len(test_idx), C, 582, 221)[:, c_idx] * M_N[:, c_idx, None, None], 0.5)
            # Block-CNN
            predictions_all['Block-CNN'][:, global_idx] = inverse_power_law_transform(Z_pred_cnn.reshape(len(test_idx), C, 582, 221)[:, c_idx] * M_N[:, c_idx, None, None], 0.5)
            # GPR
            
        var_idx_offset += C
        del X_raw, Z_train, Y_power
        gc.collect()

    # ---------------------------------------------------------
    # PROCESS All-CNN
    # ---------------------------------------------------------
    logger.info("Processing All-CNN...")
    all_cnn_vars = sorted(list(set(all_vars)))
    X_raw_all = np.zeros((N, len(all_cnn_vars), 582, 221), dtype=np.float32)
    for c_idx, var in enumerate(all_cnn_vars):
        var_path = os.path.join(os.path.join(_REPO_ROOT, 'temp_hybrid_cache'), var)
        if not os.path.exists(var_path): var_path = os.path.join(os.path.join(_REPO_ROOT, 'temp_hybrid_cache/data'), var)
        for i, c_name in enumerate(case_names):
            X_raw_all[i, c_idx] = np.load(os.path.join(var_path, f"{c_name}.npy")).reshape(582, 221)
            
    Y_power = apply_power_law_transform(X_raw_all, 0.5)
    M_factors_all = np.max(np.abs(Y_power.reshape(N, 23, -1)), axis=-1) + 1e-8
    Z_norm_all = Y_power / M_factors_all[:, :, None, None]
    
    all_cnn = MultiChannelCNN(in_channels=23, latent_dim=517).to(device)
    all_cnn.load_state_dict(torch.load(os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study/models/all_channels_cnn_fold_0.pt'), map_location=device, weights_only=True))
    all_cnn.eval()
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    
    Z_test_tensor = torch.from_numpy(Z_norm_all[test_idx]).float()
    Z_test_512 = F.interpolate(Z_test_tensor, size=(512, 256), mode='bilinear', align_corners=True).to(device)
    with torch.no_grad():
        Z_preds_all = []
        for b_i in range(0, Z_test_512.shape[0], 5):
            pred_batch = all_cnn(Z_test_512[b_i:b_i+5])
            pred_interp = F.interpolate(pred_batch, size=(582, 221), mode='bilinear', align_corners=True)
            Z_preds_all.append(pred_interp.cpu().numpy())
        Z_pred_all = np.concatenate(Z_preds_all, axis=0)
    
    M_N_all = M_factors_all[test_idx]
    for c_idx, var in enumerate(all_cnn_vars):
        global_idx = all_vars.index(var)
        predictions_all['All-CNN'][:, global_idx] = inverse_power_law_transform(Z_pred_all[:, c_idx] * M_N_all[:, c_idx, None, None], 0.5)
        
    del X_raw_all, Y_power, Z_norm_all
    gc.collect()

    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # CALCULATE METRICS FOR FINNIE, MCLAURY, AND ARABNEJAD
    # ---------------------------------------------------------
    from sklearn.metrics import r2_score
    import math

    def calc_metrics(a, p, vmax):
        esq = (a - p)**2
        mse = float(np.mean(esq))
        
        ss_tot = np.sum((a - np.mean(a))**2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        
        M_val = np.max(np.abs(a)) + 1e-8
        W = 1.0 + 2.0 * (np.abs(a) / M_val)
        fw_mse = float(np.average(esq, weights=W))
        
        a_w_mean = np.average(a, weights=W)
        ss_tot_w = float(np.average((a - a_w_mean)**2, weights=W))
        if ss_tot_w > 0:
            fw_r2 = float(1.0 - (fw_mse / ss_tot_w))
        else:
            fw_r2 = 0.0
            
        dynamic_range = float(max(1e-5, vmax))
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = float(10 * np.log10((dynamic_range ** 2) / (mse + 1e-15)))
            
        t_flat = a.flatten()
        p_flat = p.flatten()
        cov = np.cov(t_flat, p_flat)[0, 1]
        var_t, var_p = np.var(t_flat), np.var(p_flat)
        c1, c2 = (0.01 * dynamic_range)**2, (0.03 * dynamic_range)**2
        ssim = float(((2 * np.mean(t_flat) * np.mean(p_flat) + c1) * (2 * cov + c2)) /
                     ((np.mean(t_flat)**2 + np.mean(p_flat)**2 + c1) * (var_t + var_p + c2)))
                     
        return mse, r2, fw_mse, fw_r2, ssim, psnr

    methods = ['Actual', 'POD', 'HOSVD', 'Block-CNN', 'All-CNN', 'Composite']
    predictive_methods = ['POD', 'HOSVD', 'Block-CNN', 'All-CNN', 'Composite']
    
    v_idx = all_vars.index('v_mean_norm')
    v2_idx = all_vars.index('v2_mean_norm')
    v2_alpha_idx = all_vars.index('v2_alpha_norm')

    erosion_models = ['Finnie', 'McLaury', 'Arabnejad']
    metrics = {model: {m: {'mse':[], 'r2':[], 'fw_mse':[], 'fw_r2':[], 'ssim':[], 'psnr':[]} for m in predictive_methods} for model in erosion_models}

    # --- COMPOSITE ROM --- 
    # v_mean_norm: Best is POD
    # v2_mean_norm: Best is Block-CNN
    # v2_alpha_norm: Best is POD
    predictions_all['Composite'] = np.zeros_like(predictions_all['POD'])
    predictions_all['Composite'][:, v_idx] = predictions_all['POD'][:, v_idx]
    predictions_all['Composite'][:, v2_idx] = predictions_all['Block-CNN'][:, v2_idx]
    predictions_all['Composite'][:, v2_alpha_idx] = predictions_all['POD'][:, v2_alpha_idx]

    for idx_case, c_name in enumerate(test_cases):
        maps = {model: {} for model in erosion_models}
        
        for method in methods:
            V_mean = predictions_all[method][idx_case, v_idx]
            V2_mean = predictions_all[method][idx_case, v2_idx]
            v2_alpha = predictions_all[method][idx_case, v2_alpha_idx]
            
            V_mean = np.nan_to_num(np.clip(V_mean, 0.0, None), nan=0.0)
            V2_mean = np.nan_to_num(np.clip(V2_mean, 0.0, None), nan=0.0)
            
            A = np.zeros_like(V2_mean)
            valid_mask = V2_mean > 1e-6
            A[valid_mask] = v2_alpha[valid_mask] / V2_mean[valid_mask]
            A = np.nan_to_num(np.clip(A, 0.0, None), nan=0.0)
            
            var_V = np.clip(V2_mean - (V_mean ** 2), 0.0, None)
            
            # --- Finnie ---
            finnie = np.zeros_like(V_mean)
            mask1 = A <= 0.32288
            mask2 = A > 0.32288
            finnie[mask1] = V2_mean[mask1] * (np.sin(2*A[mask1]) - 3*(np.sin(A[mask1])**2))
            finnie[mask2] = V2_mean[mask2] * (1.0/3.0) * (np.cos(A[mask2])**2)
            maps['Finnie'][method] = np.nan_to_num(np.clip(finnie, 0.0, None), nan=0.0)
            
            # --- McLaury ---
            v_base_mcl = V_mean ** 1.73
            correction_mcl = np.zeros_like(V_mean)
            nonzero_mask = V_mean > 1e-6
            correction_mcl[nonzero_mask] = 0.63145 * (V_mean[nonzero_mask] ** (-0.27)) * var_V[nonzero_mask]
            V_jensen_mcl = v_base_mcl + correction_mcl
            
            f_theta_mcl = -34.79 * (A**2) + 12.3 * A
            f_theta_mcl = np.nan_to_num(np.clip(f_theta_mcl, 0.0, None), nan=0.0)
            mclaury = V_jensen_mcl * f_theta_mcl
            maps['McLaury'][method] = np.nan_to_num(np.clip(mclaury, 0.0, None), nan=0.0)
            
            # --- Arabnejad ---
            V_jensen_arab = (V_mean ** 2.41) + 1.69905 * (V_mean ** 0.41) * var_V
            thresh = np.arctan(0.4)
            mask_a1 = A <= thresh
            mask_a2 = A > thresh
            
            vol_c = np.zeros_like(V_mean)
            vol_c[mask_a1] = V_jensen_arab[mask_a1] * np.sin(A[mask_a1]) * (2*0.4*np.cos(A[mask_a1]) - np.sin(A[mask_a1])) / (2 * 0.4**2)
            vol_c[mask_a2] = V_jensen_arab[mask_a2] * (np.cos(A[mask_a2])**2) / 2.0
            vol_c = np.nan_to_num(np.clip(vol_c, 0.0, None), nan=0.0)
            
            vol_d = 0.5 * V2_mean * (np.sin(A)**2)
            vol_d = np.nan_to_num(np.clip(vol_d, 0.0, None), nan=0.0)
            maps['Arabnejad'][method] = vol_c + vol_d
            
        for emodel in erosion_models:
            actual_map = maps[emodel]['Actual']
            vmax = np.max(actual_map)
            if vmax == 0: vmax = 1e-8
            
            for method in predictive_methods:
                pred_map = maps[emodel][method]
                mse, r2, fw_mse, fw_r2, s, p = calc_metrics(actual_map, pred_map, vmax)
                metrics[emodel][method]['mse'].append(mse)
                metrics[emodel][method]['r2'].append(r2)
                metrics[emodel][method]['fw_mse'].append(fw_mse)
                metrics[emodel][method]['fw_r2'].append(fw_r2)
                metrics[emodel][method]['ssim'].append(s)
                metrics[emodel][method]['psnr'].append(p)
            
    for emodel in erosion_models:
        logger.info(f"============== FINAL RESULTS FOR {emodel.upper()} (Fold 0, St > 1) ==============")
        for m in predictive_methods:
            logger.info(f"Method: {m}")
            for met in ['mse', 'r2', 'fw_mse', 'fw_r2', 'ssim', 'psnr']:
                vals = metrics[emodel][m][met]
                if len(vals) == 0: continue
                best = np.min(vals) if met in ['mse', 'fw_mse'] else np.max(vals)
                worst = np.max(vals) if met in ['mse', 'fw_mse'] else np.min(vals)
                worst_idx = np.argmax(vals) if met in ['mse', 'fw_mse'] else np.argmin(vals)
                worst_case = test_cases[worst_idx]
                avg = np.mean(vals)
                logger.info(f"  {met.upper()}: Avg={avg:.4e} | Best={best:.4e} | Worst={worst:.4e} (Case: {worst_case})")

if __name__ == '__main__':
    main()
