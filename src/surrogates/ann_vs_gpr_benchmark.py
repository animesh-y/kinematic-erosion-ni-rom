import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

#!/usr/bin/env python3
import os
import sys
import json
import time
import gc
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import gpytorch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# GPyTorch Batch Gaussian Process Model
# ==========================================
class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_models, num_features=6):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_models]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5, 
                ard_num_dims=num_features, 
                batch_shape=torch.Size([num_models])
            ),
            batch_shape=torch.Size([num_models])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_batch_gpr(train_x, train_y, num_features=6, lr=0.1, training_iter=150, verbose=False):
    num_models = train_y.shape[0]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([num_models]))
    model = BatchGPModel(train_x, train_y, likelihood, num_models, num_features=num_features)
    
    try:
        likelihood.noise_covar.initialize(noise=1e-4 * torch.ones(num_models))
    except Exception:
        pass
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    t_start = time.time()
    for i in range(1, training_iter + 1):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            loss.backward()
            optimizer.step()
            if verbose and (i % 50 == 0 or i == 1):
                logger.info(f"  GPR Iteration {i}/{training_iter} | MLL Loss: {loss.item():.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning("  WARNING: GPU OOM encountered during GPR training.")
                torch.cuda.empty_cache()
            raise e
            
    if verbose:
        logger.info(f"  Batch GPR training of {num_models} target models completed in {time.time() - t_start:.2f}s.")
    return model, likelihood

def predict_batch_gpr(model, likelihood, test_x):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    likelihood.eval()
    num_models = model.mean_module.batch_shape[0]
    
    if test_x.ndim == 2:
        test_x_batch = torch.from_numpy(test_x).float().unsqueeze(0).repeat(num_models, 1, 1).to(device)
    else:
        test_x_batch = test_x.to(device)
        
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
        predictions = likelihood(model(test_x_batch))
        pred_mean = predictions.mean
        
    return pred_mean.cpu().numpy()

# ==========================================
# ANN Regression Model
# ==========================================
class CoefficientANN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, activation='relu'):
        super().__init__()
        act = torch.nn.ReLU if activation.lower() == 'relu' else torch.nn.GELU
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            act(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 512),
            act(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            act(),
            torch.nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def train_predict_ann(train_x, train_y, test_x, epochs=3000, lr=1e-3, weight_decay=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoefficientANN(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    criterion = torch.nn.MSELoss()
    
    X_train_internal, X_val_internal, y_train_internal, y_val_internal = train_test_split(train_x, train_y, test_size=0.15, random_state=42)
    
    train_x_t = torch.from_numpy(X_train_internal).float().to(device)
    train_y_t = torch.from_numpy(y_train_internal).float().to(device)
    val_x_t = torch.from_numpy(X_val_internal).float().to(device)
    val_y_t = torch.from_numpy(y_val_internal).float().to(device)
    test_x_t = torch.from_numpy(test_x).float().to(device)
    
    best_loss = float('inf')
    best_weights = None
    patience_counter = 0
    
    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_x_t)
        loss = criterion(out, train_y_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(val_x_t)
            val_loss = criterion(val_out, val_y_t)
            
        scheduler.step(val_loss)
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > 500:
                break
                
    logger.info(f"  ANN training completed in {time.time() - t_start:.2f}s (Epochs: {epoch+1}, Best Val Loss: {best_loss:.6f}).")
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        preds = model(test_x_t).cpu().numpy()
    return preds

# ==========================================
# Metric & Transformation Utilities
# ==========================================
def compute_physical_metrics(a, p):
    esq = (a - p) ** 2
    mse = float(np.mean(esq))
    ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
    r2 = float(1.0 - np.sum(esq) / ss_tot)
    actual_norm = np.linalg.norm(a)
    err_norm = np.linalg.norm(a - p)
    rel_l2 = float(err_norm / (actual_norm + 1e-12) * 100.0)
    dynamic_range = float(max(1e-5, np.max(a) - np.min(a)))
    psnr = float(10 * np.log10((dynamic_range ** 2) / (mse + 1e-15)))
    
    a_flat = a.flatten()
    p_flat = p.flatten()
    cov = np.cov(a_flat, p_flat)[0, 1]
    var_a, var_p = np.var(a_flat), np.var(p_flat)
    c1, c2 = (0.01 * dynamic_range)**2, (0.03 * dynamic_range)**2
    ssim = float(((2 * np.mean(a_flat) * np.mean(p_flat) + c1) * (2 * cov + c2)) /
                 ((np.mean(a_flat)**2 + np.mean(p_flat)**2 + c1) * (var_a + var_p + c2)))
    M_val = np.max(np.abs(a), axis=1, keepdims=True) + 1e-8
    W = 1.0 + 2.0 * (np.abs(a) / M_val)
    fw_mse = float(np.mean(W * esq))
    
    # Calculate Frequency-Weighted R2
    # W * a and sum(W) over spatial dimension for each snapshot
    weighted_mean = np.sum(W * a, axis=1, keepdims=True) / (np.sum(W, axis=1, keepdims=True) + 1e-12)
    ss_tot_w = np.sum(W * (a - weighted_mean)**2) + 1e-12
    ss_res_w = np.sum(W * esq)
    fw_r2 = float(1.0 - (ss_res_w / ss_tot_w))
    
    return {'R2': r2, 'Rel_L2_pct': rel_l2, 'SSIM': ssim, 'PSNR': psnr, 'MSE': mse, 'FW_MSE': fw_mse, 'FW_R2': fw_r2}

def apply_power_law_transform(X, power=0.5):
    return np.sign(X) * (np.abs(X) ** power)

def inverse_power_law_transform(Y, power=0.5):
    return np.sign(Y) * (np.abs(Y) ** (1.0 / power))

def compute_pod_basis(Z_train_centered, k):
    """
    Method of Snapshots to obtain orthonormal spatial projection matrix Phi_k.
    """
    K_gram = Z_train_centered @ Z_train_centered.T
    eigenvalues, eigenvectors = np.linalg.eigh(K_gram)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    valid = eigenvalues > 1e-12
    k_actual = min(k, int(np.sum(valid)))
    eigenvalues = eigenvalues[:k_actual]
    eigenvectors = eigenvectors[:, :k_actual]
    
    inv_sqrt_val = 1.0 / np.sqrt(eigenvalues)
    Phi_k_t = (eigenvectors.T * inv_sqrt_val[:, None]) @ Z_train_centered
    return Phi_k_t, k_actual

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU for GPR training.")
    else:
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
        
    cache_dir = os.path.join(_REPO_ROOT, 'temp_hybrid_cache')
    out_dir = os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study')
    config_path = os.path.join(out_dir, 'block_config.json')
    results_save_path = os.path.join(out_dir, 'block_gpr_benchmark_results.json')
    
    with open(config_path, 'r') as f:
        blocks_config = json.load(f)
        
    results_summary = {}
    test_set_predictions = {'truth': {}, 'gpr': {}, 'ann': {}}
    if os.path.exists(results_save_path):
        try:
            with open(results_save_path, 'r') as f:
                results_summary = json.load(f)
            logger.info(f"Loaded {len(results_summary)} blocks from existing benchmark results file.")
        except Exception as e:
            logger.warning(f"Could not read existing results file: {e}")

    # 1. Align cases and generate dimensionless feature parameters
    sample_var = blocks_config['Block_1']['variables'][0]
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(cache_dir, sample_var)) if f.endswith('.npy')])
    N = len(case_names)
    
    import pandas as pd
    df = pd.read_csv(os.path.join(_REPO_ROOT, 'unnormalized_erosion_vs_stokes_data.csv'))
    df["case_id"] = df["file_name"].apply(lambda x: x.split(".")[0])
    df_sorted = df.set_index("case_id").loc[case_names]
    X_raw_params = df_sorted[["bend_ratio", "velocity", "density", "size"]].values
    
    # Compute physical dimensionless feature vector x in R^6
    rho_f = 998.2        # kg/m^3
    mu_f = 0.001003      # Pa.s
    D_pipe = 0.0254      # m
    
    bend_ratio = X_raw_params[:, 0]
    velocity = X_raw_params[:, 1]
    particle_density = X_raw_params[:, 2]
    particle_size_m = X_raw_params[:, 3] * 1e-6
    
    reynolds = (rho_f * velocity * D_pipe) / mu_f
    density_ratio = particle_density / rho_f
    size_ratio = particle_size_m / D_pipe
    dean = reynolds * np.sqrt(1.0 / (2.0 * bend_ratio))
    stokes = (particle_density * (particle_size_m ** 2) * velocity) / (18.0 * mu_f * D_pipe)
    log_stokes = np.log10(stokes + 1e-12)
    
    X_features = np.column_stack((reynolds, density_ratio, size_ratio, bend_ratio))
    logger.info(f"Constructed 4D Dimensionless Parametric Design Space for {N} cases:")
    logger.info(f"  Reynolds Range   : {reynolds.min():.1f} - {reynolds.max():.1f}")
    logger.info(f"  Stokes Range     : {stokes.min():.4e} - {stokes.max():.4e} (LogSt: {log_stokes.min():.2f} to {log_stokes.max():.2f})")
    
    # GLOBALLY FILTER FOR St > 1 TO PREVENT DATA LEAKAGE
    valid_mask = stokes > 1.0
    stokes = stokes[valid_mask]
    reynolds = reynolds[valid_mask]
    density_ratio = density_ratio[valid_mask]
    size_ratio = size_ratio[valid_mask]
    bend_ratio = bend_ratio[valid_mask]
    case_names = [case_names[i] for i in range(N) if valid_mask[i]]
    N = len(case_names)
    X_features = np.column_stack((reynolds, density_ratio, size_ratio, bend_ratio))
    
    train_indices, test_indices = train_test_split(np.arange(N), test_size=0.20, random_state=42, shuffle=True)
    logger.info(f"Data Split: {len(train_indices)} Train | {len(test_indices)} Test")
    
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_features[train_indices])
    X_test_scaled = scaler.transform(X_features[test_indices])
    
    for block_id, b_info in sorted(blocks_config.items()):
        # Re-computing all blocks with standardized target conditioning
        variables = b_info['variables']
        L_sum = b_info['summed_latent_size']
        C = len(variables)
        logger.info(f"\n=======================================================")
        logger.info(f"Running GPR NI-ROM for {block_id} | {C} Variables | Latent Rank = {L_sum}")
        logger.info(f"Variables: {variables}")
        logger.info(f"=======================================================")
        
        X_raw = np.zeros((N, C, 582, 221), dtype=np.float32)
        for c_idx, var in enumerate(variables):
            var_path = os.path.join(cache_dir, var)
            for i, c_name in enumerate(case_names):
                X_raw[i, c_idx] = np.load(os.path.join(var_path, f"{c_name}.npy")).reshape(582, 221)
                
        logger.info("Applying signed power-law transformation (p = 0.5)...")
        Y_power = apply_power_law_transform(X_raw, power=0.5)
        del X_raw
        gc.collect()
        
        M_factors = np.max(np.abs(Y_power.reshape(N, C, -1)), axis=-1) + 1e-8 # Shape: (N, C)
        Z_norm = Y_power / M_factors[:, :, None, None] # Shape: (N, C, 582, 221)
        
        Z_train = Z_norm[train_indices].reshape(len(train_indices), -1) # (297, C*582*221)
        Z_test = Z_norm[test_indices].reshape(len(test_indices), -1)    # (75, C*582*221)
        M_train = M_factors[train_indices] # (297, C)
        M_test = M_factors[test_indices]   # (75, C)
        
        # Compute spatial mean on training set
        mu_train = np.mean(Z_train, axis=0, keepdims=True) # (1, M_dim)
        Z_train_centered = Z_train - mu_train
        
        # Compute POD spatial basis on training set
        logger.info(f"Computing spatial projection basis (k = {L_sum})...")
        Phi_k_t, k_actual = compute_pod_basis(Z_train_centered, k=L_sum) # Phi_k_t: (k, M_dim)
        
        # Compute training modal coefficients
        coeffs_train = (Z_train_centered @ Phi_k_t.T) # Shape: (297, k)
        
        # Prepare GPR target matrix with per-target standard normal scaling
        Y_train_target_raw = np.vstack([coeffs_train.T, M_train.T]).astype(np.float32)
        num_targets = Y_train_target_raw.shape[0]
        
        Y_mean = np.mean(Y_train_target_raw, axis=1, keepdims=True)
        Y_std = np.std(Y_train_target_raw, axis=1, keepdims=True) + 1e-8
        Y_train_target = (Y_train_target_raw - Y_mean) / Y_std
        
        logger.info(f"Training GPU Batch GPR across {num_targets} standardized targets ({k_actual} modes + {C} scaling factors)...")
        
        # Prepare tensors
        train_x_t = torch.from_numpy(X_train_scaled).float().unsqueeze(0).repeat(num_targets, 1, 1)
        train_y_t = torch.from_numpy(Y_train_target).float()
        
        # Train GPR
        start_t = time.time()
        model, likelihood = train_batch_gpr(train_x_t, train_y_t, num_features=4, lr=0.1, training_iter=200, verbose=True)
        logger.info(f"GPR training finalized in {time.time() - start_t:.2f}s.")
        
        # Predict on out-of-sample test parameters
        logger.info("Predicting test modal trajectories and scaling factors via GPR...")
        pred_targets_norm = predict_batch_gpr(model, likelihood, X_test_scaled) # Shape: (num_targets, 75)
        pred_targets = pred_targets_norm * Y_std + Y_mean
        
        pred_coeffs_gpr = pred_targets[:k_actual, :].T # Shape: (75, k)
        pred_M_gpr = np.abs(pred_targets[k_actual:, :].T) + 1e-8 # Shape: (75, C), enforce positive scaling
        
        # Train and Predict with ANN
        logger.info("Training and Predicting test modal trajectories via ANN...")
        Y_test_target_raw = np.vstack([((Z_test.reshape(len(test_indices), -1) - mu_train) @ Phi_k_t.T).T, M_test.T]).astype(np.float32)
        Y_test_target = (Y_test_target_raw - Y_mean) / Y_std
        
        ann_preds_norm = train_predict_ann(X_train_scaled, Y_train_target.T, X_test_scaled)
        ann_targets = ann_preds_norm.T * Y_std + Y_mean
        pred_coeffs_ann = ann_targets[:k_actual, :].T
        pred_M_ann = np.abs(ann_targets[k_actual:, :].T) + 1e-8
        
        # Reconstruct physical manifolds
        logger.info("Reconstructing physical fields and evaluating out-of-sample accuracy...")
        Z_test_pred_gpr = (pred_coeffs_gpr @ Phi_k_t + mu_train).reshape(len(test_indices), C, 582, 221)
        Z_test_pred_ann = (pred_coeffs_ann @ Phi_k_t + mu_train).reshape(len(test_indices), C, 582, 221)
        
        Y_test_true = Y_power[test_indices].reshape(len(test_indices), C, 582, 221)
        
        block_metrics_gpr = {}
        block_metrics_ann = {}
        for c_idx, var in enumerate(variables):
            # GPR Reconstruct
            pred_Y_gpr = Z_test_pred_gpr[:, c_idx] * pred_M_gpr[:, c_idx, None, None]
            pred_X_gpr = inverse_power_law_transform(pred_Y_gpr, power=0.5)
            
            # ANN Reconstruct
            pred_Y_ann = Z_test_pred_ann[:, c_idx] * pred_M_ann[:, c_idx, None, None]
            pred_X_ann = inverse_power_law_transform(pred_Y_ann, power=0.5)
            
            true_Y = Y_test_true[:, c_idx]
            true_X = inverse_power_law_transform(true_Y, power=0.5)
            
            block_metrics_gpr[var] = compute_physical_metrics(true_X, pred_X_gpr)
            block_metrics_ann[var] = compute_physical_metrics(true_X, pred_X_ann)
            
            test_set_predictions['truth'][var] = true_X.reshape(len(test_indices), -1)
            test_set_predictions['gpr'][var] = pred_X_gpr.reshape(len(test_indices), -1)
            test_set_predictions['ann'][var] = pred_X_ann.reshape(len(test_indices), -1)
            
            logger.info(f"  [GPR] {var:<25} | R2: {block_metrics_gpr[var]['R2']:>7.4f} | Rel L2: {block_metrics_gpr[var]['Rel_L2_pct']:>6.2f}%")
            logger.info(f"  [ANN] {var:<25} | R2: {block_metrics_ann[var]['R2']:>7.4f} | Rel L2: {block_metrics_ann[var]['Rel_L2_pct']:>6.2f}%")
            
        results_summary[block_id] = {
            'variables': variables,
            'k_modes_retained': k_actual,
            'gpr_metrics': block_metrics_gpr,
            'ann_metrics': block_metrics_ann
        }
        with open(results_save_path.replace('block_gpr_benchmark_results.json', 'block_gpr_ann_benchmark_results.json'), 'w') as f:
            json.dump(results_summary, f, indent=4)
            
        del Z_train, Z_test, Z_norm, Y_power, model, likelihood, train_x_t, train_y_t
        torch.cuda.empty_cache()
        gc.collect()
        
    np.save(os.path.join(out_dir, 'test_set_predictions.npy'), test_set_predictions)
    logger.info("\nAll blocks completed successfully! Results saved to block_gpr_benchmark_results.json and test_set_predictions.npy.")

if __name__ == '__main__':
    main()
