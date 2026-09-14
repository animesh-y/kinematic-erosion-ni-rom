#!/usr/bin/env python3
import os
import sys

# Configure PyTorch allocator to prevent fragmentation and enable expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import time
import json
import logging
import gc
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

# Setup logger
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEGENERATE_VARS = {'dp_x_norm', 'dp_y_norm', 'dp_z_norm', 'dp_mag_norm', 'dke_norm', 'frequency'}
COORD_COLS = {'s', 'r', 'theta', 's_normalized'}

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        act = nn.GELU
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), act(), nn.MaxPool2d(2, 2), # 16x256x128
            nn.Conv2d(16, 32, 3, 1, 1), act(), nn.MaxPool2d(2, 2), # 32x128x64
            nn.Conv2d(32, 64, 3, 1, 1), act(), nn.MaxPool2d(2, 2), # 64x64x32
            nn.Conv2d(64, 128, 3, 1, 1), act(), nn.MaxPool2d(2, 2)) # 128x32x16
        self.fc_encode = nn.Linear(128 * 32 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 32 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2), act(), # 64x64x32
            nn.ConvTranspose2d(64, 32, 2, 2), act(), # 32x128x64
            nn.ConvTranspose2d(32, 16, 2, 2), act(), # 16x256x128
            nn.ConvTranspose2d(16, 1, 2, 2)) # 1x512x256

    def encode(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        return self.fc_encode(h)

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 128, 32, 16)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.encode(x))

def compute_global_ssim(x, y, L):
    """
    Computes Structural Similarity Index (SSIM) globally between 1D arrays of physical fields.
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mu_x) * (y - mu_y))
    
    k1, k2 = 0.01, 0.03
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    
    num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)
    return float(num / den)

def eval_field_errors(S_actual, S_pred, test_idx, case_names):
    """
    Evaluates reconstruction performance using multiple metrics (R2, Rel L2, MSE, PSNR, SSIM)
    averaged across test cases and case-wise.
    """
    cases = []
    r2_list = []
    rel_l2_list = []
    mse_list = []
    psnr_list = []
    ssim_list = []
    
    for i in range(len(test_idx)):
        a = S_actual[test_idx[i]]
        p = S_pred[i]
        
        # 1. MSE (physical units)
        esq = (a - p) ** 2
        mse = float(np.mean(esq))
        mse_list.append(mse)
        
        # 2. R2 Score
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        r2_list.append(r2)
        
        # 3. Relative L2 error (%)
        actual_norm = np.linalg.norm(a)
        err_norm = np.linalg.norm(a - p)
        rel_l2 = float(err_norm / (actual_norm + 1e-12) * 100)
        rel_l2_list.append(rel_l2)
        
        # 4. PSNR
        dynamic_range = float(max(1e-5, np.max(a) - np.min(a)))
        psnr = float(10 * np.log10((dynamic_range ** 2) / (mse + 1e-15)))
        psnr_list.append(psnr)
        
        # 5. SSIM
        ssim = compute_global_ssim(a, p, dynamic_range)
        ssim_list.append(ssim)
        
        cases.append({
            'case_name': case_names[test_idx[i]],
            'r2': r2,
            'rel_l2_pct': rel_l2,
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        })
        
    return {
        'r2': float(np.mean(r2_list)),
        'rel_l2_pct': float(np.mean(rel_l2_list)),
        'mse': float(np.mean(mse_list)),
        'psnr': float(np.mean(psnr_list)),
        'ssim': float(np.mean(ssim_list)),
        'cases': cases
    }

def inverse_transform(pred_norm, M_true, method, eps=1e-8):
    trans = pred_norm * M_true[:, None]
    if method == 'cube-root': return trans ** 3
    if method == 'square-root': return trans ** 2 * np.sign(trans)
    if method == 'log-transform': return np.exp(trans) - eps
    return trans

def get_latent_dims(k_95):
    if k_95 <= 2:
        return [1]
    elif k_95 == 3:
        return [1, 2]
    elif k_95 == 4:
        return [1, 2, 3]
    elif k_95 == 5:
        return [1, 2, 4]
    else:
        l1 = max(1, int(k_95 * 0.25))
        l2 = max(2, int(k_95 * 0.50))
        l3 = max(3, int(k_95 * 0.75))
        dims = sorted(list(set([l1, l2, l3])))
        dims = [d for d in dims if d < k_95]
        if len(dims) < 3:
            d3 = k_95 - 1
            d2 = max(2, k_95 // 2)
            d1 = max(1, k_95 // 4)
            if d2 == d3: d2 = d3 - 1
            if d1 == d2: d1 = d2 - 1
            return sorted([d1, d2, d3])
        return dims

def train_and_eval_global(ld, S_train_gpu, S_test_gpu, W_train_gpu, W_test_gpu, total_epochs, batch_size, loss_func, model_save_dir, var, weighted, s_min_val, s_max_val, M, test_idx, norm_method, S, grid_s, grid_theta, case_names, eps):
    """
    Global training function to avoid Python closures keeping references in memory.
    Also tracks training/validation loss curves and generalization gaps.
    """
    model = ConvAutoencoder(latent_dim=ld).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = GradScaler('cuda')
    N_train = S_train_gpu.shape[0]
    
    train_loss_history = []
    val_loss_history = []
    gen_gap_history = []
    
    for epoch in range(total_epochs):
        # 1. Training step
        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        indices = torch.randperm(N_train, device='cuda')
        for start_idx in range(0, N_train, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            xb = S_train_gpu[batch_idx]
            
            opt.zero_grad()
            with autocast('cuda'):
                pred = model(xb)
                if weighted:
                    wb = W_train_gpu[batch_idx]
                    if loss_func == 'l1':
                        loss = torch.mean(wb * torch.abs(pred - xb))
                    else:
                        loss = torch.mean(wb * (pred - xb)**2)
                else:
                    if loss_func == 'l1':
                        loss = F.l1_loss(pred, xb)
                    else:
                        loss = F.mse_loss(pred, xb)
                        
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            train_loss_sum += float(loss.item())
            train_steps += 1
            
        epoch_train_loss = train_loss_sum / train_steps
        train_loss_history.append(epoch_train_loss)
        
        # 2. Validation step
        model.eval()
        val_loss_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for start_idx in range(0, S_test_gpu.shape[0], batch_size):
                xb_val = S_test_gpu[start_idx : start_idx + batch_size]
                with autocast('cuda'):
                    pred_val = model(xb_val)
                    if weighted:
                        wb_val = W_test_gpu[start_idx : start_idx + batch_size]
                        if loss_func == 'l1':
                            loss_val = torch.mean(wb_val * torch.abs(pred_val - xb_val))
                        else:
                            loss_val = torch.mean(wb_val * (pred_val - xb_val)**2)
                    else:
                        if loss_func == 'l1':
                            loss_val = F.l1_loss(pred_val, xb_val)
                        else:
                            loss_val = F.mse_loss(pred_val, xb_val)
                val_loss_sum += float(loss_val.item())
                val_steps += 1
                
        epoch_val_loss = val_loss_sum / val_steps
        val_loss_history.append(epoch_val_loss)
        gen_gap_history.append(epoch_val_loss - epoch_train_loss)

    # Save Model
    model_type = "wgt" if weighted else "unw"
    save_path = os.path.join(model_save_dir, f"{var}_dim{ld}_{model_type}.pt")
    torch.save(model.state_dict(), save_path)
    
    # Final physical evaluation
    model.eval()
    with torch.no_grad():
        pred_test = model(S_test_gpu)
        pred_test_582 = F.interpolate(pred_test, size=(grid_s, grid_theta), mode='bilinear', align_corners=True)
        pred_np = pred_test_582.cpu().numpy().reshape(len(test_idx), -1)
        
    pred_trans = pred_np * (s_max_val - s_min_val) + s_min_val
    pred_phys_test = inverse_transform(pred_trans, M[test_idx], norm_method, eps)
    
    metrics = eval_field_errors(S, pred_phys_test, test_idx, case_names)
    metrics['train_loss_history'] = train_loss_history
    metrics['val_loss_history'] = val_loss_history
    metrics['gen_gap_history'] = gen_gap_history
    
    # Clean up CUDA allocations
    del model, opt, scaler
    gc.collect()
    torch.cuda.empty_cache()
    return metrics

def process_variable(var, train_idx, test_idx, case_names, tmp_dir, output_dir, model_save_dir, n_spatial, grid_s, grid_theta, eps=1e-8):
    # Setup logger
    v_logger = logging.getLogger(f"Worker-{var}")
    v_logger.setLevel(logging.INFO)
    
    torch.set_num_threads(1)
    var_json_path = os.path.join(output_dir, 'var_results', f'{var}.json')
    
    # Load weight mask
    mask_var_dir = os.path.join(tmp_dir, 'frequency_norm')
    N = len(case_names)
    W_mask_raw = np.zeros((N, n_spatial), dtype=np.float32)
    for i, cn in enumerate(case_names):
        W_mask_raw[i] = np.load(os.path.join(mask_var_dir, f'{cn}.npy'))
    W_log = np.log(np.clip(W_mask_raw, 0.0, None) + eps)
    W_scaled = (W_log - float(W_log.min())) / (float(W_log.max()) - float(W_log.min()) + eps)
    
    W_tensor_all = torch.from_numpy(W_scaled.reshape(N, 1, grid_s, grid_theta)).cuda().float()
    W_tensor_512 = F.interpolate(W_tensor_all, size=(512, 256), mode='bilinear', align_corners=True)
    W_train_gpu = W_tensor_512[train_idx]
    W_test_gpu = W_tensor_512[test_idx]
    
    v_logger.info(f"Loading snapshots for {var}...")
    S = np.zeros((N, n_spatial), dtype=np.float32)
    for i, cn in enumerate(case_names):
        S[i] = np.load(os.path.join(tmp_dir, var, f'{cn}.npy'))

    # Transformation Logic
    loss_func = 'mse'
    if '3' in var or '25' in var:
        norm_method = "cube-root"
        S_train_transform = np.sign(S) * np.abs(S) ** (1.0/3.0)
        loss_func = 'l1'
    elif '2' in var and '25' not in var:
        norm_method = "square-root"
        S_train_transform = np.sign(S) * np.abs(S) ** (1.0/2.0)
        loss_func = 'l1'
    else:
        S_flat = S.flatten()
        S_active = S_flat[np.abs(S_flat) > 1e-6]
        skewness = scipy.stats.skew(S_active) if len(S_active) > 0 else 0
        peak_to_mean = np.max(np.abs(S)) / (np.mean(np.abs(S)) + eps)
        if skewness > 5.0 or peak_to_mean > 50 or 'erosion' in var:
            norm_method = "log-transform"
            S_train_transform = np.log(np.clip(S, 0.0, None) + eps)
        else:
            norm_method = "max-normalization"
            S_train_transform = S
            
    M = np.max(np.abs(S_train_transform), axis=1) + eps
    S_train_norm = S_train_transform / M[:, None]
    
    # POD Evaluation
    mean_transform = np.mean(S_train_norm, axis=0)
    S_c = S_train_norm - mean_transform
    U_s, sig, Vt = np.linalg.svd(S_c, full_matrices=False)
    cum_e = np.cumsum(sig ** 2) / (np.sum(sig ** 2) + 1e-30)
    k_95 = max(int(np.argmax(cum_e >= 0.95) + 1), 1)
    
    latent_dims = get_latent_dims(k_95)
    v_logger.info(f"{var} | k_95 = {k_95} | Latent dims to test: {latent_dims}")
    
    # Skip checking (re-run if metrics not present)
    if os.path.exists(var_json_path):
        try:
            with open(var_json_path, 'r') as f:
                existing = json.load(f)
            has_all_dims = True
            for ld in latent_dims:
                d_str = f'dim_{ld}'
                if (d_str not in existing.get('pod', {}) or 
                    d_str not in existing.get('cnn_unw', {}) or 
                    'ssim' not in existing.get('cnn_unw', {}).get(d_str, {}) or
                    'train_loss_history' not in existing.get('cnn_unw', {}).get(d_str, {})):
                    has_all_dims = False
                    break
            if has_all_dims:
                v_logger.info(f"Skipping {var}: already fully computed with new metrics.")
                return
        except Exception:
            pass
            
    var_res = {
        'norm_method': norm_method,
        'loss_func': loss_func,
        'k_95_modes': k_95,
        'pod': {},
        'cnn_unw': {},
        'cnn_w': {}
    }

    # Record 95% POD reconstruction error
    Vt_k = Vt[:k_95]
    Z_pod = S_c[test_idx] @ Vt_k.T
    S_trans_pred_pod = Z_pod @ Vt_k + mean_transform
    S_pred_phys_pod = inverse_transform(S_trans_pred_pod, M[test_idx], norm_method, eps)
    pod_95_metrics = eval_field_errors(S, S_pred_phys_pod, test_idx, case_names)
    pod_95_metrics['modes'] = k_95
    var_res['pod']['95'] = pod_95_metrics

    # Pre-Interpolation on GPU
    s_min_val, s_max_val = float(S_train_norm.min()), float(S_train_norm.max())
    S_scaled = (S_train_norm - s_min_val) / (s_max_val - s_min_val + eps)
    S_scaled_2d = S_scaled.reshape(N, 1, grid_s, grid_theta)
    
    S_tensor_all = torch.from_numpy(S_scaled_2d).cuda().float()
    S_tensor_512 = F.interpolate(S_tensor_all, size=(512, 256), mode='bilinear', align_corners=True)
    
    S_train_gpu = S_tensor_512[train_idx]
    S_test_gpu = S_tensor_512[test_idx]
    
    total_epochs = 500
    batch_size = 64
    torch.backends.cudnn.benchmark = True

    for ld in latent_dims:
        v_logger.info(f"Training {var} at latent dim {ld}...")
        
        # POD at matching dimension
        Vt_ld = Vt[:ld]
        Z_pod_ld = S_c[test_idx] @ Vt_ld.T
        S_trans_pred_pod_ld = Z_pod_ld @ Vt_ld + mean_transform
        S_pred_phys_pod_ld = inverse_transform(S_trans_pred_pod_ld, M[test_idx], norm_method, eps)
        pod_ld_metrics = eval_field_errors(S, S_pred_phys_pod_ld, test_idx, case_names)
        pod_ld_metrics['modes'] = ld
        var_res['pod'][f'dim_{ld}'] = pod_ld_metrics

        # Train Unweighted
        unw_metrics = train_and_eval_global(
            ld, S_train_gpu, S_test_gpu, W_train_gpu, W_test_gpu, total_epochs, batch_size, 
            loss_func, model_save_dir, var, False, s_min_val, s_max_val, M, 
            test_idx, norm_method, S, grid_s, grid_theta, case_names, eps
        )
        var_res['cnn_unw'][f'dim_{ld}'] = unw_metrics
        
        # Train Weighted
        w_metrics = train_and_eval_global(
            ld, S_train_gpu, S_test_gpu, W_train_gpu, W_test_gpu, total_epochs, batch_size, 
            loss_func, model_save_dir, var, True, s_min_val, s_max_val, M, 
            test_idx, norm_method, S, grid_s, grid_theta, case_names, eps
        )
        var_res['cnn_w'][f'dim_{ld}'] = w_metrics
        
        v_logger.info(f"{var} | Dim {ld:2d} | POD R2: {pod_ld_metrics['r2']:.4f} | Unw R2: {unw_metrics['r2']:.4f} | Wgt R2: {w_metrics['r2']:.4f}")

    with open(var_json_path, 'w') as f:
        json.dump(var_res, f, indent=4)
        
    del S, S_train_norm, S_c, U_s, sig, Vt, S_tensor_all, S_tensor_512, S_train_gpu, S_test_gpu, W_tensor_all, W_tensor_512, W_train_gpu, W_test_gpu
    gc.collect()
    torch.cuda.empty_cache()
    v_logger.info(f"Worker for {var} completed successfully and released all VRAM.")

def combine_results(output_dir):
    """
    Combines all individual variable JSON results into a single results.json.
    """
    var_results_dir = os.path.join(output_dir, 'var_results')
    combined = {}
    if os.path.exists(var_results_dir):
        for f_name in sorted(os.listdir(var_results_dir)):
            if f_name.endswith('.json'):
                var = f_name.replace('.json', '')
                with open(os.path.join(var_results_dir, f_name), 'r') as f:
                    combined[var] = json.load(f)
                    
    master_path = os.path.join(output_dir, 'results.json')
    with open(master_path, 'w') as f:
        json.dump(combined, f, indent=4)
    logger.info(f"Updated master results file: {master_path}")
    return combined

def main():
    input_dir = 'trimmed_data_csv'
    output_dir = 'latent_dim_sensitivity_study'
    model_save_dir = os.path.join(output_dir, 'models')
    tmp_dir = 'temp_hybrid_cache'
    grid_s, grid_theta = 582, 221
    n_spatial = grid_s * grid_theta
    eps = 1e-8
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'var_results'), exist_ok=True)

    # Active variables + synthetic erosion variables
    sample = pd.read_csv(os.path.join(input_dir, sorted(os.listdir(input_dir))[0]))
    base_vars = [c for c in sample.columns if c not in COORD_COLS and c not in DEGENERATE_VARS and 'variance' not in c]
    synth_erosion_vars = ['finnie_erosion', 'mclaury_erosion', 'arabnejad_erosion']
    all_vars = base_vars + synth_erosion_vars

    mask_var_dir = os.path.join(tmp_dir, 'frequency_norm')
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(mask_var_dir) if f.endswith('.npy')])
    N = len(case_names)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    # Import any pre-existing master results.json to avoid losing data
    master_path = os.path.join(output_dir, 'results.json')
    if os.path.exists(master_path):
        try:
            with open(master_path, 'r') as f:
                old_results = json.load(f)
            for var, val in old_results.items():
                var_json_path = os.path.join(output_dir, 'var_results', f'{var}.json')
                if not os.path.exists(var_json_path):
                    with open(var_json_path, 'w') as f_out:
                        json.dump(val, f_out, indent=4)
            logger.info("Imported pre-existing master results into var_results directory.")
        except Exception as e:
            logger.warning(f"Could not import old master results: {e}")

    # Determine variables that need processing
    vars_to_process = []
    for var in all_vars:
        vars_to_process.append(var)

    logger.info(f"Starting latent dimension tuning study for {len(vars_to_process)} variables in parallel...")
    
    # We will use spawn start method for CUDA safety in multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Concurrency limit (2 workers to avoid GPU memory OOM)
    n_workers = 2
    
    active_processes = []
    
    for var in vars_to_process:
        # Check if already processed with new metrics
        var_json_path = os.path.join(output_dir, 'var_results', f'{var}.json')
        if os.path.exists(var_json_path):
            try:
                with open(var_json_path, 'r') as f:
                    data = json.load(f)
                has_new_metrics = True
                for key in ['cnn_unw', 'cnn_w']:
                    for dim_key, dim_val in data.get(key, {}).items():
                        if 'ssim' not in dim_val or 'train_loss_history' not in dim_val:
                            has_new_metrics = False
                            break
                if has_new_metrics:
                    logger.info(f"Skipping {var}: already completed with new metrics.")
                    continue
            except Exception:
                pass
                
        # Wait until we have fewer than n_workers active processes
        while len([p for v, p in active_processes if p.is_alive()]) >= n_workers:
            for v, p in active_processes:
                if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
                    logger.error(f"Worker process for variable {v} failed with exit code {p.exitcode}!")
                    raise RuntimeError(f"Worker process for variable {v} failed with exit code {p.exitcode}")
            active_processes = [(v, p) for v, p in active_processes if p.is_alive()]
            time.sleep(1)
            combine_results(output_dir)
            
        logger.info(f"Launching worker process for {var}...")
        p = mp.Process(
            target=process_variable, 
            args=(var, train_idx, test_idx, case_names, tmp_dir, output_dir, model_save_dir, n_spatial, grid_s, grid_theta, eps)
        )
        p.start()
        active_processes.append((var, p))
        
    # Wait for remaining processes to finish
    while len(active_processes) > 0:
        for v, p in active_processes:
            if not p.is_alive() and p.exitcode is not None and p.exitcode != 0:
                logger.error(f"Worker process for variable {v} failed with exit code {p.exitcode}!")
                raise RuntimeError(f"Worker process for variable {v} failed with exit code {p.exitcode}")
        active_processes = [(v, p) for v, p in active_processes if p.is_alive()]
        time.sleep(1)
        combine_results(output_dir)
        
    combine_results(output_dir)
    logger.info("Latent dimension sensitivity study completed successfully!")

if __name__ == '__main__':
    main()
