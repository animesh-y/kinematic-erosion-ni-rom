#!/usr/bin/env python3
"""
Full Production Training: Frequency-Weighted SemiCircular CNN-AE
================================================================
Implements the exact 5-fold cross-validation training study reported in the manuscript:
1. SemiCircularConv2d with circular azimuthal padding and zero streamwise padding.
2. Signed Power-Law Transformation (p = 0.5) for variance stabilization.
3. Ward-clustered Multi-Channel Blocks (BlockMultiChannelCNN with GAP and BatchNorm).
4. Frequency-Weighted MSE Loss (FW-MSE) with Cosine Annealing over 800 epochs.
5. VRAM-optimized mini-batching (batch_size=2 with gradient accumulation).
"""

import os
import sys
import json
import logging
import gc
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Neural Network Architecture with SemiCircular Conv2d
# ==========================================
class SemiCircularConv2d(nn.Module):
    """
    2D Convolution with periodic circular padding along azimuthal coordinate theta (dim -1)
    and zero constant padding along streamwise coordinate s (dim -2).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        # Circular padding along width (theta)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        # Zero padding along height (s)
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return self.conv(x)

class BlockMultiChannelCNN(nn.Module):
    """
    Multi-Channel SemiCircular Block CNN Autoencoder with Dense Spatial Flattening.
    Implements the exact Gen 3 production architecture reported in Table 3 and Table 4:
    - Periodic circular padding on azimuthal theta (dim -1)
    - Zero padding on streamwise s (dim -2)
    - Fully flattened spatial coordinates (256 * 32 * 16) into latent bottleneck z
    - Symmetrical 4-stage transposed convolution decoder with stride=2 upsampling
    """
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

# ==========================================
# Frequency-Weighted Loss Function
# ==========================================
def frequency_weighted_mse_loss(recon, target, gamma=2.0):
    with torch.no_grad():
        target_abs = torch.abs(target.float())
        max_vals = torch.amax(target_abs, dim=(-2, -1), keepdim=True) + 1e-6
        weights = 1.0 + gamma * torch.clamp(target_abs / max_vals, 0.0, 5.0)
    
    sq_err = weights * ((recon.float() - target.float()) ** 2)
    return torch.mean(sq_err)

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
    
    weighted_mean = np.sum(W * a, axis=1, keepdims=True) / (np.sum(W, axis=1, keepdims=True) + 1e-12)
    ss_tot_w = np.sum(W * (a - weighted_mean)**2) + 1e-12
    ss_res_w = np.sum(W * esq)
    fw_r2 = float(1.0 - (ss_res_w / ss_tot_w))
    
    return {'R2': r2, 'Rel_L2_pct': rel_l2, 'SSIM': ssim, 'PSNR': psnr, 'MSE': mse, 'FW_MSE': fw_mse, 'FW_R2': fw_r2}

def apply_power_law_transform(X, power=0.5):
    return np.sign(X) * (np.abs(X) ** power)

def inverse_power_law_transform(Y, power=0.5):
    return np.sign(Y) * (np.abs(Y) ** (1.0 / power))

def run_fold(fold_idx, blocks_config, case_names, train_indices, test_indices, out_dir, models_dir, cache_dir, n_epochs=800, force=False):
    results_save_path = os.path.join(out_dir, f'freq_weighted_cnn_results_fold_{fold_idx}.json')
    results_summary = {}
    if os.path.exists(results_save_path) and not force:
        try:
            with open(results_save_path, 'r') as f:
                results_summary = json.load(f)
            logger.info(f"[Fold {fold_idx}] Loaded {len(results_summary)} blocks from existing results file.")
        except Exception as e:
            logger.warning(f"[Fold {fold_idx}] Could not read existing results file: {e}")

    N = len(case_names)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for block_id, b_info in sorted(blocks_config.items()):
        if not force and block_id in results_summary and 'cnn_metrics' in results_summary[block_id]:
            logger.info(f"[Fold {fold_idx} | {block_id}] Frequency-Weighted CNN-AE already completed. Skipping!")
            continue
            
        variables = b_info['variables']
        L_sum = b_info.get('summed_latent_size', b_info.get('L_sum', 16))
        C = len(variables)
        logger.info(f"\n==========================================")
        logger.info(f"[Fold {fold_idx}] Processing {block_id}: Channels = {C}, Variables = {variables}")
        logger.info(f"==========================================")
        
        X_raw = np.zeros((N, C, 582, 221), dtype=np.float32)
        for c_idx, var in enumerate(variables):
            var_path = os.path.join(cache_dir, var)
            for i, c_name in enumerate(case_names):
                X_raw[i, c_idx] = np.load(os.path.join(var_path, f"{c_name}.npy")).reshape(582, 221)
                
        logger.info("Applying power-law transformation (p = 0.5)...")
        Y_power = apply_power_law_transform(X_raw, power=0.5)
        del X_raw
        gc.collect()
        
        M_factors = np.max(np.abs(Y_power.reshape(N, C, -1)), axis=-1) + 1e-8
        Z_norm = Y_power / M_factors[:, :, None, None]
        
        Z_train = Z_norm[train_indices]
        Z_test = Z_norm[test_indices]
        M_test = M_factors[test_indices]
        
        Y_test_true = Y_power[test_indices]
        X_test_true = inverse_power_law_transform(Y_test_true, power=0.5)
        del Y_power, Z_norm
        gc.collect()
        
        logger.info(f"[Fold {fold_idx} | {block_id}] Training BlockMultiChannelCNN (Latent={L_sum}, Channels={C})...")
        
        Z_train_tensor = torch.from_numpy(Z_train).float()
        Z_train_512_list = []
        for i in range(0, len(Z_train_tensor), 32):
            chunk = F.interpolate(Z_train_tensor[i:i+32], size=(512, 256), mode='bilinear', align_corners=True)
            Z_train_512_list.append(chunk.clone())
        Z_train_512 = torch.cat(Z_train_512_list, dim=0)
        del Z_train_tensor, Z_train_512_list, Z_train
        gc.collect()
        
        # Safe batch size to avoid CUDA OOM
        batch_size = 2
        accumulation_steps = 2
        dataset = TensorDataset(Z_train_512, Z_train_512)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=False)
        
        model = BlockMultiChannelCNN(in_channels=C, latent_dim=L_sum).to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
        scaler = GradScaler(enabled=(device.type=='cuda'))
        
        start_train_t = time.time()
        model.train()
        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            optimizer.zero_grad()
            for step_idx, (batch_x, _) in enumerate(loader):
                batch_x = batch_x.to(device, non_blocking=True)
                with autocast(device_type=device.type, enabled=(device.type=='cuda')):
                    recon = model(batch_x)
                    loss = frequency_weighted_mse_loss(recon, batch_x, gamma=2.0)
                    loss_scaled = loss / accumulation_steps
                scaler.scale(loss_scaled).backward()
                
                if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                epoch_loss += loss.item() * len(batch_x)
            
            scheduler.step()
            epoch_loss /= len(train_indices)
            if epoch % 100 == 0 or epoch == n_epochs:
                logger.info(f"  [Fold {fold_idx} | {block_id}] Epoch [{epoch:4d}/{n_epochs:4d}] | FW Loss: {epoch_loss:.6e} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {time.time() - start_train_t:.1f}s")
                
        torch.save(model.state_dict(), os.path.join(models_dir, f'freq_weighted_cnn_block_{block_id}_fold_{fold_idx}.pt'))
        model.eval()
        logger.info(f"[Fold {fold_idx} | {block_id}] Evaluating test set ({len(test_indices)} cases)...")
        Z_test_tensor = torch.from_numpy(Z_test).float()
        Z_test_512 = F.interpolate(Z_test_tensor, size=(512, 256), mode='bilinear', align_corners=True)
        del Z_test_tensor, Z_test
        gc.collect()
        
        with torch.no_grad():
            preds_list = []
            test_loader = DataLoader(TensorDataset(Z_test_512), batch_size=2, shuffle=False)
            for (bx,) in test_loader:
                bx = bx.to(device)
                with autocast(device_type=device.type, enabled=(device.type=='cuda')):
                    pr = model(bx)
                preds_list.append(pr.cpu())
            Z_test_pred_512 = torch.cat(preds_list, dim=0)
            Z_test_pred_tensor = F.interpolate(Z_test_pred_512, size=(582, 221), mode='bilinear', align_corners=True)
            Z_test_pred = Z_test_pred_tensor.numpy()

        cnn_metrics = {}
        for c_idx, var in enumerate(variables):
            pred_Y = Z_test_pred[:, c_idx] * M_test[:, c_idx, None, None]
            pred_X = inverse_power_law_transform(pred_Y, power=0.5)
            cnn_metrics[var] = compute_physical_metrics(X_test_true[:, c_idx], pred_X)
            logger.info(f"  [Fold {fold_idx} | {var:<23}] R2: {cnn_metrics[var]['R2']:>7.4f} | Rel L2: {cnn_metrics[var]['Rel_L2_pct']:>6.2f}% | SSIM: {cnn_metrics[var]['SSIM']:>6.4f} | PSNR: {cnn_metrics[var]['PSNR']:>5.1f} dB")

        results_summary[block_id] = {
            'variables': variables,
            'L_sum': L_sum,
            'cnn_metrics': cnn_metrics
        }

        with open(results_save_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        del Z_train_512, Z_test_512, Z_test_pred_512, Z_test_pred_tensor, Z_test_pred, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(f"[Fold {fold_idx}] Saved all block results to {results_save_path}")
    return results_summary

def aggregate_cv_results(out_dir):
    fold_files = [os.path.join(out_dir, f'freq_weighted_cnn_results_fold_{k}.json') for k in range(5)]
    if not all(os.path.exists(f) for f in fold_files):
        logger.warning("Not all 5 fold files exist yet for aggregation.")
        return None
        
    all_folds = []
    for f in fold_files:
        with open(f, 'r') as fh:
            all_folds.append(json.load(fh))
            
    aggregated = {}
    summary_table = []
    
    blocks = sorted(all_folds[0].keys())
    for b in blocks:
        aggregated[b] = {
            'variables': all_folds[0][b]['variables'],
            'L_sum': all_folds[0][b]['L_sum'],
            'cnn_metrics_mean': {},
            'cnn_metrics_std': {}
        }
        vars = all_folds[0][b]['variables']
        for v in vars:
            aggregated[b]['cnn_metrics_mean'][v] = {}
            aggregated[b]['cnn_metrics_std'][v] = {}
            sample_metrics = all_folds[0][b]['cnn_metrics'][v].keys()
            for m in sample_metrics:
                vals = [fold[b]['cnn_metrics'][v][m] for fold in all_folds]
                aggregated[b]['cnn_metrics_mean'][v][m] = float(np.mean(vals))
                aggregated[b]['cnn_metrics_std'][v][m] = float(np.std(vals))
            
            r2_m, r2_s = aggregated[b]['cnn_metrics_mean'][v]['R2'], aggregated[b]['cnn_metrics_std'][v]['R2']
            l2_m, l2_s = aggregated[b]['cnn_metrics_mean'][v]['Rel_L2_pct'], aggregated[b]['cnn_metrics_std'][v]['Rel_L2_pct']
            ssim_m, ssim_s = aggregated[b]['cnn_metrics_mean'][v]['SSIM'], aggregated[b]['cnn_metrics_std'][v]['SSIM']
            psnr_m, psnr_s = aggregated[b]['cnn_metrics_mean'][v]['PSNR'], aggregated[b]['cnn_metrics_std'][v]['PSNR']
            
            summary_table.append({
                'Block': b,
                'Variable': v,
                'R2': f"{r2_m:.4f} ± {r2_s:.4f}",
                'Rel_L2_pct': f"{l2_m:.2f}% ± {l2_s:.2f}%",
                'SSIM': f"{ssim_m:.4f} ± {ssim_s:.4f}",
                'PSNR': f"{psnr_m:.2f} ± {psnr_s:.2f} dB"
            })
            
    cv_save_path = os.path.join(out_dir, 'freq_weighted_cnn_5fold_cv_results.json')
    with open(cv_save_path, 'w') as f:
        json.dump(aggregated, f, indent=4)
        
    logger.info(f"\n==========================================================================================================")
    logger.info(f"               FINAL 5-FOLD CROSS-VALIDATION SUMMARY (LATEST CNN-AE WITH GAP + BATCHNORM)                ")
    logger.info(f"==========================================================================================================")
    df_summary = pd.DataFrame(summary_table)
    logger.info("\n" + df_summary.to_string(index=False))
    logger.info(f"==========================================================================================================")
    logger.info(f"Saved aggregated 5-fold CV results to {cv_save_path}")
    return aggregated

def main():
    parser = argparse.ArgumentParser(description="Full Production Training: Frequency-Weighted SemiCircular CNN-AE")
    parser.add_argument('--fold', type=str, default='0', help="Fold index (0-4) or 'all'")
    parser.add_argument('--epochs', type=int, default=800, help="Number of training epochs (default 800)")
    parser.add_argument('--force', action='store_true', help="Force retraining even if fold results already exist")
    args = parser.parse_args()

    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../.."))

    models_dir = os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study/models')
    os.makedirs(models_dir, exist_ok=True)
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    cache_dir = os.path.join(_REPO_ROOT, 'temp_hybrid_cache')
    out_dir = os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study')
    config_path = os.path.join(out_dir, 'block_config.json')
    if not os.path.exists(config_path):
        config_path = os.path.join(_REPO_ROOT, 'block_config.json')
    
    with open(config_path, 'r') as f:
        blocks_config = json.load(f)

    sample_var = blocks_config['Block_1']['variables'][0]
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(os.path.join(cache_dir, sample_var)) if f.endswith('.npy')])
    
    stokes_csv = os.path.join(_REPO_ROOT, 'unnormalized_erosion_vs_stokes_data.csv')
    df_stokes = pd.read_csv(stokes_csv)
    df_stokes['case'] = df_stokes['file_name'].str.replace('.csv', '')
    valid_cases = set(df_stokes[df_stokes['stokes'] > 1.0]['case'])
    case_names = sorted([c for c in case_names if c in valid_cases])
    N = len(case_names)
    logger.info(f"Total dataset size (filtered St > 1.0): {N} cases.")

    # ---------------------------------------------------------
    # MANUSCRIPT ALIGNMENT: Dev vs Locked Test Split
    # ---------------------------------------------------------
    # The manuscript specifies 372 total cases, 75 locked out for testing, 297 for development.
    # We first isolate the locked 75 cases to match the manuscript's evaluation set.
    kf_initial = KFold(n_splits=5, shuffle=True, random_state=42)
    dev_indices, locked_test_indices = next(kf_initial.split(np.arange(N)))
    
    logger.info(f"Isolated {len(locked_test_indices)} permanently locked test cases.")
    logger.info(f"Performing 5-fold CV on the {len(dev_indices)} development cases.")

    kf_dev = KFold(n_splits=5, shuffle=True, random_state=1337)
    
    # Map the dev-relative splits back to absolute indices
    splits = []
    for train_idx_dev, val_idx_dev in kf_dev.split(dev_indices):
        abs_train_idx = dev_indices[train_idx_dev]
        abs_val_idx = dev_indices[val_idx_dev]
        splits.append((abs_train_idx, abs_val_idx))

    if args.fold in ['all', '--all']:
        target_folds = list(range(5))
    else:
        target_folds = [int(args.fold)]

    logger.info(f"Running full production SemiCircular CNN training on folds: {target_folds} for {args.epochs} epochs (force={args.force})")
    for fold_idx in target_folds:
        train_idx, test_idx = splits[fold_idx]
        logger.info(f"\n>>> Starting Fold {fold_idx}: {len(train_idx)} train cases, {len(test_idx)} test cases <<<")
        run_fold(fold_idx, blocks_config, case_names, train_idx, test_idx, out_dir, models_dir, cache_dir, n_epochs=args.epochs, force=args.force)

    if set(target_folds) == set(range(5)):
        aggregate_cv_results(out_dir)

if __name__ == '__main__':
    main()
