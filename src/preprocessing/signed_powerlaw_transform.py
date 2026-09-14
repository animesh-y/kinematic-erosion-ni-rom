#!/usr/bin/env python3
import os, sys, time, json, logging
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEGENERATE_VARS = {'dp_x_norm', 'dp_y_norm', 'dp_z_norm', 'dp_mag_norm', 'dke_norm', 'frequency'}
COORD_COLS = {'s', 'r', 'theta', 's_normalized'}

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc_encode = nn.Linear(128 * 32 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 32 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 2, 2), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 2, 2))

    def encode(self, x):
        x_r = F.interpolate(x, size=(512, 256), mode='bilinear', align_corners=True)
        h = self.encoder(x_r).view(x.size(0), -1)
        return self.fc_encode(h)

    def decode(self, z):
        h = self.fc_decode(z).view(z.size(0), 128, 32, 16)
        return F.interpolate(self.decoder(h), size=(582, 221), mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.decode(self.encode(x))

def eval_field_errors(S_actual, S_pred, test_idx):
    sum_sq, tot_el = 0.0, 0
    g_max, g_min = -np.inf, np.inf
    cases = []
    for i in range(len(test_idx)):
        a = S_actual[test_idx[i]]
        p = S_pred[i]
        e = a - p
        esq = e ** 2
        rmse = float(np.sqrt(np.mean(esq)))
        rng = float(np.max(a) - np.min(a))
        nrmse = float(rmse / (rng + 1e-12) * 100)
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        sum_sq += np.sum(esq)
        tot_el += len(a)
        g_max = max(g_max, np.max(a))
        g_min = min(g_min, np.min(a))
        cases.append({'nrmse_pct': nrmse, 'r2': r2})
    g_rmse = float(np.sqrt(sum_sq / tot_el))
    g_nrmse = float(g_rmse / (g_max - g_min + 1e-12) * 100)
    return {
        'global_nrmse_pct': g_nrmse,
        'r2': float(np.mean([c['r2'] for c in cases])),
    }

def main():
    input_dir = 'trimmed_data_csv'
    output_dir = 'exp_transform_study'
    tmp_dir = 'temp_hybrid_cache'
    grid_s, grid_theta = 582, 221
    n_spatial = grid_s * grid_theta
    eps = 1e-8
    os.makedirs(output_dir, exist_ok=True)

    # Fast iteration: just the 5 key variables
    active_vars = ['oka_erosion_norm', 'velocity_variance_norm', 'v_mean_norm', 'alpha_variance', 'frequency_norm']

    mask_var_dir = os.path.join(tmp_dir, 'frequency_norm')
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(mask_var_dir) if f.endswith('.npy')])
    N = len(case_names)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    all_results = {}
    
    for vi, var in enumerate(active_vars):
        logger.info(f"\n--- [{vi+1}/{len(active_vars)}] {var} ---")
        vdir = os.path.join(tmp_dir, var)
        S = np.zeros((N, n_spatial), dtype=np.float32)
        for i, cn in enumerate(case_names):
            S[i] = np.load(os.path.join(vdir, f'{cn}.npy'))

        S_flat = S.flatten()
        S_active = S_flat[np.abs(S_flat) > 1e-6]
        skewness = scipy.stats.skew(S_active) if len(S_active) > 0 else 0
        peak_to_mean = np.max(np.abs(S)) / (np.mean(np.abs(S)) + eps)
        
        M = np.max(np.abs(S), axis=1) + eps
        
        if skewness > 5.0 or peak_to_mean > 50:
            norm_method = "exp-transform"
            # Normalize to [0,1] first to prevent np.exp() overflow
            S_hat = S / M[:, None]
            # Exponential transform (exp(x) - 1 maps [0,1] to [0, 1.718])
            S_train_transform = np.exp(S_hat) - 1.0
        else:
            norm_method = "max-normalization"
            S_train_transform = S / M[:, None]

        var_res = {'norm_method': norm_method, 'pod': {}, 'cnn': {}}

        # Train POD
        mean_transform = np.mean(S_train_transform, axis=0)
        S_c = S_train_transform - mean_transform
        U_s, sig, Vt = np.linalg.svd(S_c, full_matrices=False)
        cum_e = np.cumsum(sig ** 2) / (np.sum(sig ** 2) + 1e-30)
        
        k_95 = max(int(np.argmax(cum_e >= 0.95) + 1), 1)
        
        for p_name, k in [('95', k_95)]:
            Vt_k = Vt[:k]
            Z = S_c[test_idx] @ Vt_k.T
            S_trans_pred = Z @ Vt_k + mean_transform
            
            if norm_method == "exp-transform":
                # Inverse transform: ln(S_trans_pred + 1)
                S_pred_hat = np.log(np.clip(S_trans_pred + 1.0, eps, None))
                S_pred_phys = S_pred_hat * M[test_idx, None]
            else:
                S_pred_phys = S_trans_pred * M[test_idx, None]
                
            var_res['pod'][p_name] = {'modes': k, 'metrics': eval_field_errors(S, S_pred_phys, test_idx)}

        # Train Standard Unweighted CNN
        s_min_val = float(S_train_transform.min())
        s_max_val = float(S_train_transform.max())
        S_scaled = (S_train_transform - s_min_val) / (s_max_val - s_min_val + eps)
        S_scaled_2d = S_scaled.reshape(N, 1, grid_s, grid_theta)
        
        for ld in [10, k_95]:
            logger.info(f"Training Exp-Transformed CNN-AE {ld} dims...")
            cnn = ConvAutoencoder(latent_dim=ld).cuda()
            opt = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-5)
            
            # Using standard unweighted MSE loss to see how exp transform affects it
            train_loader = DataLoader(TensorDataset(
                torch.from_numpy(S_scaled_2d[train_idx]).float()
            ), batch_size=32, shuffle=True)
            
            cnn.train()
            for ep in range(50):
                for xb, in train_loader:
                    xb = xb.cuda()
                    opt.zero_grad()
                    pred = cnn(xb)
                    loss = F.mse_loss(pred, xb)
                    loss.backward()
                    opt.step()
            cnn.eval()
            
            with torch.no_grad():
                pred_scaled = []
                t_ds = TensorDataset(torch.from_numpy(S_scaled_2d[test_idx]).float())
                t_loader = DataLoader(t_ds, batch_size=16, shuffle=False)
                for batch in t_loader:
                    pred_scaled.append(cnn(batch[0].cuda()).cpu().numpy())
            pred_scaled = np.concatenate(pred_scaled, axis=0).reshape(len(test_idx), -1)
            
            S_trans_pred_cnn = pred_scaled * (s_max_val - s_min_val) + s_min_val
            
            if norm_method == "exp-transform":
                S_pred_hat_cnn = np.log(np.clip(S_trans_pred_cnn + 1.0, eps, None))
                S_pred_phys_cnn = S_pred_hat_cnn * M[test_idx, None]
            else:
                S_pred_phys_cnn = S_trans_pred_cnn * M[test_idx, None]
                
            var_res['cnn'][f'dim_{ld}'] = {'latent': ld, 'metrics': eval_field_errors(S, S_pred_phys_cnn, test_idx)}

        all_results[var] = var_res
        
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

    logger.info("Completed Exp Transform Study!")

if __name__ == '__main__':
    main()
