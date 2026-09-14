#!/usr/bin/env python3
import os, sys, time, json, logging, gc
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.model_selection import train_test_split

import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

DEGENERATE_VARS = {'dp_x_norm', 'dp_y_norm', 'dp_z_norm', 'dp_mag_norm', 'dke_norm', 'frequency'}
COORD_COLS = {'s', 'r', 'theta', 's_normalized'}

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        act = nn.GELU
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), act(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1), act(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1), act(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), act(), nn.MaxPool2d(2, 2))
        self.fc_encode = nn.Linear(128 * 32 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128 * 32 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2), act(),
            nn.ConvTranspose2d(64, 32, 2, 2), act(),
            nn.ConvTranspose2d(32, 16, 2, 2), act(),
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
    cases = []
    g_max, g_min = -np.inf, np.inf
    sum_sq, tot_el = 0.0, 0
    for i in range(len(test_idx)):
        a = S_actual[test_idx[i]]
        p = S_pred[i]
        esq = (a - p) ** 2
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        cases.append(r2)
        sum_sq += np.sum(esq)
        tot_el += len(a)
        g_max = max(g_max, np.max(a))
        g_min = min(g_min, np.min(a))
        
    g_rmse = float(np.sqrt(sum_sq / tot_el))
    g_nrmse = float(g_rmse / (g_max - g_min + 1e-12) * 100)
    return {'r2': float(np.mean(cases)), 'global_nrmse_pct': g_nrmse}

def main():
    input_dir = 'trimmed_data_csv'
    output_dir = 'final_hybrid_dr_results'
    latent_dir = 'final_latents'
    tmp_dir = 'temp_hybrid_cache'
    grid_s, grid_theta = 582, 221
    n_spatial = grid_s * grid_theta
    eps = 1e-8
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(latent_dir, 'pod'), exist_ok=True)
    os.makedirs(os.path.join(latent_dir, 'cnn_unw'), exist_ok=True)
    os.makedirs(os.path.join(latent_dir, 'cnn_w'), exist_ok=True)

    sample = pd.read_csv(os.path.join(input_dir, sorted(os.listdir(input_dir))[0]))
    active_vars = [c for c in sample.columns if c not in COORD_COLS and c not in DEGENERATE_VARS and 'variance' not in c]

    mask_var_dir = os.path.join(tmp_dir, 'frequency_norm')
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(mask_var_dir) if f.endswith('.npy')])
    N = len(case_names)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    logger.info("Preparing Log-Transformed frequency_norm weight matrix...")
    W_mask_raw = np.zeros((N, n_spatial), dtype=np.float32)
    for i, cn in enumerate(case_names):
        W_mask_raw[i] = np.load(os.path.join(mask_var_dir, f'{cn}.npy'))
    W_log = np.log(np.clip(W_mask_raw, 0.0, None) + eps)
    W_scaled = (W_log - float(W_log.min())) / (float(W_log.max()) - float(W_log.min()) + eps)
    W_tensor_train = torch.from_numpy(W_scaled.reshape(N, 1, grid_s, grid_theta)[train_idx]).float().cuda()

    all_results = {}
    
    for vi, var in enumerate(active_vars):
        logger.info(f"\n======================================")
        logger.info(f"--- [{vi+1}/{len(active_vars)}] {var} ---")
        
        S = np.zeros((N, n_spatial), dtype=np.float32)
        for i, cn in enumerate(case_names):
            S[i] = np.load(os.path.join(tmp_dir, var, f'{cn}.npy'))

        # Determine Transformation Logic
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
            if skewness > 5.0 or peak_to_mean > 50:
                norm_method = "log-transform"
                S_train_transform = np.log(np.clip(S, 0.0, None) + eps)
            else:
                norm_method = "max-normalization"
                S_train_transform = S
                
        M = np.max(np.abs(S_train_transform), axis=1) + eps
        S_train_norm = S_train_transform / M[:, None]
        
        var_res = {'norm_method': norm_method, 'loss_func': loss_func, 'pod': {}, 'cnn_unw': {}, 'cnn_w': {}}

        # --- 1. POD Evaluation & Latent Extraction ---
        mean_transform = np.mean(S_train_norm, axis=0)
        S_c = S_train_norm - mean_transform
        U_s, sig, Vt = np.linalg.svd(S_c, full_matrices=False)
        cum_e = np.cumsum(sig ** 2) / (np.sum(sig ** 2) + 1e-30)
        k_95 = max(int(np.argmax(cum_e >= 0.95) + 1), 1)
        
        Vt_k = Vt[:k_95]
        Z_pod = S_c @ Vt_k.T
        np.save(os.path.join(latent_dir, 'pod', f'{var}_Z.npy'), Z_pod)
        
        S_trans_pred_pod = Z_pod[test_idx] @ Vt_k + mean_transform
        
        def inverse_transform(pred_norm, M_true, method):
            trans = pred_norm * M_true[:, None]
            if method == 'cube-root': return trans ** 3
            if method == 'square-root': return trans ** 2 * np.sign(trans)
            if method == 'log-transform': return np.exp(trans) - eps
            return trans

        S_pred_phys_pod = inverse_transform(S_trans_pred_pod, M[test_idx], norm_method)
        m_pod = eval_field_errors(S, S_pred_phys_pod, test_idx)
        var_res['pod']['95'] = {'modes': k_95, 'metrics': m_pod}
        logger.info(f"[{norm_method.upper()}] POD 95% ({k_95} modes) | R2: {m_pod['r2']:.4f}")

        # --- 2. CNN Setup ---
        s_min_val, s_max_val = float(S_train_norm.min()), float(S_train_norm.max())
        S_scaled = (S_train_norm - s_min_val) / (s_max_val - s_min_val + eps)
        S_scaled_2d = S_scaled.reshape(N, 1, grid_s, grid_theta)
        
        S_train_tensor = torch.from_numpy(S_scaled_2d[train_idx]).float()
        train_loader_unw = DataLoader(TensorDataset(S_train_tensor), batch_size=32, shuffle=True)
        train_loader_w = DataLoader(TensorDataset(S_train_tensor, W_tensor_train.cpu()), batch_size=32, shuffle=True)
        t_ds = TensorDataset(torch.from_numpy(S_scaled_2d).float())
        t_loader = DataLoader(t_ds, batch_size=16, shuffle=False)
        
        ld = 10 if var == 'oka_erosion_norm' else k_95 # Standardize dimension comparison for simplicity
        
        def train_cnn(loader, weighted=False):
            cnn = ConvAutoencoder(latent_dim=ld).cuda()
            opt = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-5)
            cnn.train()
            for ep in range(50):
                for batch in loader:
                    xb = batch[0].cuda()
                    opt.zero_grad()
                    pred = cnn(xb)
                    
                    if weighted:
                        wb = batch[1].cuda()
                        if loss_func == 'l1': loss = torch.mean(wb * torch.abs(pred - xb))
                        else: loss = torch.mean(wb * (pred - xb)**2)
                    else:
                        if loss_func == 'l1': loss = F.l1_loss(pred, xb)
                        else: loss = F.mse_loss(pred, xb)
                        
                    loss.backward()
                    opt.step()
                    
            cnn.eval()
            Z_all, pred_all = [], []
            with torch.no_grad():
                for batch in t_loader:
                    x = batch[0].cuda()
                    z = cnn.encode(x)
                    p = cnn.decode(z)
                    Z_all.append(z.cpu().numpy())
                    pred_all.append(p.cpu().numpy())
            
            Z_np = np.concatenate(Z_all, axis=0)
            pred_np = np.concatenate(pred_all, axis=0).reshape(N, -1)
            
            pred_trans = pred_np * (s_max_val - s_min_val) + s_min_val
            pred_phys_test = inverse_transform(pred_trans[test_idx], M[test_idx], norm_method)
            m_cnn = eval_field_errors(S, pred_phys_test, test_idx)
            
            return cnn, Z_np, m_cnn

        # --- 3. Unweighted CNN ---
        cnn_unw, Z_unw, m_unw = train_cnn(train_loader_unw, weighted=False)
        np.save(os.path.join(latent_dir, 'cnn_unw', f'{var}_Z.npy'), Z_unw)
        var_res['cnn_unw'][f'dim_{ld}'] = {'metrics': m_unw}
        logger.info(f"[{loss_func.upper()}] Unw-CNN ({ld} dims) | R2: {m_unw['r2']:.4f}")
        del cnn_unw
        torch.cuda.empty_cache(); gc.collect()

        # --- 4. Freq-Weighted CNN ---
        cnn_w, Z_w, m_w = train_cnn(train_loader_w, weighted=True)
        np.save(os.path.join(latent_dir, 'cnn_w', f'{var}_Z.npy'), Z_w)
        var_res['cnn_w'][f'dim_{ld}'] = {'metrics': m_w}
        logger.info(f"[{loss_func.upper()}] Wgt-CNN ({ld} dims) | R2: {m_w['r2']:.4f}")
        del cnn_w
        torch.cuda.empty_cache(); gc.collect()

        all_results[var] = var_res
        with open(os.path.join(output_dir, 'results.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

    logger.info("Completed Final Hybrid DR Extraction!")

if __name__ == '__main__':
    main()
