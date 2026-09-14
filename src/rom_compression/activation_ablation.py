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

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=10, act_name='relu'):
        super().__init__()
        
        if act_name == 'relu':
            act = nn.ReLU
        elif act_name == 'leaky_relu':
            act = nn.LeakyReLU
        elif act_name == 'gelu':
            act = nn.GELU
        elif act_name == 'elu':
            act = nn.ELU
        else:
            act = nn.ReLU
            
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
    sum_sq = 0.0
    cases = []
    for i in range(len(test_idx)):
        a = S_actual[test_idx[i]]
        p = S_pred[i]
        e = a - p
        esq = e ** 2
        ss_tot = np.sum((a - np.mean(a)) ** 2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        cases.append(r2)
    return float(np.mean(cases))

def main():
    tmp_dir = 'temp_hybrid_cache'
    grid_s, grid_theta = 582, 221
    n_spatial = grid_s * grid_theta
    eps = 1e-8

    # We will test on just one variable to see the effect: oka_erosion_norm
    var = 'oka_erosion_norm'
    logger.info(f"--- Running Activation Study on {var} ---")
    
    mask_var_dir = os.path.join(tmp_dir, 'frequency_norm')
    case_names = sorted([f.replace('.npy', '') for f in os.listdir(mask_var_dir) if f.endswith('.npy')])
    N = len(case_names)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    vdir = os.path.join(tmp_dir, var)
    S = np.zeros((N, n_spatial), dtype=np.float32)
    for i, cn in enumerate(case_names):
        S[i] = np.load(os.path.join(vdir, f'{cn}.npy'))

    M = np.max(np.abs(S), axis=1) + eps
    S_log = np.log(np.clip(S, 0.0, None) + eps)
    S_train_transform = S_log
    
    s_min_val = float(S_train_transform.min())
    s_max_val = float(S_train_transform.max())
    S_scaled = (S_train_transform - s_min_val) / (s_max_val - s_min_val + eps)
    S_scaled_2d = S_scaled.reshape(N, 1, grid_s, grid_theta)

    acts = ['relu', 'leaky_relu', 'gelu', 'elu']
    results = {}

    for act in acts:
        logger.info(f"Training CNN-AE (10 dims) with {act} activation...")
        cnn = ConvAutoencoder(latent_dim=10, act_name=act).cuda()
        opt = optim.Adam(cnn.parameters(), lr=1e-3, weight_decay=1e-5)
        
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
        
        S_trans_pred = pred_scaled * (s_max_val - s_min_val) + s_min_val
        S_pred_phys = np.exp(S_trans_pred) - eps
        
        r2 = eval_field_errors(S, S_pred_phys, test_idx)
        logger.info(f"Result for {act}: R2 = {r2:.4f}")
        results[act] = r2

    print("\n=== ACTIVATION STUDY RESULTS ===")
    for act, r2 in results.items():
        print(f"{act:>12}: {r2:.4f}")

if __name__ == '__main__':
    main()
