#!/usr/bin/env python3
"""
Multi-Variable CNN Autoencoder + GPR ROM Pipeline (Parallelized)
================================================================
Runs ConvAutoencoder (10 latent variables) + GPU-accelerated GPR for ALL active
erosion impact statistics variables in parallel.

Outputs: multivar_cnn_results/results_all_variables.json
"""
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import time
import json
import logging
import shutil
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import gpytorch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Physical parameter space
PARTICLE_SIZES_micro = [50, 100, 150, 200, 300, 500]
DENSITY = [2000, 5000, 10000, 15000, 30000]
VELOCITY = [5, 10, 15, 18, 20]
BEND_RATIO = [1.5, 2.0, 5.0]

DEGENERATE_VARS = {'dp_x_norm', 'dp_y_norm', 'dp_z_norm', 'dp_mag_norm', 'dke_norm', 'frequency'}
COORD_COLS = {'s', 'r', 'theta', 's_normalized'}

# ─── SemiCircularConv2d ────────────────────────────────────────────
class SemiCircularConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = F.pad(x, (0, 0, 1, 1), mode='constant', value=0)
        return self.conv(x)


# ─── ConvAutoencoder ───────────────────────────────────────────────
class ConvAutoencoder(nn.Module):
    """
    Semi-Circular Convolutional Autoencoder with Dense Spatial Flattening.
    Preserves 2D spatial coordinate topology (s, theta) without global pooling,
    ensuring precise reconstruction of localized extrados wear craters.
    """
    def __init__(self, latent_dim=10):
        super(ConvAutoencoder, self).__init__()
        act = nn.GELU
        self.encoder = nn.Sequential(
            SemiCircularConv2d(1, 16, kernel_size=3, stride=1),
            act(), nn.MaxPool2d(2, 2),
            
            SemiCircularConv2d(16, 32, kernel_size=3, stride=1),
            act(), nn.MaxPool2d(2, 2),
            
            SemiCircularConv2d(32, 64, kernel_size=3, stride=1),
            act(), nn.MaxPool2d(2, 2),
            
            SemiCircularConv2d(64, 128, kernel_size=3, stride=1),
            act(), nn.MaxPool2d(2, 2)
        )
        self.flatten_size = 128 * 32 * 16
        self.fc_encode = nn.Linear(self.flatten_size, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            act(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            act(),
            
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            act(),
            
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )

    def encode_rescaled(self, x_rescaled):
        h = self.encoder(x_rescaled).view(x_rescaled.size(0), -1)
        z = self.fc_encode(h)
        return z

    def decode_rescaled(self, z):
        h = self.fc_decode(z).view(z.size(0), 128, 32, 16)
        recon = self.decoder(h)
        return recon

    def encode(self, x):
        x_rescaled = F.interpolate(x, size=(512, 256), mode='bilinear', align_corners=True)
        return self.encode_rescaled(x_rescaled)

    def decode(self, z):
        recon_rescaled = self.decode_rescaled(z)
        recon = F.interpolate(recon_rescaled, size=(582, 221), mode='bilinear', align_corners=True)
        return recon

    def forward(self, x):
        x_rescaled = F.interpolate(x, size=(512, 256), mode='bilinear', align_corners=True)
        z = self.encode_rescaled(x_rescaled)
        recon_rescaled = self.decode_rescaled(z)
        recon = F.interpolate(recon_rescaled, size=(582, 221), mode='bilinear', align_corners=True)
        return recon

# ─── GPyTorch Model ────────────────────────────────────────────────
class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_models):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_models]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=4,
                                          batch_shape=torch.Size([num_models])),
            batch_shape=torch.Size([num_models]))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

def train_batch_gpr(train_x, train_y, dev, lr=0.1, iters=150):
    n = train_y.shape[0]
    lik = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=torch.Size([n]),
        noise_constraint=gpytorch.constraints.Interval(1e-4, 0.5)).to(dev)
    mdl = BatchGPModel(train_x.to(dev), train_y.to(dev), lik, n).to(dev)
    train_x, train_y = train_x.to(dev), train_y.to(dev)
    mdl.train(); lik.train()
    opt = torch.optim.Adam(mdl.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, mdl)
    for _ in range(iters):
        opt.zero_grad()
        for jit in [1e-3, 1e-2, 1e-1]:
            try:
                with gpytorch.settings.cholesky_jitter(jit):
                    loss = -mll(mdl(train_x), train_y).sum()
                    loss.backward(); opt.step(); break
            except Exception:
                opt.zero_grad()
    return mdl, lik

def predict_batch_gpr(mdl, lik, test_x, dev):
    mdl.eval(); lik.eval()
    n = mdl.mean_module.batch_shape[0]
    tx = torch.from_numpy(test_x).float().unsqueeze(0).repeat(n,1,1).to(dev)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
        return lik(mdl(tx)).mean.cpu().numpy()

# ─── Helpers ───────────────────────────────────────────────────────
def decode_filename(fn):
    b = os.path.basename(fn).replace('.csv', '')
    if len(b) != 4 or not b.isdigit():
        return None
    i, j, k, l = [int(c)-1 for c in b]
    return [BEND_RATIO[i], VELOCITY[j], DENSITY[k], PARTICLE_SIZES_micro[l]]

def build_interp(df):
    pts = np.column_stack((df['s_normalized'].values, df['theta'].values))
    return (LinearNDInterpolator(pts, np.zeros(len(pts))),
            NearestNDInterpolator(pts, np.zeros(len(pts))))

def interp_file(fn, input_dir, variables, sg, tg, tmp):
    params = decode_filename(fn)
    if params is None:
        return None
    df = pd.read_csv(os.path.join(input_dir, fn))
    bp = df[df['theta'] < -np.pi/2].copy(); bp['theta'] += 2*np.pi
    bn = df[df['theta'] > np.pi/2].copy(); bn['theta'] -= 2*np.pi
    dfp = pd.concat([df, bp, bn], ignore_index=True)
    li, ni = build_interp(dfp)
    for v in variables:
        vals = dfp[v].values if v in dfp.columns else np.zeros(len(dfp))
        li.values = vals[:, np.newaxis]
        ni.values = vals[:, np.newaxis]
        g = li(sg, tg).squeeze()
        if np.isnan(g).any():
            gn = ni(sg, tg).squeeze()
            g[np.isnan(g)] = gn[np.isnan(g)]
        vd = os.path.join(tmp, v)
        os.makedirs(vd, exist_ok=True)
        np.save(os.path.join(vd, f'{fn.replace(".csv","")}.npy'), g.astype(np.float32).flatten())
    return params, fn

def eval_field_errors(S_actual, S_pred, test_idx):
    """Compute error metrics between actual and predicted fields."""
    sum_sq, tot_el = 0.0, 0
    g_max, g_min = -np.inf, np.inf
    cases = []
    for i in range(len(test_idx)):
        a = S_actual[test_idx[i]].flatten()
        p = S_pred[i].flatten()
        e = a - p
        esq = e**2
        mae = float(np.mean(np.abs(e)))
        maxae = float(np.max(np.abs(e)))
        rmse = float(np.sqrt(np.mean(esq)))
        rng = float(np.max(a) - np.min(a))
        nrmse = float(rmse / (rng + 1e-12) * 100)
        anorm = np.linalg.norm(a)
        l2e = float(np.linalg.norm(e) / (anorm + 1e-12) * 100)
        ss_tot = np.sum((a - np.mean(a))**2) + 1e-12
        r2 = float(1.0 - np.sum(esq) / ss_tot)
        sum_sq += np.sum(esq)
        tot_el += len(a)
        g_max = max(g_max, np.max(a))
        g_min = min(g_min, np.min(a))
        cases.append({'mae': mae, 'max_ae': maxae, 'nrmse_pct': nrmse,
                      'l2_error_pct': l2e, 'r2': r2})
    g_rmse = float(np.sqrt(sum_sq / tot_el))
    g_nrmse = float(g_rmse / (g_max - g_min + 1e-12) * 100)
    return {
        'global_nrmse_pct': g_nrmse,
        'mae': float(np.mean([c['mae'] for c in cases])),
        'max_ae': float(np.mean([c['max_ae'] for c in cases])),
        'r2': float(np.mean([c['r2'] for c in cases])),
        'l2_error_pct': float(np.mean([c['l2_error_pct'] for c in cases]))
    }

# ─── Main ──────────────────────────────────────────────────────────
def main():
    input_dir = 'trimmed_data_csv'
    output_dir = 'multivar_cnn_results'
    tmp_dir = 'temp_multivar_cnn_cache'
    grid_s, grid_theta = 582, 221
    latent_dim = 10

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    if not torch.cuda.is_available():
        logger.error("CUDA not available!"); sys.exit(1)

    device = torch.device('cuda')

    # Discover active variables from a sample file
    sample = pd.read_csv(os.path.join(input_dir,
                         sorted(os.listdir(input_dir))[0]))
    all_vars = [c for c in sample.columns if c not in COORD_COLS]
    active_vars = [v for v in all_vars if v not in DEGENERATE_VARS]
    logger.info(f"Active variables: {len(active_vars)}")

    # File list
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.csv')],
                       key=lambda x: int(x.replace('.csv','')) if x.replace('.csv','').isdigit() else 9999)
    N = len(all_files)

    # Grid
    s1 = np.linspace(0.0, 2.0, grid_s)
    t1 = np.linspace(-np.pi, np.pi, grid_theta)
    sg, tg = np.meshgrid(s1, t1, indexing='ij')

    # ── Stage 1: Interpolate all variables for all files ──
    if os.path.exists('temp_hybrid_cache') and all(os.path.exists(os.path.join('temp_hybrid_cache', v)) for v in active_vars):
        logger.info("Using existing cache from temp_hybrid_cache (bypassing interpolation)")
        tmp_dir = 'temp_hybrid_cache'
        valid = [(decode_filename(f), f) for f in all_files if decode_filename(f) is not None]
        X_params = np.array([r[0] for r in valid])
        case_names = [r[1].replace('.csv', '') for r in valid]
    else:
        logger.info("=== Stage 1: Interpolating all variables ===")
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(interp_file)(f, input_dir, active_vars, sg, tg, tmp_dir)
            for f in tqdm(all_files, desc="Interpolating"))

        valid = [r for r in results if r is not None]
        X_params = np.array([r[0] for r in valid])
        case_names = [r[1].replace('.csv', '') for r in valid]

    # Dimensionless parameters
    rho_f, mu_f, D = 998.2, 0.001003, 0.0254
    Re = rho_f * X_params[:,1] * D / mu_f
    dr = X_params[:,2] / rho_f
    sr = X_params[:,3] * 1e-6 / D
    br = X_params[:,0]
    X_dim = np.column_stack((Re, dr, sr, br))

    # Train/test split (shared across all variables)
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=0.20, random_state=42)

    X_tr, X_te = X_dim[train_idx], X_dim[test_idx]
    scaler_x = MinMaxScaler()
    X_tr_s = scaler_x.fit_transform(X_tr)
    X_te_s = scaler_x.transform(X_te)

    # ── Stage 2: Per-variable CNN + GPR in parallel ──
    all_results = {}

    def _execute_single_variable(var, dev):
        # Load snapshot matrix, shape (N, 1, 582, 221)
        vdir = os.path.join(tmp_dir, var)
        S = np.zeros((N, 1, grid_s, grid_theta), dtype=np.float32)
        for i, cn in enumerate(case_names):
            S[i, 0, :, :] = np.load(os.path.join(vdir, f'{cn}.npy')).reshape(grid_s, grid_theta)

        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        # Scale variable globally to [0,1]
        S_min = float(S.min())
        S_max = float(S.max())
        S_scaled = (S - S_min) / (S_max - S_min + 1e-12)

        # Train ConvAutoencoder
        S_tr_np = S_scaled[train_idx]
        train_ds = TensorDataset(torch.from_numpy(S_tr_np).float())
        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

        ae = ConvAutoencoder(latent_dim=latent_dim).to(dev)
        opt = optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
        crit = nn.MSELoss()
        use_amp = (dev.type == 'cuda')
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        ae.train()
        for epoch in range(50):
            for batch in train_loader:
                xb = batch[0].to(dev)
                opt.zero_grad()
                with torch.amp.autocast(device_type=dev.type, enabled=use_amp):
                    loss = crit(ae(xb), xb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

        ae.eval()
        del opt, train_loader, train_ds
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract latent variables
        Z_list = []
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp):
            for i in range(0, N, 4):
                batch_s = torch.from_numpy(S_scaled[i:i+4]).float().to(dev)
                Z_list.append(ae.encode(batch_s).cpu().numpy())
        Z = np.concatenate(Z_list, axis=0)  # (N, latent_dim)

        # Train GPR on latent variables
        Y_tr = Z[train_idx].T  # (latent_dim, n_train)
        X_tr_t = torch.from_numpy(X_tr_s).float().unsqueeze(0).repeat(latent_dim, 1, 1)
        Y_tr_t = torch.from_numpy(Y_tr).float()

        model, lik = train_batch_gpr(X_tr_t, Y_tr_t, dev, lr=0.1, iters=150)
        pred_Z = predict_batch_gpr(model, lik, X_te_s, dev)  # (latent_dim, n_test)

        del model, lik
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        # Decode reconstructed test profiles in batches
        pred_Z_t = torch.from_numpy(pred_Z.T).float().to(dev)
        recon_scaled_list = []
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp):
            for i in range(0, len(test_idx), 4):
                recon_scaled_list.append(ae.decode(pred_Z_t[i:i+4]).cpu().numpy())
        recon_scaled = np.concatenate(recon_scaled_list, axis=0)  # (n_test, 1, 582, 221)

        # Unscale
        S_pred = recon_scaled * (S_max - S_min) + S_min

        # Compute pure autoencoder reconstruction (projection limit)
        test_scaled_t = torch.from_numpy(S_scaled[test_idx]).float().to(dev)
        proj_scaled_list = []
        with torch.no_grad(), torch.amp.autocast(device_type=dev.type, enabled=use_amp):
            for i in range(0, len(test_idx), 4):
                proj_scaled_list.append(ae(test_scaled_t[i:i+4]).cpu().numpy())
        proj_scaled = np.concatenate(proj_scaled_list, axis=0)
        S_proj = proj_scaled * (S_max - S_min) + S_min

        # Evaluate
        gpr_metrics = eval_field_errors(S, S_pred, test_idx)
        proj_metrics = eval_field_errors(S, S_proj, test_idx)

        del ae
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'latent_dim': latent_dim,
            'gpr_rom': gpr_metrics,
            'projection_only': proj_metrics
        }

    def process_variable(var_idx, var):
        t0 = time.time()
        logger.info(f"[{var_idx+1}/{len(active_vars)}] Starting: {var}")
        try:
            res = _execute_single_variable(var, dev=device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.warning(f"CUDA issue on {var} ({e}), falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            res = _execute_single_variable(var, dev=torch.device('cpu'))

        gpr_metrics = res['gpr_rom']
        proj_metrics = res['projection_only']
        logger.info(f"[{var_idx+1}/{len(active_vars)}] Completed: {var} | GPR NRMSE: {gpr_metrics['global_nrmse_pct']:.4f}% | R²: {gpr_metrics['r2']:.4f} | Proj NRMSE: {proj_metrics['global_nrmse_pct']:.4f}% | ({time.time()-t0:.1f}s)")
        return var, res

    # Run in parallel with 2 workers for safe VRAM usage
    logger.info("Starting multi-variable CNN pipeline in parallel with 2 workers...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_variable, i, var) for i, var in enumerate(active_vars)]
        for f in futures:
            try:
                v_name, v_res = f.result()
                all_results[v_name] = v_res
            except Exception as e:
                logger.error(f"Error processing a variable thread: {e}")

    # Save
    out_path = os.path.join(output_dir, 'results_all_variables.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved results to {out_path}")

    # Cleanup temp
    if tmp_dir != 'temp_hybrid_cache' and os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        logger.info("Done! Cleaned up temp cache.")
    else:
        logger.info("Done! Preserved temp_hybrid_cache.")

if __name__ == '__main__':
    main()
