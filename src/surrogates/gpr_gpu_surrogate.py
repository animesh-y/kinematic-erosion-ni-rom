#!/usr/bin/env python3
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from joblib import Parallel, delayed
from tqdm import tqdm

import torch
import gpytorch

# Define GPyTorch batch GP model
class BatchGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_models):
        super(BatchGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_models]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=4, batch_shape=torch.Size([num_models])),
            batch_shape=torch.Size([num_models])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def build_interpolators(df_padded):
    points = np.column_stack((df_padded['s_normalized'].values, df_padded['theta'].values))
    dummy_vals = np.zeros(len(points))
    lin_interp = LinearNDInterpolator(points, dummy_vals)
    near_interp = NearestNDInterpolator(points, dummy_vals)
    return lin_interp, near_interp

def process_single_test_file(file_name, variables, s_grid, theta_grid, input_dir):
    try:
        df = pd.read_csv(os.path.join(input_dir, file_name))
        buffer_pos = df[df['theta'] < -np.pi/2].copy()
        buffer_pos['theta'] += 2*np.pi
        buffer_neg = df[df['theta'] > np.pi/2].copy()
        buffer_neg['theta'] -= 2*np.pi
        df_padded = pd.concat([df, buffer_pos, buffer_neg], ignore_index=True)
        lin_interp, near_interp = build_interpolators(df_padded)
        
        file_data = {}
        for var in variables:
            if var not in df_padded.columns:
                values = np.zeros(len(df_padded))
            else:
                values = df_padded[var].values
                
            lin_interp.values = values[:, np.newaxis]
            near_interp.values = values[:, np.newaxis]
            interp = lin_interp(s_grid, theta_grid).squeeze()
            if np.isnan(interp).any():
                interp_near = near_interp(s_grid, theta_grid).squeeze()
                interp[np.isnan(interp)] = interp_near[np.isnan(interp)]
            file_data[var] = interp.astype(np.float32).flatten()
        return file_name, file_data
    except Exception as e:
        print(f"Error in pre-interpolating {file_name}: {e}")
        return file_name, None

def train_batch_gpr(train_x, train_y, lr=0.1, training_iter=150, verbose=False):
    num_models = train_y.shape[0]
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([num_models]))
    model = BatchGPModel(train_x, train_y, likelihood, num_models)
    
    # Initialize noise to 1e-4 for numerical stability
    try:
        likelihood.noise_covar.initialize(noise=1e-4 * torch.ones(num_models))
    except Exception:
        pass
        
    model = model.cuda()
    likelihood = likelihood.cuda()
    train_x = train_x.cuda()
    train_y = train_y.cuda()
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    t_start = time.time()
    for i in range(training_iter):
        try:
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            loss.backward()
            optimizer.step()
            if verbose and (i+1) % 50 == 0:
                print(f"  Iteration {i+1}/{training_iter} | Loss: {loss.item():.4f}")
        except RuntimeError as e:
            if "singular" in str(e).lower() or "cholesky" in str(e).lower():
                optimizer.zero_grad()
                with gpytorch.settings.cholesky_jitter(1e-3):
                    output = model(train_x)
                    loss = -mll(output, train_y).sum()
                    loss.backward()
                    optimizer.step()
            else:
                raise e
                
    if verbose:
        print(f"  Batch training of {num_models} models took {time.time() - t_start:.2f} seconds.")
    return model, likelihood

def predict_batch_gpr(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    num_models = model.mean_module.batch_shape[0]
    # Prepare test_x as a batch [num_models, N_test, 4]
    if test_x.ndim == 2:
        test_x_batch = torch.from_numpy(test_x).float().unsqueeze(0).repeat(num_models, 1, 1).cuda()
    else:
        test_x_batch = test_x.cuda()
        
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-4):
        predictions = likelihood(model(test_x_batch))
        pred_mean = predictions.mean
        
    return pred_mean.cpu().numpy()

def main():
    pod_dir = 'pod_individual_999'
    input_dir = 'trimmed_data_csv'
    grid_s = 582
    grid_theta = 221
    var = 'oka_erosion_norm'
    
    print(f"============================================================")
    print(f"Starting GPU-accelerated GPR ROM Pipeline for: {var}")
    print(f"============================================================")
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA GPU is not available! The user requested GPU execution.")
        sys.exit(1)
    print(f"GPU device detected: {torch.cuda.get_device_name(0)}")
    
    # 1. Load snapshots and parameters
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    all_files.sort(key=lambda x: int(x.replace('.csv', '')) if x.replace('.csv', '').isdigit() else 9999)
    num_snapshots = len(all_files)
    print(f"Total snapshots in dataset: {num_snapshots}")
    
    X = np.load(os.path.join(pod_dir, 'X_Parameters.npy'))
    
    # Dimensionless parameter conversion
    rho_f = 998.2        # fluid density (kg/m3)
    mu_f = 0.001003      # dynamic viscosity (Pa.s)
    D_pipe = 0.0254      # pipe diameter (m)
    
    # 1. Reynolds number: Re = (rho_f * V * D) / mu_f
    reynolds = (rho_f * X[:, 1] * D_pipe) / mu_f
    # 2. Density ratio: particle density / fluid density
    density_ratio = X[:, 2] / rho_f
    # 3. Size ratio: particle diameter / pipe diameter (particle size in X is in micrometers)
    size_ratio = (X[:, 3] * 1e-6) / D_pipe
    # 4. Bend ratio
    bend_ratio = X[:, 0]
    
    X_dimensionless = np.column_stack((reynolds, density_ratio, size_ratio, bend_ratio))
    print(f"Input features computed:")
    print(f"  Reynolds Range: {reynolds.min():.1f} - {reynolds.max():.1f}")
    print(f"  Density Ratio Range: {density_ratio.min():.2f} - {density_ratio.max():.2f}")
    print(f"  Size Ratio Range: {size_ratio.min():.5f} - {size_ratio.max():.5f}")
    print(f"  Bend Ratio Range: {bend_ratio.min():.2f} - {bend_ratio.max():.2f}")
    
    # Load SVD data for the target variable
    var_data_dir = os.path.join(pod_dir, 'data', var)
    print(f"Loading SVD data from {var_data_dir}...")
    Phi_k = np.load(os.path.join(var_data_dir, 'Phi_k.npy'))
    Sigma_k = np.load(os.path.join(var_data_dir, 'Sigma_k.npy'))
    Vt_k = np.load(os.path.join(var_data_dir, 'Vt_k.npy'))
    mean_ensemble = np.load(os.path.join(var_data_dir, 'Mean_ensemble.npy'))
    mean_spatial = np.load(os.path.join(var_data_dir, 'Mean_spatial.npy'))
    spatial_norms = np.load(os.path.join(var_data_dir, 'Spatial_norms.npy'))
    
    k = Phi_k.shape[1]
    print(f"POD configuration for {var}: {k} modes retained.")
    
    # Targets for GPR
    # We predict: 1) Vt_k coefficients (shape: k x N), 2) mean_spatial (shape: N), 3) spatial_norms (shape: N)
    # Concatenate targets to run them in a single batch GP model
    Y = np.vstack([Vt_k, mean_spatial[np.newaxis, :], spatial_norms[np.newaxis, :]])
    num_targets = Y.shape[0] # k + 2
    
    # 80/20 Train/Test split
    indices = np.arange(num_snapshots)
    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=42)
    print(f"Data Split:")
    print(f"  Training set size: {len(train_idx)} cases")
    print(f"  Testing set size: {len(test_idx)} cases")
    
    X_train, X_test = X_dimensionless[train_idx], X_dimensionless[test_idx]
    Y_train, Y_test = Y[:, train_idx], Y[:, test_idx]
    
    # Scale inputs using MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 5-Fold Cross Validation on the training set
    print("------------------------------------------------------------")
    print("Performing 5-Fold Cross-Validation on the training set...")
    print("------------------------------------------------------------")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_r2_all = []
    cv_nrmse_all = []
    
    fold_idx = 1
    for tr_split_idx, val_split_idx in kf.split(X_train_scaled):
        t_fold_start = time.time()
        X_tr, X_val = X_train_scaled[tr_split_idx], X_train_scaled[val_split_idx]
        Y_tr, Y_val = Y_train[:, tr_split_idx], Y_train[:, val_split_idx]
        
        # Convert to torch tensors
        X_tr_t = torch.from_numpy(X_tr).float().unsqueeze(0).repeat(num_targets, 1, 1)
        Y_tr_t = torch.from_numpy(Y_tr).float()
        
        # Train batch GP model
        model_cv, likelihood_cv = train_batch_gpr(X_tr_t, Y_tr_t, lr=0.1, training_iter=120)
        
        # Predict on validation fold
        pred_val = predict_batch_gpr(model_cv, likelihood_cv, X_val)
        
        # Compute R2 and NRMSE for this fold across all targets
        ss_res = np.sum((Y_val - pred_val) ** 2, axis=1)
        Y_val_mean = np.mean(Y_val, axis=1, keepdims=True)
        ss_tot = np.sum((Y_val - Y_val_mean) ** 2, axis=1) + 1e-12
        fold_r2 = 1.0 - (ss_res / ss_tot)
        
        rmse = np.sqrt(np.mean((Y_val - pred_val) ** 2, axis=1))
        y_range = np.max(Y_tr, axis=1) - np.min(Y_tr, axis=1) + 1e-12
        fold_nrmse = rmse / y_range * 100
        
        cv_r2_all.append(fold_r2)
        cv_nrmse_all.append(fold_nrmse)
        
        print(f"  Fold {fold_idx}/5 Completed | Avg Coefficient R2: {np.mean(fold_r2[:k]):.4f} | Avg Coefficient NRMSE: {np.mean(fold_nrmse[:k]):.4f}% | Time: {time.time() - t_fold_start:.2f}s")
        fold_idx += 1
        
    # Calculate average CV metrics across folds
    cv_r2_mean = np.mean(cv_r2_all, axis=0) # shape: (num_targets,)
    cv_nrmse_mean = np.mean(cv_nrmse_all, axis=0) # shape: (num_targets,)
    
    avg_coeff_cv_r2 = np.mean(cv_r2_mean[:k])
    avg_coeff_cv_nrmse = np.mean(cv_nrmse_mean[:k])
    
    print(f"Cross-Validation Summary (Averaged over 5 folds):")
    print(f"  Mean R2 of POD Coefficients: {avg_coeff_cv_r2:.5f}")
    print(f"  Mean NRMSE of POD Coefficients: {avg_coeff_cv_nrmse:.4f}%")
    print(f"  Mean R2 of Spatial Mean GPR: {cv_r2_mean[k]:.5f}")
    print(f"  Mean R2 of Spatial Norm GPR: {cv_r2_mean[k+1]:.5f}")
    
    # 3. Train final GPR models on the full training set
    print("------------------------------------------------------------")
    print("Training final GPR models on full training dataset (360 cases)...")
    print("------------------------------------------------------------")
    X_train_t = torch.from_numpy(X_train_scaled).float().unsqueeze(0).repeat(num_targets, 1, 1)
    Y_train_t = torch.from_numpy(Y_train).float()
    
    final_model, final_likelihood = train_batch_gpr(X_train_t, Y_train_t, lr=0.1, training_iter=150, verbose=True)
    
    # 4. Predict on the test set (90 cases)
    print("Predicting coefficients and parameters on test set...")
    Y_test_pred = predict_batch_gpr(final_model, final_likelihood, X_test_scaled)
    
    pred_Vt = Y_test_pred[:k] # shape: (k, 90)
    pred_mean_spatial = Y_test_pred[k] # shape: (90,)
    pred_spatial_norms = Y_test_pred[k+1] # shape: (90,)
    
    # 5. Field Reconstruction and Validation
    print("------------------------------------------------------------")
    print("Reconstructing physical fields for validation...")
    print("------------------------------------------------------------")
    
    # Pre-interpolate actual fields for the test set
    s_1d = np.linspace(0.0, 2.0, grid_s)
    theta_1d = np.linspace(-np.pi, np.pi, grid_theta)
    s_grid, theta_grid = np.meshgrid(s_1d, theta_1d, indexing='ij')
    
    test_files = [all_files[idx] for idx in test_idx]
    
    print("Interpolating physical test snapshots (parallelized)...")
    t_interp_start = time.time()
    test_interpolated_results = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_single_test_file)(
            file_name, [var], s_grid, theta_grid, input_dir
        )
        for file_name in tqdm(test_files, desc="Interpolating test files")
    )
    print(f"Pre-interpolation completed in {time.time() - t_interp_start:.2f} seconds.")
    
    # Store actual fields in order of test_idx
    test_file_to_data = dict(test_interpolated_results)
    actual_test_fields = []
    for file_name in test_files:
        actual_test_fields.append(test_file_to_data[file_name][var])
        
    # Reconstruct predictions
    # S_recon_centered = Phi_k @ (Sigma_k * pred_Vt)
    S_recon_centered = Phi_k @ (Sigma_k[:, np.newaxis] * pred_Vt)
    
    test_case_results = []
    
    sum_err_sq = 0.0
    total_elements = 0
    global_max_actual = -np.inf
    global_min_actual = np.inf
    
    # Compute error metrics for every test snapshot (every "time")
    for idx in range(len(test_idx)):
        case_name = test_files[idx].replace('.csv', '')
        actual = actual_test_fields[idx]
        
        # Reconstruction formula
        recon_centered = S_recon_centered[:, idx] + mean_ensemble[:, 0]
        recon = recon_centered * pred_spatial_norms[idx] + pred_mean_spatial[idx]
        
        # Compute error vector
        err = actual - recon
        abs_err = np.abs(err)
        err_sq = err ** 2
        
        # Metrics
        mae = float(np.mean(abs_err))
        max_ae = float(np.max(abs_err))
        rmse = float(np.sqrt(np.mean(err_sq)))
        
        actual_range = float(np.max(actual) - np.min(actual))
        nrmse = float(rmse / (actual_range + 1e-12) * 100)
        
        actual_norm = np.linalg.norm(actual)
        l2_err = float((np.linalg.norm(err) / (actual_norm + 1e-12)) * 100)
        
        ss_tot = np.sum((actual - np.mean(actual)) ** 2) + 1e-12
        r2 = float(1.0 - (np.sum(err_sq) / ss_tot))
        
        # Keep track of global metrics
        sum_err_sq += np.sum(err_sq)
        total_elements += len(actual)
        global_max_actual = max(global_max_actual, np.max(actual))
        global_min_actual = min(global_min_actual, np.min(actual))
        
        # Extract inputs for reporting
        inputs = X[test_idx[idx]]
        br = float(inputs[0])
        vel = float(inputs[1])
        dens = float(inputs[2])
        size = float(inputs[3])
        
        test_case_results.append({
            'case_name': case_name,
            'bend_ratio': br,
            'velocity': vel,
            'particle_density': dens,
            'particle_size': size,
            'reynolds': float(X_test[idx, 0]),
            'density_ratio': float(X_test[idx, 1]),
            'size_ratio': float(X_test[idx, 2]),
            'mae': mae,
            'max_ae': max_ae,
            'rmse': rmse,
            'nrmse_pct': nrmse,
            'l2_error_pct': l2_err,
            'r2': r2
        })
        
    # Global stats
    global_rmse = float(np.sqrt(sum_err_sq / total_elements))
    global_range = float(global_max_actual - global_min_actual)
    global_nrmse = float(global_rmse / (global_range + 1e-12) * 100)
    
    avg_mae = float(np.mean([r['mae'] for r in test_case_results]))
    avg_max_ae = float(np.mean([r['max_ae'] for r in test_case_results]))
    max_of_max_ae = float(np.max([r['max_ae'] for r in test_case_results]))
    avg_r2 = float(np.mean([r['r2'] for r in test_case_results]))
    avg_l2_err = float(np.mean([r['l2_error_pct'] for r in test_case_results]))
    
    print("\n=================== TEST SET PERFORMANCE ===================")
    print(f"Global RMSE: {global_rmse:.6e}")
    print(f"Global Range-Normalized RMSE: {global_nrmse:.4f}%")
    print(f"Average MAE across test cases: {avg_mae:.6e}")
    print(f"Average Max Absolute Error: {avg_max_ae:.6e}")
    print(f"Worst Max Absolute Error: {max_of_max_ae:.6e}")
    print(f"Average Field Reconstruction R2: {avg_r2:.5f}")
    print(f"Average Relative L2 Error: {avg_l2_err:.4f}%")
    print("============================================================")
    
    # Save results to JSON
    summary = {
        'variable': var,
        'modes_retained': k,
        'cv_averages': {
            'coeff_r2': float(avg_coeff_cv_r2),
            'coeff_nrmse_pct': float(avg_coeff_cv_nrmse),
            'mean_spatial_r2': float(cv_r2_mean[k]),
            'spatial_norm_r2': float(cv_r2_mean[k+1])
        },
        'test_averages': {
            'mae': avg_mae,
            'max_ae': avg_max_ae,
            'max_of_max_ae': max_of_max_ae,
            'r2': avg_r2,
            'l2_error_pct': avg_l2_err,
            'global_nrmse_pct': global_nrmse
        },
        'test_cases': test_case_results
    }
    
    output_json = 'gpr_gpu_results.json'
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Saved results details to {output_json}")
    
    # Write Markdown Report
    report_path = 'gpr_gpu_pipeline_report.md'
    with open(report_path, 'w') as r:
        r.write(f"# GPU-Accelerated Gaussian Process Regression (GPR) Report\n\n")
        r.write(f"This report presents the validation results for the GPU-accelerated GPR surrogate model developed for the `{var}` field in the `pod_individual_999` dataset.\n\n")
        
        r.write("## Methodology Setup\n")
        r.write("- **Variable Modelled**: `oka_erosion_norm` (Oka erosion function).\n")
        r.write("- **Retained POD Modes**: 325 modes (capturing 99.9% of the dataset energy).\n")
        r.write("- **Dimensionless Input Parameters**:\n")
        r.write("  1. **Reynolds Number** ($Re$): $\\rho_f = 998.2 \\text{ kg/m}^3$, $\\mu = 0.001003 \\text{ Pa.s}$, $D = 0.0254 \\text{ m}$, velocity $U$ from dataset.\n")
        r.write("  2. **Density Ratio**: particle density $\\rho_p$ to fluid density $\\rho_f$ ratio.\n")
        r.write("  3. **Size Ratio**: particle diameter $d_p$ to pipe diameter $D$ ratio ($D = 0.0254 \\text{ m}$).\n")
        r.write("  4. **Bend Ratio**: inlet bend radius to pipe diameter ($B_r$).\n")
        r.write("- **Data Division**: 80% Train (360 cases), 20% Test (90 cases).\n")
        r.write("- **Cross-Validation**: 5-Fold Cross-Validation on the training dataset. GP hyperparameters optimized on GPU via GPyTorch (batch mode).\n")
        r.write("- **Surrogate Architecture**: 327 independent GPs (325 modes + 1 spatial mean + 1 spatial norm) trained simultaneously on GPU.\n\n")
        
        r.write("## Performance Summary\n")
        r.write("### 5-Fold Cross-Validation (Training Set)\n")
        r.write(f"- Average $R^2$ of POD Coefficients: **{avg_coeff_cv_r2:.5f}**\n")
        r.write(f"- Average NRMSE of POD Coefficients: **{avg_coeff_cv_nrmse:.4f}%**\n")
        r.write(f"- Average $R^2$ of Spatial Mean: **{cv_r2_mean[k]:.5f}**\n")
        r.write(f"- Average $R^2$ of Spatial Norm: **{cv_r2_mean[k+1]:.5f}**\n\n")
        
        r.write("### Validation Test Set Performance (Unseen snapshots)\n")
        r.write(f"- Average Mean Absolute Error (MAE): **{avg_mae:.6e}**\n")
        r.write(f"- Average Maximum Absolute Error (MaxAE): **{avg_max_ae:.6e}**\n")
        r.write(f"- Worst-case Maximum Absolute Error (MaxAE): **{max_of_max_ae:.6e}**\n")
        r.write(f"- Average Field Reconstruction $R^2$: **{avg_r2:.5f}**\n")
        r.write(f"- Average Field Relative $L_2$ Error: **{avg_l2_err:.4f}%**\n")
        r.write(f"- Global Range-Normalized RMSE (NRMSE): **{global_nrmse:.4f}%**\n\n")
        
        r.write("## Test Cases Error Metrics Table\n")
        r.write("Below is the full evaluation table reporting the average and maximum absolute error, NRMSE, and relative L2 error for **every test case (time/snapshot)** in the 20% validation split.\n\n")
        r.write("| Test Index | Case Name | Bend Ratio | Velocity (m/s) | Particle Density (kg/m³) | Particle Size (µm) | Average Error (MAE) | Maximum Error (MaxAE) | NRMSE (%) | Rel L2 Error (%) | Field $R^2$ |\n")
        r.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        
        for idx, res in enumerate(test_case_results):
            r.write(f"| {idx+1} | `{res['case_name']}` | {res['bend_ratio']:.2f} | {res['velocity']:.1f} | {res['particle_density']:.1f} | {res['particle_size']:.1f} | {res['mae']:.4e} | {res['max_ae']:.4e} | {res['nrmse_pct']:.4f}% | {res['l2_error_pct']:.4f}% | {res['r2']:.5f} |\n")
            
    print(f"Saved Markdown report to {report_path}")
    print("GPU GPR ROM Pipeline execution completed successfully.")

if __name__ == '__main__':
    main()
