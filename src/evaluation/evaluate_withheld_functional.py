import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

import numpy as np
import os
import matplotlib.pyplot as plt

# 1. Fit the angular function to the basis
angles = np.linspace(0, np.pi/2, 1000)
f_true = (np.sin(angles)**0.8) * ((1 + 0.5 * np.cos(angles))**1.3)

X_basis = np.column_stack([
    np.ones_like(angles),          # c0
    np.sin(angles),                # c1
    np.sin(angles)**2,             # c2
    np.sin(angles)**3,             # c3
    np.sin(angles)*np.cos(angles), # c4
    angles,                        # c5
    angles**2                      # c6
])

coeffs, _, _, _ = np.linalg.lstsq(X_basis, f_true, rcond=None)
print("Fitted coefficients:", coeffs)

# 2. Load predictions
preds_path = os.path.join(_REPO_ROOT, 'block_pod_powerlaw_study/test_set_predictions.npy')
data = np.load(preds_path, allow_pickle=True).item()

# 3. Variables
var_mean = 'v25_mean_norm'
var_alpha = 'v25_alpha_norm'
var_alpha2 = 'v25_alpha2_norm'
var_sin = 'v25_sin_norm'
var_sin2 = 'v25_sin2_norm'
var_sin3 = 'v25_sin3_norm'
var_sincos = 'v25_sincos_norm'

vars_list = [var_mean, var_sin, var_sin2, var_sin3, var_sincos, var_alpha, var_alpha2]

def get_field(source, var):
    return np.nan_to_num(np.clip(data[source][var], 0.0, None), nan=0.0)

# Evaluate on first 15 cases
num_cases = min(15, data['truth'][var_mean].shape[0])
print(f"Evaluating on {num_cases} cases...")

err_trunc_l2 = []
err_trunc_r2 = []
err_trunc_peak = []
err_trunc_fw_r2 = []

err_e2e_l2 = []
err_e2e_r2 = []
err_e2e_peak = []
err_e2e_fw_r2 = []

for i in range(num_cases):
    # CFD Truth Moments
    v25_mean = get_field('truth', var_mean)[i]
    v25_alpha = get_field('truth', var_alpha)[i]
    v25_alpha2 = get_field('truth', var_alpha2)[i]
    v25_sin = get_field('truth', var_sin)[i]
    v25_sin2 = get_field('truth', var_sin2)[i]
    v25_sin3 = get_field('truth', var_sin3)[i]
    v25_sincos = get_field('truth', var_sincos)[i]
    
    # GPR Predicted Moments
    v25_mean_gpr = get_field('gpr', var_mean)[i]
    v25_alpha_gpr = get_field('gpr', var_alpha)[i]
    v25_alpha2_gpr = get_field('gpr', var_alpha2)[i]
    v25_sin_gpr = get_field('gpr', var_sin)[i]
    v25_sin2_gpr = get_field('gpr', var_sin2)[i]
    v25_sin3_gpr = get_field('gpr', var_sin3)[i]
    v25_sincos_gpr = get_field('gpr', var_sincos)[i]
    
    # 1. Ground Truth E_true (Jensen)
    valid_mask = v25_mean > 1e-6
    A = np.zeros_like(v25_mean)
    A[valid_mask] = v25_alpha[valid_mask] / v25_mean[valid_mask]
    A = np.clip(A, 0.0, np.pi/2)
    f_A = (np.sin(A)**0.8) * ((1 + 0.5 * np.cos(A))**1.3)
    E_true = v25_mean * f_A
    
    # 2. Perfect Moment Recon
    E_recon_cfd = (coeffs[0]*v25_mean + coeffs[1]*v25_sin + coeffs[2]*v25_sin2 + 
                   coeffs[3]*v25_sin3 + coeffs[4]*v25_sincos + coeffs[5]*v25_alpha + coeffs[6]*v25_alpha2)
    E_recon_cfd = np.clip(E_recon_cfd, 0.0, None)
    
    # 3. Surrogate Recon
    E_recon_gpr = (coeffs[0]*v25_mean_gpr + coeffs[1]*v25_sin_gpr + coeffs[2]*v25_sin2_gpr + 
                   coeffs[3]*v25_sin3_gpr + coeffs[4]*v25_sincos_gpr + coeffs[5]*v25_alpha_gpr + coeffs[6]*v25_alpha2_gpr)
    E_recon_gpr = np.clip(E_recon_gpr, 0.0, None)
    
    def calc_metrics(true, pred):
        mse = np.mean((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2) + 1e-12
        r2 = 1.0 - np.sum((true - pred)**2) / ss_tot
        
        rel_l2 = np.linalg.norm(true - pred) / (np.linalg.norm(true) + 1e-12) * 100
        
        peak_err = np.abs(np.max(pred) - np.max(true)) / (np.max(true) + 1e-12) * 100
        
        M_val = np.max(np.abs(true)) + 1e-8
        W = 1.0 + 2.0 * (np.abs(true) / M_val)
        fw_mse = np.average((true - pred)**2, weights=W)
        a_w_mean = np.average(true, weights=W)
        ss_tot_w = np.average((true - a_w_mean)**2, weights=W)
        fw_r2 = 1.0 - (fw_mse / ss_tot_w) if ss_tot_w > 0 else 0.0
        
        return r2, rel_l2, peak_err, fw_r2
        
    r2_trunc, l2_trunc, peak_trunc, fw_r2_trunc = calc_metrics(E_true, E_recon_cfd)
    err_trunc_r2.append(r2_trunc)
    err_trunc_l2.append(l2_trunc)
    err_trunc_peak.append(peak_trunc)
    err_trunc_fw_r2.append(fw_r2_trunc)
    
    r2_e2e, l2_e2e, peak_e2e, fw_r2_e2e = calc_metrics(E_true, E_recon_gpr)
    err_e2e_r2.append(r2_e2e)
    err_e2e_l2.append(l2_e2e)
    err_e2e_peak.append(peak_e2e)
    err_e2e_fw_r2.append(fw_r2_e2e)

print("--- Truncation Error (epsilon_g) ---")
print(f"R2: {np.mean(err_trunc_r2):.4f}")
print(f"FW-R2: {np.mean(err_trunc_fw_r2):.4f}")
print(f"Rel L2: {np.mean(err_trunc_l2):.2f}%")
print(f"Peak Error: {np.mean(err_trunc_peak):.2f}%")

print("\n--- End-to-End ROM Error ---")
print(f"R2: {np.mean(err_e2e_r2):.4f}")
print(f"FW-R2: {np.mean(err_e2e_fw_r2):.4f}")
print(f"Rel L2: {np.mean(err_e2e_l2):.2f}%")
print(f"Peak Error: {np.mean(err_e2e_peak):.2f}%")

os.makedirs(os.path.join(_REPO_ROOT, 'latex_paper_draft/figures'), exist_ok=True)

# Generate a figure comparing true, truncation, and surrogate for one representative case
i_rep = 0 # Case 0
v25_mean = get_field('truth', var_mean)[i_rep]
v25_alpha = get_field('truth', var_alpha)[i_rep]
v25_alpha2 = get_field('truth', var_alpha2)[i_rep]
v25_sin = get_field('truth', var_sin)[i_rep]
v25_sin2 = get_field('truth', var_sin2)[i_rep]
v25_sin3 = get_field('truth', var_sin3)[i_rep]
v25_sincos = get_field('truth', var_sincos)[i_rep]
valid_mask = v25_mean > 1e-6
A = np.zeros_like(v25_mean)
A[valid_mask] = v25_alpha[valid_mask] / v25_mean[valid_mask]
A = np.clip(A, 0.0, np.pi/2)
f_A = (np.sin(A)**0.8) * ((1 + 0.5 * np.cos(A))**1.3)
E_true = v25_mean * f_A

E_recon_cfd = (coeffs[0]*v25_mean + coeffs[1]*v25_sin + coeffs[2]*v25_sin2 + 
               coeffs[3]*v25_sin3 + coeffs[4]*v25_sincos + coeffs[5]*v25_alpha + coeffs[6]*v25_alpha2)
E_recon_cfd = np.clip(E_recon_cfd, 0.0, None)

v25_mean_gpr = get_field('gpr', var_mean)[i_rep]
v25_alpha_gpr = get_field('gpr', var_alpha)[i_rep]
v25_alpha2_gpr = get_field('gpr', var_alpha2)[i_rep]
v25_sin_gpr = get_field('gpr', var_sin)[i_rep]
v25_sin2_gpr = get_field('gpr', var_sin2)[i_rep]
v25_sin3_gpr = get_field('gpr', var_sin3)[i_rep]
v25_sincos_gpr = get_field('gpr', var_sincos)[i_rep]
E_recon_gpr = (coeffs[0]*v25_mean_gpr + coeffs[1]*v25_sin_gpr + coeffs[2]*v25_sin2_gpr + 
               coeffs[3]*v25_sin3_gpr + coeffs[4]*v25_sincos_gpr + coeffs[5]*v25_alpha_gpr + coeffs[6]*v25_alpha2_gpr)
E_recon_gpr = np.clip(E_recon_gpr, 0.0, None)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im1 = axes[0].imshow(E_true.reshape(582, 221), aspect='auto', cmap='jet')
axes[0].set_title('Ground Truth (Jensen)')
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(E_recon_cfd.reshape(582, 221), aspect='auto', cmap='jet')
axes[1].set_title('Basis Truncation (CFD Moments)')
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(E_recon_gpr.reshape(582, 221), aspect='auto', cmap='jet')
axes[2].set_title('End-to-End ROM (GPR Moments)')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(os.path.join(_REPO_ROOT, 'latex_paper_draft/figures/withheld_model_comparison.png'), dpi=300)
print("Saved figure to latex_paper_draft/figures/withheld_model_comparison.png")
