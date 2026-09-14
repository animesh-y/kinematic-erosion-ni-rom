import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

import numpy as np
import matplotlib.pyplot as plt
import os

# Publication grade settings
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'font.family': 'sans-serif'
})

def compute_erosion_exact(fields, dp=150.0):
    v = np.maximum(fields['v_mean_norm'], 1e-8)
    v2_sin = np.maximum(fields['v2_sin_norm'], 0.0)
    v2_sincos = np.maximum(fields['v2_sincos_norm'], 0.0)
    v2_sin2 = np.maximum(fields['v2_sin2_norm'], 0.0)
    
    # We will include v2_mean_norm to show its sensitivity is exactly 0
    v2_mean = np.maximum(fields.get('v2_mean_norm', 0.0), 0.0)
    
    v_eff2 = np.maximum(v**2, 1e-8)
    sin_a = np.clip(v2_sin / v_eff2, 0.0, 1.0)
    alpha = np.arcsin(sin_a)
    
    # 1. Oka
    E90_fac = 65.0 * (1.8 ** -0.12) * ((dp / 326.0) ** 0.19)
    k2 = 2.353
    v_oka = (v / 104.0) ** k2
    E90 = E90_fac * v_oka
    g_alpha = (sin_a ** 0.8) * ((1.0 + 1.8 * (1.0 - sin_a)) ** 1.3)
    oka = E90 * g_alpha
    
    # 2. Finnie
    m_shallow = alpha <= 0.32288
    finnie = np.zeros_like(v)
    finnie = np.where(m_shallow, 2.0 * v2_sincos - 3.0 * v2_sin2, (1.0 / 3.0) * (v_eff2 - v2_sin2))
    finnie = np.clip(finnie, 0.0, None)
    
    # 3. McLaury
    v_mcl = v ** 1.73
    f_mcl = np.where(alpha <= 0.32266, -34.79 * (alpha**2) + 12.3 * alpha, 
                     0.3147 * (np.cos(alpha)**2) * np.sin(alpha) + 0.03609 * (np.sin(alpha)**2) + 0.2532)
    mclaury = np.clip(v_mcl * f_mcl, 0, None)
    
    # 4. Arabnejad
    v_arab = v ** 2.41
    th_a = np.arctan(0.4)
    vol_c = np.where(alpha <= th_a,
                     v_arab * sin_a * (2*0.4*np.cos(alpha) - sin_a) / (2 * 0.4**2),
                     v_arab * (np.cos(alpha)**2) / 2.0)
    vol_d = 0.5 * v2_sin2
    arab = np.clip(vol_c + vol_d, 0, None)
    
    return {'Oka': oka, 'Finnie': finnie, 'McLaury': mclaury, 'Arabnejad': arab}

def main():
    # Representative values
    v_mean_val = 15.0
    alpha_val = 11.5 * np.pi / 180.0
    v2_mean_val = v_mean_val**2 + 2.0 # Some variance
    
    base_fields = {
        'v_mean_norm': v_mean_val,
        'v2_mean_norm': v2_mean_val,
        'v2_sin_norm': v2_mean_val * np.sin(alpha_val),
        'v2_sincos_norm': v2_mean_val * np.sin(alpha_val) * np.cos(alpha_val),
        'v2_sin2_norm': v2_mean_val * (np.sin(alpha_val)**2),
    }
    
    moments = ['v_mean_norm', 'v2_mean_norm', 'v2_sin_norm', 'v2_sincos_norm', 'v2_sin2_norm']
    models = ['Oka', 'Finnie', 'McLaury', 'Arabnejad']
    
    delta = 1e-5
    base = compute_erosion_exact(base_fields)
    
    sensitivity_matrix = np.zeros((len(models), len(moments)))
    
    for j, m_name in enumerate(moments):
        p_fields = base_fields.copy()
        p_fields[m_name] += delta
        p_res = compute_erosion_exact(p_fields)
        
        for i, mod in enumerate(models):
            dE = (p_res[mod] - base[mod]) / delta
            # Elasticity
            elasticity = dE * (base_fields[m_name] / base[mod]) if base[mod] > 0 else 0
            sensitivity_matrix[i, j] = elasticity
            
    print("Sensitivity matrix:")
    print(sensitivity_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.imshow(sensitivity_matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, label='Elasticity (Sensitivity)')
    
    ax.set_xticks(np.arange(len(moments)))
    ax.set_xticklabels(moments, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    
    for (i, j), z in np.ndenumerate(sensitivity_matrix):
        ax.text(j, i, f'{z:0.3f}', ha='center', va='center', 
                color='white' if np.abs(z) > 0.5 else 'black')
    
    plt.title('Relative Sensitivity of Cross-Moments in Exact Formulation', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.join(_REPO_ROOT, 'latex_paper_draft/figures'), exist_ok=True)
    out_path = os.path.join(_REPO_ROOT, 'latex_paper_draft/figures/fig_moment_sensitivity.png')
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == '__main__':
    main()
