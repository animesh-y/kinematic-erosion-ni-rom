import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

import numpy as np
import matplotlib.pyplot as plt
import os

exp_data_dir = os.path.join(_REPO_ROOT, 'Mesh_independence/Experimental_data')

# 1. Parse CFD data
lines = open(os.path.join(_REPO_ROOT, 'Mesh_independence/experimental_validation.prof')).readlines()
fields = {}
current_field = None
for line in lines:
    line = line.strip()
    if line.startswith('('):
        parts = line.replace('(', '').split()
        if len(parts) == 1 or parts[0] in ['x', 'y', 'z'] or parts[0].startswith('udm'):
            current_field = parts[0]
            fields[current_field] = []
    elif current_field and not line.startswith(')') and line:
        try: fields[current_field].append(float(line.replace(')', '')))
        except: pass

x = np.array(fields['x']); y = np.array(fields['y']); z = np.array(fields['z'])
R_b = 0.1524
R_pipe = 0.0508

s = np.zeros_like(x); theta = np.zeros_like(x); bend_angle_deg = np.zeros_like(x)
m_in = x <= 0
bend_angle_deg[m_in] = (x[m_in] / R_b) * (180/np.pi)
theta[m_in] = np.arctan2(z[m_in], -(y[m_in] + R_b))

m_b = (x > 0) & (y < 0)
beta = np.arctan2(x[m_b], -y[m_b]) 
bend_angle_deg[m_b] = beta * (180/np.pi)
r_xy = np.sqrt(x[m_b]**2 + y[m_b]**2)
theta[m_b] = np.arctan2(z[m_b], r_xy - R_b)

m_out = y >= 0
bend_angle_deg[m_out] = 90 + (y[m_out] / R_b) * (180/np.pi)
theta[m_out] = np.arctan2(z[m_out], x[m_out] - R_b)
theta_deg = theta * (180/np.pi)

u = {}
for i in range(37):
    k = f'udm-{i}'
    if k in fields: u[i] = np.array(fields[k])

N = u[0].copy(); m_valid = N > 0; N[~m_valid] = 1.0
E_V1 = u[1] / N; E_V2 = u[2] / N; E_V25 = u[18] / N; E_V3 = u[3] / N; E_A = u[9] / N

def log_interp(target_p, p_array, val_array):
    safe_vals = np.maximum(val_array, 1e-30)
    return np.exp(np.interp(target_p, p_array, np.log(safe_vals)))

p_V = np.array([1.0, 2.0, 2.5, 3.0])
E_V173 = np.zeros_like(N); E_V2338 = np.zeros_like(N); E_V241 = np.zeros_like(N)
for i in range(len(N)):
    if m_valid[i]:
        vals_V = np.array([E_V1[i], E_V2[i], E_V25[i], E_V3[i]])
        E_V173[i] = log_interp(1.73, p_V, vals_V)
        E_V2338[i] = log_interp(2.338, p_V, vals_V)
        E_V241[i] = log_interp(2.41, p_V, vals_V)

g_A = (np.sin(E_A)**0.753) * ((1.0 + 1.53 * (1.0 - np.sin(E_A)))**1.59)
ER_oka = E_V2338 * g_A * N

f_mclaury = np.zeros_like(E_A)
mask1 = E_A <= 0.2618; mask2 = E_A > 0.2618
f_mclaury[mask1] = -34.79 * (E_A[mask1]**2) + 12.3 * E_A[mask1]
f_mclaury[mask2] = -34.79 * (0.2618**2) + 12.3 * 0.2618 
ER_mc = E_V173 * f_mclaury * N

thresh = np.arctan(0.4)
mask1 = E_A <= thresh; mask2 = E_A > thresh
vol_c = np.zeros_like(E_V1)
vol_c[mask1] = (E_V241[mask1]) * np.sin(E_A[mask1]) * (2*0.4*np.cos(E_A[mask1]) - np.sin(E_A[mask1])) / (2 * 0.4**2)
vol_c[mask2] = (E_V241[mask2]) * (np.cos(E_A[mask2])**2) / 2.0
vol_d = 0.5 * (E_V2 * (np.sin(E_A)**2))
ER_arab = (vol_c + vol_d) * N

def extract_and_smooth(er_field, phi_target, tol=3.0):
    mask = np.abs(np.abs(theta_deg) - phi_target) < tol
    ang = bend_angle_deg[mask]
    val = er_field[mask]
    
    if len(ang) == 0:
        return np.array([]), np.array([])
        
    idx = np.argsort(ang)
    ang = ang[idx]
    val = val[idx]
    
    box_pts = 5
    if box_pts > 1:
        box = np.ones(box_pts)/box_pts
        val = np.convolve(val, box, mode='same')
    return ang, val

ang_0, oka_0 = extract_and_smooth(ER_oka, 0)
_, mc_0 = extract_and_smooth(ER_mc, 0)
_, arab_0 = extract_and_smooth(ER_arab, 0)

max_oka_extrados = np.max(oka_0)
max_mc_extrados = np.max(mc_0)
max_arab_extrados = np.max(arab_0)

# The LSQ Multipliers obtained from extrados only:
m_oka = 1.285
m_mc = 1.331
m_arab = 1.308

exp_max = 1.5935955376047546 # global max is at extrados

targets = [
    ('A.csv', 0, 'Extrados Centerline (0°)'),
    ('B.csv', 9, '9° Circumferential Plane'),
    ('D.csv', 27, '27° Circumferential Plane'),
    ('F.csv', 45, '45° Circumferential Plane'),
    ('H.csv', 63, '63° Circumferential Plane'),
    ('K.csv', 81, '81° Circumferential Plane')
]

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.flatten()

for i, (fname, phi, title) in enumerate(targets):
    exp_data = np.loadtxt(os.path.join(exp_data_dir, fname), delimiter=',')
    exp_idx = np.argsort(exp_data[:,0])
    exp_ang = exp_data[exp_idx, 0]
    exp_val = exp_data[exp_idx, 1] / exp_max 
    
    ang_cfd, oka_cfd = extract_and_smooth(ER_oka, phi)
    _, mc_cfd = extract_and_smooth(ER_mc, phi)
    _, arab_cfd = extract_and_smooth(ER_arab, phi)
    
    oka_scaled = (oka_cfd / max_oka_extrados) * m_oka
    mc_scaled = (mc_cfd / max_mc_extrados) * m_mc
    arab_scaled = (arab_cfd / max_arab_extrados) * m_arab
    
    oka_samp = np.interp(exp_ang, ang_cfd, oka_scaled)
    mc_samp = np.interp(exp_ang, ang_cfd, mc_scaled)
    arab_samp = np.interp(exp_ang, ang_cfd, arab_scaled)
    
    def get_r2(true, pred):
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - np.mean(true))**2)
        return 1 - (ss_res / (ss_tot + 1e-12))
    
    r2_oka = get_r2(exp_val, oka_samp)
    r2_mc = get_r2(exp_val, mc_samp)
    r2_arab = get_r2(exp_val, arab_samp)
    
    print(f"{fname} ({phi}°): R2 Oka={r2_oka:.3f}, Mc={r2_mc:.3f}, Arab={r2_arab:.3f}")
    
    ax = axes[i]
    ax.plot(exp_ang, exp_val, color='black', lw=2.5, marker='o', markersize=5, label='Solnordal (Exp)')
    ax.plot(exp_ang, oka_samp, color='#d95f02', lw=2.5, marker='s', markersize=4, label=f'Oka (R²={r2_oka:.2f})')
    ax.plot(exp_ang, mc_samp, color='#1b9e77', lw=2.5, marker='^', markersize=5, label=f'McLaury (R²={r2_mc:.2f})')
    ax.plot(exp_ang, arab_samp, color='#7570b3', lw=2.5, marker='D', markersize=4, label=f'Arabnejad (R²={r2_arab:.2f})')
    
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlim(-5, 95)
    ax.set_ylim(-0.05, 1.45)
    if i >= 4: ax.set_xlabel('Bend Angle (Degrees)')
    if i % 2 == 0: ax.set_ylabel('Normalized E/E_max')
    ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(_REPO_ROOT, 'latex_paper_draft/figures/solnordal_lateral_profiles.png'), dpi=300)
print("Saved grid plot of lateral profiles.")

