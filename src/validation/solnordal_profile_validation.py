import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

import numpy as np
import os

exp_data_dir = os.path.join(_REPO_ROOT, 'Mesh_independence/Experimental_data')

# Parse CFD data
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
E_V2338 = np.zeros_like(N)
for i in range(len(N)):
    if m_valid[i]:
        vals_V = np.array([E_V1[i], E_V2[i], E_V25[i], E_V3[i]])
        E_V2338[i] = log_interp(2.338, p_V, vals_V)

# DNV / Piecewise model (Solnordal baseline)
# E = 1.44e-8 * V^2.2 * f(alpha)
E_V22 = np.zeros_like(N)
for i in range(len(N)):
    if m_valid[i]:
        vals_V = np.array([E_V1[i], E_V2[i], E_V25[i], E_V3[i]])
        E_V22[i] = log_interp(2.2, p_V, vals_V)

f_dnv = np.zeros_like(E_A)
m_shallow = E_A <= 0.4014
f_dnv[m_shallow] = -7.0 * (E_A[m_shallow]**2) + 5.45 * E_A[m_shallow]
f_dnv[~m_shallow] = 0.4 * (np.cos(E_A[~m_shallow])**2) * np.sin(-3.4 * E_A[~m_shallow]) - 0.9 * (np.sin(E_A[~m_shallow])**2) + 1.556

ER_dnv = E_V22 * f_dnv * N

def extract_and_smooth(er_field, phi_target, tol=3.0):
    mask = np.abs(np.abs(theta_deg) - phi_target) < tol
    ang = bend_angle_deg[mask]
    val = er_field[mask]
    if len(ang) == 0:
        return np.array([]), np.array([])
    idx = np.argsort(ang)
    ang = ang[idx]
    val = val[idx]
    box = np.ones(5)/5
    val = np.convolve(val, box, mode='same')
    return ang, val

planes = [
    ('A (Centerline 0°)', 'A.csv', 0),
    ('B (9° Off-center)', 'B.csv', 9),
    ('D (27° Off-center)', 'D.csv', 27),
    ('F (45° Off-center)', 'F.csv', 45),
    ('H (63° Off-center)', 'H.csv', 63),
    ('K (81° Sidewall Crown)', 'K.csv', 81),
]

# Scale factor from centerline peak
ang_0, val_0 = extract_and_smooth(ER_dnv, 0)
exp_A = np.loadtxt(os.path.join(exp_data_dir, 'A.csv'), delimiter=',')
exp_A_idx = np.argsort(exp_A[:, 0])
max_exp_0 = np.max(exp_A[:, 1])
scale_K = max_exp_0 / np.max(val_0)

print(f"{'Plane':<25} | {'Exp Peak (mm)':<14} | {'CFD Peak (mm)':<14} | {'Peak Error (%)':<15} | {'Rel L2 Error (%)':<16} | {'R2 Score'}")
print("-" * 105)

all_rel_l2 = []
all_peak_err = []

for name, fname, phi in planes:
    exp_data = np.loadtxt(os.path.join(exp_data_dir, fname), delimiter=',')
    idx = np.argsort(exp_data[:, 0])
    exp_ang = exp_data[idx, 0]
    exp_ero = exp_data[idx, 1]
    
    cfd_ang, cfd_ero = extract_and_smooth(ER_dnv, phi)
    cfd_scaled = cfd_ero * scale_K
    
    # Interpolate CFD to exp angles
    cfd_interp = np.interp(exp_ang, cfd_ang, cfd_scaled, left=0, right=0)
    
    # Errors
    peak_exp = np.max(exp_ero)
    peak_cfd = np.max(cfd_scaled)
    peak_err = np.abs(peak_cfd - peak_exp) / (peak_exp + 1e-12) * 100
    
    rel_l2 = np.linalg.norm(cfd_interp - exp_ero) / (np.linalg.norm(exp_ero) + 1e-12) * 100
    
    ss_res = np.sum((exp_ero - cfd_interp)**2)
    ss_tot = np.sum((exp_ero - np.mean(exp_ero))**2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    
    all_rel_l2.append(rel_l2)
    all_peak_err.append(peak_err)
    
    print(f"{name:<25} | {peak_exp:<14.3f} | {peak_cfd:<14.3f} | {peak_err:<14.1f}% | {rel_l2:<15.1f}% | {r2:.3f}")

print("-" * 105)
print(f"Average across all planes: Peak Error = {np.mean(all_peak_err):.1f}%, Rel L2 Error = {np.mean(all_rel_l2):.1f}%")
