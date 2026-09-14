#!/usr/bin/env python3
"""
Block Proper Orthogonal Decomposition (Block-POD) Pipeline
-----------------------------------------------------------
This script implements a unified, memory-optimized end-to-end pipeline to:
1. Load spatial variables from simulations in `trimmed_data_csv`.
2. Group similar variables together using Hierarchical Correlation Clustering (or predefined groups).
3. Interpolate variables onto a master grid (with periodic boundary padding in theta).
4. Perform Block-POD on each group of variables.
5. Save POD modes, coefficients, mean fields, and parameter metadata.
6. Generate visualizations for dendrograms, singular value decay, and energy convergence.
"""

import os
import sys
import time
import argparse
import json
import logging
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from joblib import Parallel, delayed
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# Constant Definitions (Physical Parameter Space)
# ==========================================
PARTICLE_SIZES_micro = [50, 100, 150, 200, 300, 500]
DENSITY = [2000, 5000, 10000, 15000, 30000]
VELOCITY = [5, 10, 15, 18, 20]
BEND_RATIO = [1.5, 2.0, 5.0]

# Predefined block architecture (from previous manual analysis)
PREDEFINED_BLOCKS = {
    '1': ['frequency_norm', 'f_alpha_norm'],
    '2': ['f_v_norm'],
    '3': ['velocity_variance_norm', 'v2_mean_norm', 'sin_mean'],
    '4': ['v_mean_norm', 'v2_cos2_norm', 'v25_mean_norm', 'v3_mean_norm'],
    '5': ['dp_mag_norm', 'dke_norm'],
    '6': [
        'alpha_variance', 'v2_sin_norm', 'v2_sin2_norm', 'v2_sin3_norm', 'v2_sincos_norm',
        'v2_alpha_norm', 'v2_alpha2_norm', 'v25_sin_norm', 'v25_sin2_norm', 'v25_sin3_norm',
        'v25_sincos_norm', 'v25_alpha_norm', 'v25_alpha2_norm', 'v3_sin_norm', 'v3_sin2_norm',
        'v3_sin3_norm', 'v3_sincos_norm', 'v3_alpha_norm', 'v3_alpha2_norm', 'sin3_mean',
        'sincos_mean', 'alpha_mean', 'alpha2_mean', 'oka_erosion_norm'
    ],
    '7': ['sin2_mean'],
    '8': ['cos2_mean']
}

# ==========================================
# Progress Bar Context Manager for Joblib
# ==========================================
@contextlib.contextmanager

def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report progress into tqdm."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback

# ==========================================
# Helper Functions
# ==========================================
def decode_filename(filename):
    """Extracts physical parameters from the 4-digit filename."""
    base_name = os.path.basename(filename).replace('.csv', '')
    if len(base_name) != 4 or not base_name.isdigit():
        return None
    i, j, k, l = [int(char) - 1 for char in base_name]
    return [BEND_RATIO[i], VELOCITY[j], DENSITY[k], PARTICLE_SIZES_micro[l]]

def build_interpolators(df_padded):
    """Builds Linear and Nearest Neighbor spatial interpolators."""
    points = np.column_stack((df_padded['s_normalized'].values, df_padded['theta'].values))
    dummy_vals = np.zeros(len(points))
    lin_interp = LinearNDInterpolator(points, dummy_vals)
    near_interp = NearestNDInterpolator(points, dummy_vals)
    return lin_interp, near_interp

# ==========================================
# Clustering & Grouping Pipeline Step
# ==========================================
def group_variables_by_similarity(input_dir, sample_ratio=0.1, max_k_clusters=12, output_dir='pod_artifacts'):
    """
    Computes variable correlation matrix from a sampled subset of simulation files,
    performs hierarchical clustering (Average Linkage), and automatically finds
    the optimal cluster grouping using the Silhouette Score.
    """
    logger.info("--- Stage 1: Variable Similarity Clustering ---")
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not all_files:
        raise ValueError(f"No CSV files found in {input_dir}")
        
    # Sample files to construct correlation matrix efficiently
    sample_size = max(1, int(len(all_files) * sample_ratio))
    sampled_files = np.random.choice(all_files, sample_size, replace=False)
    logger.info(f"Sampling {sample_size} files (out of {len(all_files)}) to compute correlation...")
    
    # Load and combine data
    df_list = []
    for file in sampled_files:
        df = pd.read_csv(os.path.join(input_dir, file))
        df_list.append(df)
    master_df = pd.concat(df_list, ignore_index=True)
    
    # Isolate physics variables
    coords = ['s', 'r', 'theta', 's_normalized']
    physics_vars = [c for c in master_df.columns if c not in coords]
    df_physics = master_df[physics_vars]
    
    # Compute Correlation and Distance Matrix
    logger.info("Computing correlation matrix...")
    corr_matrix = df_physics.corr().fillna(0)

    for var, label in zip(physics_vars, optimal_labels):
        blocks[str(label)].append(var)
        
    # Save the optimal groupings as JSON configuration
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, 'variable_blocks_config.json')
    with open(config_path, 'w') as f:
        json.dump(blocks, f, indent=4)
    logger.info(f"Saved variable blocks configuration to {config_path}")
    
    # Plot & Save Dendrogram
    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=physics_vars, leaf_rotation=90, leaf_font_size=10, 
               color_threshold=Z[-(best_n-1), 2] if best_n > 1 else None) 
    plt.title(f"Hierarchical Clustering of Variables (Optimal Clusters: {best_n})")
    plt.ylabel("Distance (1 - |Correlation|)")
    plt.tight_layout()
    dendrogram_path = os.path.join(output_dir, 'optimal_dendrogram.png')
    plt.savefig(dendrogram_path, dpi=300)
    plt.close()
    logger.info(f"Saved optimal dendrogram visualization to {dendrogram_path}")
    
    return blocks

# ==========================================
# Assembly Stage
# ==========================================
def process_single_file_joblib(file_name, input_dir, blocks, s_grid, theta_grid):
    """Worker function for parallel data extraction and interpolation."""
    params = decode_filename(file_name)
    if params is None:
        return None
        
    df = pd.read_csv(os.path.join(input_dir, file_name))
    
    # Periodic Padding to handle boundary conditions in theta
    buffer_pos = df[df['theta'] < -np.pi/2].copy()
    buffer_pos['theta'] += 2*np.pi
    buffer_neg = df[df['theta'] > np.pi/2].copy()
    buffer_neg['theta'] -= 2*np.pi
    df_padded = pd.concat([df, buffer_pos, buffer_neg], ignore_index=True)
    
    # Build geometry on the fly (isolated per process)
    lin_interp, near_interp = build_interpolators(df_padded)
    
    file_results = {}
    for key, variables in blocks.items():
        block_vectors = []
        for var in variables:
            # Handle case where dynamically clustered variable is missing from a file
            if var not in df_padded.columns:
                logger.warning(f"Variable {var} missing from file {file_name}. Padding with zeros.")
                values = np.zeros(len(df_padded))
            else:
                values = df_padded[var].values
            
            # Inject variable values into the pre-built geometry interpolators
            lin_interp.values = values[:, np.newaxis]
            near_interp.values = values[:, np.newaxis]
            
            # Evaluate grid and squeeze
            interp = lin_interp(s_grid, theta_grid).squeeze()
            
            # Handle boundary NaNs with nearest neighbor
            if np.isnan(interp).any():
                interp_near = near_interp(s_grid, theta_grid).squeeze()
                interp[np.isnan(interp)] = interp_near[np.isnan(interp)]
            
            # Z-score standardization
            interp_scaled = (interp - np.mean(interp)) / (np.std(interp) + 1e-8)
            
            # Memory Fix: Downcast to float32 before flattening
            block_vectors.append(interp_scaled.astype(np.float32).flatten())
            
        file_results[key] = np.hstack(block_vectors)
        
    return params, file_results

def run_snapshot_assembly(input_dir, blocks, s_grid, theta_grid, n_jobs=-1):
    """Executes parallel spatial interpolation and snapshot assembly."""
    logger.info("--- Stage 2: Parallel Assembly of Snapshot Matrices ---")
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    all_files.sort(key=lambda x: int(x.replace('.csv', '')) if x.replace('.csv', '').isdigit() else 9999)
    
    start_time = time.time()
    logger.info(f"Assembling data from {len(all_files)} files using {n_jobs if n_jobs != -1 else 'all'} cores...")
    
    # Execute Joblib with Loky backend and progress tracking
    with tqdm_joblib(tqdm(desc="Processing Simulations", total=len(all_files))):
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(process_single_file_joblib)(file, input_dir, blocks, s_grid, theta_grid) for file in all_files
        )
    
    # Unpack parameters and file data
    X_parameters = []
    S_matrices_lists = {key: [] for key in blocks.keys()}
    
    for res in results:
        if res is None:
            continue
        params, file_data = res
        X_parameters.append(params)
        for key in blocks.keys():
            S_matrices_lists[key].append(file_data[key])
            
    # Storing combined snapshot matrices (Optimized stacking with memory cleanup)
    logger.info("Stacking snapshot matrices into RAM with garbage collection...")
    S_matrices = {}
    for key in list(blocks.keys()):
        S_matrices[key] = np.column_stack(S_matrices_lists[key])
        # Free memory of lists immediately
        del S_matrices_lists[key]
        
    X_matrix = np.array(X_parameters)
    logger.info(f"Assembly completed successfully in {time.time() - start_time:.2f} seconds.")
    return S_matrices, X_matrix

# ==========================================
# Proper Orthogonal Decomposition (POD) Stage
# ==========================================
def extract_pod_modes(S_matrix, block_name, variance_threshold=0.95, output_dir='pod_artifacts'):
    """Performs Mean Centering, SVD, and Energy Truncation on a snapshot matrix."""
    logger.info(f"Performing POD on Block {block_name} (Snapshot shape: {S_matrix.shape})...")
    
    # Node-Specific Mean Centering
    mean_field = np.mean(S_matrix, axis=1, keepdims=True)
    S_fluctuating = S_matrix - mean_field
    
    # SVD (Economy Mode: full_matrices=False)
    U, Sigma, Vt = np.linalg.svd(S_fluctuating, full_matrices=False)
    
    # Energy Truncation analysis
    eigenvalues = Sigma ** 2
    total_energy = np.sum(eigenvalues)
    cumulative_energy = np.cumsum(eigenvalues) / total_energy
    
    # Number of modes to capture threshold variance
    k = np.argmax(cumulative_energy >= variance_threshold) + 1
    logger.info(f"Block {block_name}: Retained {k} modes to capture {variance_threshold*100}% variance.")
    
    # Truncate and Calculate Coefficients (C = Sigma * V^T)
    Phi_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    Vt_k = Vt[:k, :]
    C_k = np.dot(Sigma_k, Vt_k)
    
    # Save artifacts
    np.save(os.path.join(output_dir, f'Phi_Block_{block_name}.npy'), Phi_k)
    np.save(os.path.join(output_dir, f'C_Block_{block_name}.npy'), C_k)
    np.save(os.path.join(output_dir, f'Mean_Block_{block_name}.npy'), mean_field)
    
    # Plot Energy Convergence
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.semilogy(Sigma[:100], 'o-', color='crimson')
    plt.title(f"Block {block_name} - Singular Value Decay")
    plt.xlabel("Mode Index")
    plt.ylabel("Singular Value (Log Scale)")
    
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_energy[:100], 'o-', color='royalblue')
    plt.axhline(y=variance_threshold, color='green', linestyle='--', label=f'{variance_threshold*100}% Threshold')
    plt.axvline(x=k-1, color='orange', linestyle='--', label=f'{k} Modes')
    plt.title(f"Block {block_name} - Energy Convergence")
    plt.xlabel("Number of Modes")
    plt.ylabel("Cumulative Energy")
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'energy_convergence_Block_{block_name}.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    return k, cumulative_energy

# ==========================================
# Main Execution Entry
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="End-to-End Block POD Pipeline")
    parser.add_argument('--input-dir', type=str, default='trimmed_data_csv',
                        help='Directory containing the trimmed simulation CSV files.')
    parser.add_argument('--output-dir', type=str, default='pod_artifacts',
                        help='Directory where POD artifacts and plots will be stored.')
    parser.add_argument('--variance-threshold', type=float, default=0.95,
                        help='Energy threshold (0 to 1) for POD modes truncation (default 0.95).')
    parser.add_argument('--grouping-method', type=str, choices=['clustering', 'predefined'], default='predefined',
                        help='Variable grouping strategy: "clustering" (dynamic correlation clustering) or "predefined" (notebook standard).')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                        help='Subsampling ratio of files to use during similarity clustering (default 0.1).')
    parser.add_argument('--max-clusters', type=int, default=12,
                        help='Maximum number of clusters allowed during dynamic clustering optimization.')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs to use during snapshot assembly (default -1 uses all cores).')
    parser.add_argument('--grid-s', type=int, default=582,
                        help='Grid resolution along s-axis (default 582).')
    parser.add_argument('--grid-theta', type=int, default=221,
                        help='Grid resolution along theta-axis (default 221).')
    args = parser.parse_args()

    # Paths and folders
    os.makedirs(args.output_dir, exist_ok=True)
    start_pipeline_time = time.time()
    
    logger.info("Starting Block-POD Pipeline...")
    logger.info(f"Settings: input_dir={args.input_dir}, output_dir={args.output_dir}, threshold={args.variance_threshold}, method={args.grouping_method}")
    
    # Stage 1: Variable grouping
    if args.grouping_method == 'predefined':
        logger.info("Using predefined variable blocks architecture...")
        blocks = PREDEFINED_BLOCKS
        # Save predefined config as JSON for reference
        with open(os.path.join(args.output_dir, 'variable_blocks_config.json'), 'w') as f:
            json.dump(blocks, f, indent=4)
    else:
        logger.info("Determining variable blocks via hierarchical correlation clustering...")
        blocks = group_variables_by_similarity(
            input_dir=args.input_dir,
            sample_ratio=args.sample_ratio,
            max_k_clusters=args.max_clusters,
            output_dir=args.output_dir
        )
        
    # Print out blocks summary
    logger.info("Variable Blocks defined:")
    for b_id, variables in blocks.items():
        logger.info(f"  Block {b_id} (n={len(variables)}): {variables}")

    # Set up master grid
    logger.info(f"Setting up Master Spatial Grid: {args.grid_s} x {args.grid_theta} (total {args.grid_s * args.grid_theta} nodes)...")
    s_1d = np.linspace(0.0, 2.0, args.grid_s)
    theta_1d = np.linspace(-np.pi, np.pi, args.grid_theta)
    s_grid, theta_grid = np.meshgrid(s_1d, theta_1d, indexing='ij')

    # Stage 2: Assembly
    S_matrices, X_matrix = run_snapshot_assembly(
        input_dir=args.input_dir,
        blocks=blocks,
        s_grid=s_grid,
        theta_grid=theta_grid,
        n_jobs=args.n_jobs
    )
    
    # Save parameter space coordinates matrix
    x_matrix_path = os.path.join(args.output_dir, 'X_Parameters.npy')
    np.save(x_matrix_path, X_matrix)
    logger.info(f"Saved Input Matrix X_Parameters.npy to {x_matrix_path} (Shape: {X_matrix.shape})")

    # Stage 3: POD SVD Extraction
    logger.info("--- Stage 3: Proper Orthogonal Decomposition ---")
    block_results = {}
    for key in blocks.keys():
        k_retained, cum_energy = extract_pod_modes(
            S_matrix=S_matrices[key],
            block_name=key,
            variance_threshold=args.variance_threshold,
            output_dir=args.output_dir
        )
        block_results[key] = {
            'modes_retained': k_retained,
            'num_variables': len(blocks[key]),
            'cumulative_energy_95': float(cum_energy[min(k_retained-1, len(cum_energy)-1)]),
            'variables': blocks[key]
        }
        
    # Free memory
    del S_matrices
    
    # Create Pipeline Summary Report
    elapsed_time = time.time() - start_pipeline_time
    logger.info(f"All stages completed successfully in {elapsed_time/60:.2f} minutes.")
    
    report_path = os.path.join(args.output_dir, 'pipeline_report.md')
    with open(report_path, 'w') as r:
        r.write("# Block POD Pipeline Execution Summary\n\n")
        r.write(f"- **Execution Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        r.write(f"- **Total Elapsed Time**: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)\n")
        r.write(f"- **Input Directory**: `{args.input_dir}`\n")
        r.write(f"- **Output Directory**: `{args.output_dir}`\n")
        r.write(f"- **Grouping Strategy**: `{args.grouping_method}`\n")
        r.write(f"- **Variance Threshold**: `{args.variance_threshold * 100}%`\n")
        r.write(f"- **Master Grid Size**: {args.grid_s} x {args.grid_theta} ({args.grid_s * args.grid_theta} nodes)\n")
        r.write(f"- **Metadata Matrix**: `X_Parameters.npy` (Shape: {X_matrix.shape})\n\n")
        
        r.write("## Variable Block Groupings & Retained Modes\n\n")
        r.write("| Block | Variable Count | Modes Retained | Captured Variance | Variables List |\n")
        r.write("| :---: | :---: | :---: | :---: | :--- |\n")
        for key, res in block_results.items():
            vars_str = ", ".join([f"`{v}`" for v in res['variables']])
            r.write(f"| **{key}** | {res['num_variables']} | {res['modes_retained']} | {res['cumulative_energy_95']*100:.2f}% | {vars_str} |\n")
            
        r.write("\n\n*Note: POD mathematical artifacts (`Phi_Block_X.npy`, `C_Block_X.npy`, `Mean_Block_X.npy`) and convergence plots are stored under the output directory.*")
        
    logger.info(f"Saved pipeline execution report to {report_path}")

if __name__ == "__main__":
    main()
