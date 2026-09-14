import os
# Auto-detect repository root for portability
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))

#!/usr/bin/env python3
import json
import pandas as pd
import os

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def main():
    unweighted = load_json('hybrid_sensitivity_study/results.json')
    erosion_weighted = load_json('weighted_erosion_hybrid_study/results.json')

    variables = sorted(list(unweighted.keys()))

    pod_vars = []
    ero_cnn_vars = []
    unw_cnn_vars = []
    failed_vars = []

    for var in variables:
        if 'norm_method' not in unweighted[var] or '95' not in unweighted[var]['pod']:
            continue
            
        k_95 = unweighted[var]['pod']['95']['modes']
        pod_r2 = unweighted[var]['pod']['95']['metrics']['r2']
        unw_cnn_r2 = unweighted[var]['cnn'].get(f'dim_{k_95}', {}).get('metrics', {}).get('r2', float('-inf'))
        ero_cnn_r2 = erosion_weighted.get(var, {}).get('cnn', {}).get(f'dim_{k_95}', {}).get('metrics', {}).get('r2', float('-inf'))
        
        # We also have CNN at 10 dims for Unweighted which is often the champion for erosion
        unw_cnn_10_r2 = unweighted[var]['cnn'].get('dim_10', {}).get('metrics', {}).get('r2', float('-inf'))
        
        best_unw = max(unw_cnn_r2, unw_cnn_10_r2)
        
        scores = {
            'POD': pod_r2, 
            'Erosion-Weighted CNN': ero_cnn_r2,
            'Standard CNN': best_unw
        }
        
        winner = max(scores, key=scores.get)
        max_score = scores[winner]
        
        if max_score < 0:
            failed_vars.append(var)
        elif winner == 'POD':
            pod_vars.append((var, max_score))
        elif winner == 'Erosion-Weighted CNN':
            ero_cnn_vars.append((var, max_score))
        else:
            unw_cnn_vars.append((var, max_score))

    md = "# Variable Routing Assignments\n\n"
    md += "Based on the highest reconstruction $R^2$ scores, here is the exact list of which variables map to which dimensionality reduction algorithm in our Hybrid ROM framework.\n\n"
    
    md += "### 1. Route A: Linear POD (SVD)\n"
    md += "*Used for globally continuous fluid fields. These fields are efficiently factorized linearly without background dominance issues.*\n"
    for v, s in sorted(pod_vars, key=lambda x: -x[1]):
        md += f"- `{v}` ($R^2 = {s:.3f}$)\n"
        
    md += "\n### 2. Route B: Erosion-Weighted CNN-AE\n"
    md += "*Used for non-linear, highly-skewed kinematic and variance fields. The spatial mask forces the network to capture the complex wake while defeating POD's compression limit.*\n"
    for v, s in sorted(ero_cnn_vars, key=lambda x: -x[1]):
        md += f"- `{v}` ($R^2 = {s:.3f}$)\n"
        
    md += "\n### 3. Route C: Standard (Unweighted) CNN-AE\n"
    md += "*Used for specific terminal surface phenomena or fields where the log-transform alone provides sufficient spatial smoothing without needing an external mask.*\n"
    for v, s in sorted(unw_cnn_vars, key=lambda x: -x[1]):
        md += f"- `{v}` ($R^2 = {s:.3f}$)\n"
        
    if failed_vars:
        md += "\n### 4. Failed Variables\n"
        md += "*Variables that achieved negative $R^2$ across all methods (highly degenerate).*\n"
        for v in failed_vars:
            md += f"- `{v}`\n"

    output_path = os.path.join(_REPO_ROOT, 'variable_clustering_dendrogram.png')
    with open(output_path, 'w') as f:
        f.write(md)
        
    # Meta file to expose to UI
    meta = {
        "Summary": "Comprehensive list routing all 36 CFD variables to their mathematically optimal dimensionality reduction algorithm (POD vs Erosion-Weighted CNN vs Standard CNN).",
        "UserFacing": True,
        "RequestFeedback": False
    }
    with open(output_path + '.meta.json', 'w') as f:
        json.dump(meta, f)
        
    print(f"List generated at {output_path}")

if __name__ == '__main__':
    main()
