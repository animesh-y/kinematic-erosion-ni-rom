# Multi-Model Non-Intrusive Reduced-Order Framework for Parametric Erosion Prediction

 [![arXiv](https://img.shields.io/badge/arXiv-2609.09997-b31b1b.svg)](https://arxiv.org/abs/2609.09997)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![GPyTorch](https://img.shields.io/badge/GPyTorch-1.11+-red.svg)](https://gpytorch.ai/)

This repository contains the official implementation of the paper:
> **"A Multi-Model Non-Intrusive Reduced-Order Framework for Parametric Erosion Prediction via Kinematic Cross-Moment Compression"**  
> *Animesh Yadav, Rajesh K. Shukla* (Department of Mechanical Engineering, Thapar Institute of Engineering and Technology).

---

## 📌 Key Highlights

- **Kinematic Cross-Moment Abstraction:** Bypasses "model-locked" limitations by compressing **23 Eulerian boundary velocity-angle cross-moments** ($\mathbb{E}[V_p^u \sin^vlpha_p \cos^wlpha_p]$) instead of pre-computed scalar wear fields.
- **Multi-Model Post-Hoc Evaluation:** A single trained kinematic surrogate evaluates **Finnie (exact), Oka, McLaury, and Arabnejad (approximated)** wear equations in milliseconds without surrogate retraining.
- **Composite Reduced-Order Modeling:** Integrates **Linear W-POD and Mode-1 Tensor Unfolding SVD** (for coupled velocity-angle moments) with **SemiCircular Residual CNN Autoencoders** (for convective transport and uncoupled flux fields).
- **Sub-2 ms Online Inference:** Accelerates full-field 2D/3D spatial wear prediction from **6–8 CPU hours (CFD-DPM)** down to **~2 ms on a single GPU** with $R^2 > 0.99$ on primary kinematic fields.

---

## 🗂 Repository Structure

```
├── data/
│   └── experimental_solnordal/        # Benchmark CMM wear depth profiles from Solnordal et al. (2015)
│       ├── A.csv (0° Centerline)
│       ├── B.csv (9°) ... K.csv (81° Sidewall Crown)
│
├── manuscript/                        # Complete LaTeX paper source and high-resolution figures
│   ├── main.tex
│   ├── references.bib
│   └── figures/
│
├── src/
│   ├── cfd_udf/                       # Fluent UDFs for per-particle kinematic moment accumulation
│   │   ├── moment_accumulation_udf.c  # Jensen-inequality safe cross-moment accumulation UDF
│   │   ├── rebound_grant_tabakoff.c   # Grant & Tabakoff (1977) restitution polynomials
│   │   └── fluent_batch_automation.py # Batch execution controller
│   │
│   ├── preprocessing/                 # Surface unwrapping, power-law scaling, Ward clustering
│   │   ├── curvilinear_grid_mapping.ipynb # 3D Cartesian -> Invariant 2D (512x256) mapping
│   │   ├── signed_powerlaw_transform.py   # Signed square-root (p=0.5) variance stabilization
│   │   └── ward_clustering.py             # Pearson correlation grouping into 5 physical blocks
│   │
│   ├── rom_compression/              # POD, Mode-1 SVD, and Deep SemiCircular CNN-AE
│   │   ├── pod_hosvd_pipeline.py      # Snapshot POD, W-POD, and Mode-1 Tensor Unfolding SVD
│   │   ├── semicircular_cnn_ae.py     # PyTorch CNN-AE with circular azimuthal padding & FW-MSE
│   │   ├── composite_rom_pipeline.py  # Unified hybrid dimensionality reduction architecture
│   │   └── activation_ablation.py     # GELU vs LeakyReLU vs Tanh benchmark
│   │
│   ├── surrogates/                    # Gaussian Process Regression & Deep ANN
│   │   ├── gpr_gpu_surrogate.py       # GPU GPyTorch ARD Matérn-5/2 surrogate pipeline
│   │   ├── ann_vs_gpr_benchmark.py   # 5-layer deep residual MLP vs GPR comparison
│   │   └── latent_dimension_tuning.py # 5-fold CV hyperparameter optimization
│   │
│   ├── evaluation/                    # Post-hoc reconstruction, sensitivity, and error audits
│   │   ├── evaluate_locked_testset.py # Full evaluation on 75 locked test cases (Oka/Finnie/McLaury/Arabnejad)
│   │   ├── evaluate_withheld_functional.py # Generalization test on non-polynomial synthetic model
│   │   ├── moment_sensitivity_analysis.py  # Analytical & numerical elasticity sensitivity (Sk = 0)
│   │   └── audit_worst_cases.py       # Test set error auditor (ballistic vs near-tracer)
│   │
│   └── validation/                    # Experimental validation benchmarks
│       ├── solnordal_profile_validation.py # Quantitative multi-planar L2 error vs Solnordal (2015)
│       └── solnordal_multiplanar_overlay.py# Multi-curve depth overlay generator
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md
```

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/username/erosion-pipe-bend-rom.git
cd erosion-pipe-bend-rom
pip install -r requirements.txt
```

### 2. Run GPR Parametric Surrogate Training & Inference
Train the anisotropic Gaussian Process surrogate mapping 4 dimensionless $\Pi$-groups to latent modal coordinates:
```bash
python src/surrogates/gpr_gpu_surrogate.py
```

### 3. Evaluate Multi-Model Reconstruction on Held-Out Test Set
Evaluate full 2D spatial wear fields for Finnie, Oka, McLaury, and Arabnejad models across the 75 locked test cases:
```bash
python src/evaluation/evaluate_locked_testset.py
```

### 4. Run Solnordal et al. (2015) Experimental Validation
Compute absolute penetration depth ($\mathrm{mm}$) and profile errors against experimental data:
```bash
python src/validation/solnordal_profile_validation.py
```

### 5. Run Moment Sensitivity Analysis
Verify that uncoupled moments (such as `v2_mean_norm`) have identically zero sensitivity ($S_k = 0$) across empirical models:
```bash
python src/evaluation/moment_sensitivity_analysis.py
```

---

## 📊 Summary of Benchmark Results

| Model / Architecture | Global $R^2$ | Focal Crater $	ext{FW-}R^2$ | Peak Depth Error ($\mathcal{E}_{	ext{peak}}$) | Inference Time |
| :--- | :---: | :---: | :---: | :---: |
| **Finnie (1960)** | **$0.9729$** | **$0.9686$** | **$3.12\%$** | **$2.0\,\mathrm{ms}$** |
| **Oka et al. (2005)** | **$0.9684$** | **$0.9632$** | **$3.48\%$** | **$2.0\,\mathrm{ms}$** |
| **Arabnejad et al. (2015)** | **$0.9598$** | **$0.9543$** | **$3.84\%$** | **$2.0\,\mathrm{ms}$** |
| **McLaury / Tulsa (1996)** | **$0.9571$** | **$0.9545$** | **$4.05\%$** | **$2.0\,\mathrm{ms}$** |
| **Withheld Synthetic Model** | **$0.9264$** | **$0.9229$** | **$13.54\%$** | **$2.0\,\mathrm{ms}$** |

---

## 📖 Citation

If you use this codebase or methodology in your research, please cite:
```bibtex
@misc{yadav2026multimodel,
  title={A Multi-Model Non-Intrusive Reduced-Order Framework for Parametric Erosion Prediction via Kinematic Cross-Moment Compression},
  author={Animesh Yadav and Rajesh K. Shukla},
  year={2026},
  eprint={2609.09997},
  archivePrefix={arXiv},
  primaryClass={physics.flu-dyn}
}
```
