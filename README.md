# CAFA 6 Protein Function Prediction

Competition: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

## Overview

Multi-model ensemble approach for protein function prediction using transformer embeddings and graph neural networks.

**Target:** 0.32-0.36 F-max (Top 2-3)
**GPU Programming:** Custom CUDA kernels for sequence alignment and graph operations (Problem 2(b))

## Hardware Requirements

**Minimum:** 12GB VRAM GPU, 32GB RAM, 50GB storage
**Current Setup:** RTX 5070 Ti (12GB), i9 Ultra 2, 64GB RAM

## Quick Setup

```bash
# 1. Create environment
conda create -n cafa6 python=3.10 -y
conda activate cafa6

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify setup
cd cafa6_project/scripts
python test_setup.py
```

**Note:** For RTX 5070 Ti, use PyTorch nightly with CUDA 12.8 (already in requirements.txt)

## Project Plan

### Phase 1: Feature Engineering (Current) ✅
**Goal:** Generate 4,500-dim protein embeddings from 4 transformer models

**Models:**
- ESM-2 3B (1,280-dim) - Global context
- ESM-2 650M (1,280-dim) - Local patterns
- ProtBERT (1,024-dim) - Domain boundaries
- Ankh Large (1,536-dim) - Protein families

**GPU Optimizations Implemented:**
- CPU vs GPU benchmarking framework
- Comprehensive performance metrics & logging
- Kernel profiling (torch.profiler integration)
- Advanced optimizations (cuDNN, TF32, fp16, torch.compile)
- Real-time GPU monitoring & analysis

**Time:** ~14-15 hours for all models

### Phase 2: Model Architecture (Week 2-3)
- Asymmetric Loss for class imbalance
- Graph Neural Network for GO hierarchy
- Base learner design

### Phase 3: Training Strategy (Week 3-4)
- Hierarchical Bayesian inference on GO DAG
- Per-term threshold optimization
- Hyperparameter tuning

### Phase 4: Ensemble & Calibration (Week 4)
- Train 7+ calibrated models
- Meta-ensemble learning
- Probability calibration

### Phase 5: Custom CUDA Kernels (GPU Programming)
- Smith-Waterman sequence alignment (100x speedup target)
- GO graph propagation kernel (20x speedup target)
- K-mer counting for similarity search (50x speedup target)
- Benchmark custom kernels vs library implementations

### Phase 6: Final Submission
- Competition submission
- Performance analysis
- Documentation

## Current Status

### ✅ Completed (Phase 1 Pipeline)
- [x] Environment setup (PyTorch 2.10.0.dev+cu128, RTX 5070 Ti)
- [x] Enhanced embedding generation with CPU/GPU modes
- [x] Comprehensive performance metrics & JSON logging
- [x] CPU vs GPU benchmarking framework (`benchmark_cpu_gpu.py`)
- [x] Kernel profiling integration (`profile_embeddings.py`)
- [x] GPU optimizations (cuDNN, TF32, fp16, torch.compile)
- [x] Real-time monitoring tools (`gpu_monitor.py`, `analyze_gpu_logs.py`)
- [x] Configuration management (`config.yaml`)
- [x] Validation utilities (ID checks, NaN detection)
- [x] Competition data downloaded and verified (82K train, 224K test)

### 🔄 Ready to Execute
```bash
cd cafa6_project/scripts

# Generate embeddings (14-15 hours)
python generate_embeddings.py --device cuda

# Concatenate features
python concat_embeddings.py

# Verify results
python analyze_embeddings.py
```

### 📋 Next Steps
1. Execute Phase 1 embedding generation (~15 hours)
2. Begin Phase 2 model architecture design
3. Implement Asymmetric Loss
4. Design Graph Neural Network for GO hierarchy

## Key Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `generate_embeddings.py` | Generate embeddings (CPU/GPU) | `python generate_embeddings.py --device cuda` |
| `benchmark_cpu_gpu.py` | Measure CPU vs GPU speedup | `python benchmark_cpu_gpu.py --subset 1000` |
| `profile_embeddings.py` | Profile kernels for optimization | `python profile_embeddings.py --model esm2_650m` |
| `gpu_monitor.py` | Real-time GPU monitoring | `python gpu_monitor.py` |
| `analyze_gpu_logs.py` | Analyze GPU logs & visualize | `python analyze_gpu_logs.py` |
| `concat_embeddings.py` | Concatenate multi-model features | `python concat_embeddings.py` |
| `analyze_embeddings.py` | Validate embeddings | `python analyze_embeddings.py` |

## GPU Programming Features

### CPU vs GPU Benchmark 🔥
```bash
# Quick benchmark (5 minutes)
python benchmark_cpu_gpu.py --models esm2_650m --subset 1000

# Output: Speedup metrics, JSON results, comparison plots
```

### Kernel Profiling 🔬
```bash
# Profile embedding generation (2 minutes)
python profile_embeddings.py --model esm2_650m --num-batches 10

# Output: Chrome trace (chrome://tracing), optimization suggestions
```

### Expected Performance
| Model | CPU (prot/sec) | GPU (prot/sec) | Speedup |
|-------|---------------|---------------|---------|
| ESM-2 650M | ~30 | ~1000 | ~30x |
| ESM-2 3B | ~10 | ~400 | ~40x |
| ProtBERT | ~40 | ~800 | ~20x |
| Ankh Large | ~35 | ~750 | ~20x |

## Project Structure

```
cafa6_project/
├── data/                          # Competition data (82K train, 224K test)
├── embeddings/                    # Generated embeddings (4,500-dim features)
├── outputs/                       # Performance logs, benchmarks, plots
│   ├── logs/                      # Execution logs
│   ├── embedding_performance_report_*.json
│   ├── cpu_gpu_benchmark_results.json
│   └── profiler_trace_*.json
├── scripts/                       # Python scripts (8 files)
├── config.yaml                    # Configuration settings
└── requirements.txt               # Dependencies
```

## Configuration

Edit `config.yaml` to adjust settings:
```yaml
models:
  esm2_650m:
    batch_size: 16  # Adjust for your GPU

optimization:
  use_amp: true
  cudnn_benchmark: true
  tf32_matmul: true
```

## Troubleshooting

**Out of Memory:** Reduce batch sizes in `config.yaml`
**RTX 5070 Ti Warning:** Install PyTorch nightly with `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
**CUDA Not Available:** Check drivers with `nvidia-smi`

## Technical Approach

**Multi-Model Fusion:** 4 transformer models → 4,500-dim features
**Class Imbalance:** Asymmetric Loss (ASL)
**Hierarchy:** Bayesian inference on GO DAG
**Ensemble:** 7+ calibrated models
**GPU Optimization:** cuDNN, TF32, fp16, torch.compile
**Custom Kernels:** Smith-Waterman alignment, GO propagation (Phase 5)

## Resources

- **Competition:** https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- **ESM-2 Paper:** https://www.science.org/doi/10.1126/science.ade2574
- **Torch Profiler:** https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **CUDA Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

---

**Last Updated:** October 23, 2025
**Status:** Phase 1 Complete - Ready for Execution
**Hardware:** RTX 5070 Ti (12GB), PyTorch 2.10.0.dev+cu128
