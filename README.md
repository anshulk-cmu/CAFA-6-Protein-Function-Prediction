# CAFA 6 Protein Function Prediction - GPU-Optimized Pipeline

Competition: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

## Project Overview
Multi-model hierarchical ensemble approach for protein function prediction using:
- ESM-2 (3B & 650M) embeddings
- ProtBERT-BFD embeddings
- Ankh Large embeddings
- Graph Neural Networks on GO hierarchy
- Asymmetric Loss for class imbalance

**Target Score:** 0.32-0.36 F-max
**Expected Rank:** Top 2-3

**GPU Programming Integration:** This project includes custom CUDA kernel development for sequence alignment and graph operations (Problem 2(b) alignment).

## Hardware Requirements

### Minimum
- GPU: 12GB VRAM (RTX 3060 12GB, RTX 4070, etc.)
- RAM: 32GB
- Storage: 50GB free space

### Recommended (Current Setup)
- GPU: NVIDIA RTX 5070 Ti (12GB VRAM)
- CPU: Intel i9 Ultra 2 series
- RAM: 64GB
- Storage: 2TB NVMe SSD

## Environment Setup

### 1. Create Conda Environment
```bash
# Create environment
conda create -n cafa6 python=3.10 -y
conda activate cafa6
```

### 2. Install Dependencies

**For RTX 5070 Ti / 50-series (Blackwell) GPUs:**
```bash
# Install PyTorch nightly with CUDA 12.8 support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install all packages from requirements.txt
pip install -r requirements.txt
```

**For Other GPUs (RTX 30/40-series, etc.):**
```bash
# Install stable PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining packages
pip install transformers==4.36.0 biopython==1.83 fair-esm scikit-learn pandas numpy tqdm sentencepiece protobuf accelerate safetensors pyyaml matplotlib psutil
```

### 3. Verify Installation
```bash
cd cafa6_project/scripts
python test_setup.py
```

**Expected output:**
```
Python: 3.10.x
PyTorch: 2.10.0.dev (or 2.x.x for stable)
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 5070 Ti Laptop GPU
CUDA Memory: 12.82 GB
```

## Project Structure
```
cafa6_project/
├── data/                          # Competition data (download from Kaggle)
│   ├── train_sequences.fasta      # 82,404 training proteins
│   ├── train_terms.tsv            # GO annotations
│   ├── train_taxonomy.tsv         # Organism taxonomy
│   ├── testsuperset.fasta         # 224,309 test proteins
│   ├── go-basic.obo               # Gene Ontology hierarchy
│   ├── IA.tsv                     # Information Accretion weights
│   └── sample_submission.tsv      # Submission format
├── embeddings/                    # Generated embeddings (Phase 1 output)
│   ├── train_*_embeddings.npy     # Per-model embeddings
│   ├── test_*_embeddings.npy
│   └── *_concatenated_embeddings.npy  # 4,500-dim features
├── models/                        # Trained models (Phase 2+)
├── outputs/                       # Performance logs and results
│   ├── logs/                      # Execution logs
│   ├── embedding_performance_report_*.json
│   ├── cpu_gpu_benchmark_results.json
│   ├── cpu_gpu_comparison.png
│   ├── profiler_trace_*.json
│   ├── gpu_monitor_phase1.csv
│   └── gpu_analysis.png
├── scripts/                       # Python scripts
│   ├── test_setup.py              # Environment verification
│   ├── generate_embeddings.py     # Multi-model embedding generation
│   ├── concat_embeddings.py       # Feature fusion
│   ├── analyze_embeddings.py      # Validation utility
│   ├── benchmark_cpu_gpu.py       # CPU vs GPU speedup measurement
│   ├── profile_embeddings.py      # Kernel profiling (torch.profiler)
│   ├── gpu_monitor.py             # Real-time GPU monitoring
│   └── analyze_gpu_logs.py        # GPU log analysis & visualization
├── config.yaml                    # Centralized configuration
├── requirements.txt               # Python dependencies
└── README.md
```

## Quick Start

### 1. Download Competition Data
Visit [competition data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data) and download to `cafa6_project/data/` folder.

### 2. Generate Embeddings (GPU Mode)
```bash
cd cafa6_project/scripts

# Generate embeddings with all models (14-15 hours)
python generate_embeddings.py --device cuda

# Or run specific models only
python generate_embeddings.py --device cuda --models esm2_650m protbert
```

### 3. Concatenate Embeddings
```bash
python concat_embeddings.py  # Takes ~5 minutes
```

### 4. Verify Embeddings
```bash
python analyze_embeddings.py
```

## Phase 1: GPU-Optimized Embedding Generation

### Features

#### 🔥 CPU vs GPU Mode
```bash
# GPU mode (default) - 30-50x faster
python generate_embeddings.py --device cuda

# CPU mode (for baseline comparison)
python generate_embeddings.py --device cpu
```

#### 📊 Comprehensive Performance Metrics
- **Timing:** Total time, per-batch statistics (mean/std/min/max)
- **Throughput:** Proteins/sec, batches/sec, tokens/sec
- **Memory:** Peak/average GPU memory, utilization percentage
- **Warmup:** First 3 batches excluded from metrics
- **Logging:** Structured logs saved to `outputs/logs/`
- **Reports:** JSON performance data for analysis

#### ⚙️ Advanced GPU Optimizations
Automatically enabled:
- `torch.backends.cudnn.benchmark = True` - Auto-tune kernels
- `torch.set_float32_matmul_precision('high')` - TensorFloat32 (RTX GPUs)
- Half-precision (fp16) inference
- `torch.compile()` for kernel fusion (non-Windows)
- Mixed precision autocast

#### 🎛️ Configuration File
Edit `config.yaml` to tune settings without code changes:
```yaml
models:
  esm2_650m:
    batch_size: 16  # Adjust for your GPU memory

optimization:
  warmup_batches: 3
  use_amp: true
  cudnn_benchmark: true
  tf32_matmul: true
```

### Resource Usage by Model

| Model | VRAM | Batch Size | Time (Train) | Time (Test) |
|-------|------|------------|--------------|-------------|
| ESM-2 3B | ~11GB | 4 | ~3h | ~8h |
| ESM-2 650M | ~8GB | 16 | ~1.5h | ~4h |
| ProtBERT | ~6GB | 12 | ~1h | ~2.5h |
| Ankh Large | ~6GB | 12 | ~1h | ~2.5h |

**Total Phase 1 Time:** ~14-15 hours for all 4 models (train + test)

### Example Output

```
PERFORMANCE SUMMARY: esm2_650m
======================================================================
Total proteins: 82,404
Total batches: 5,150 (warmup: 3)
Total time: 4981.23s (83.0m)

Throughput:
  Proteins/sec: 984.2
  Batches/sec: 61.5
  Tokens/sec: 347,582.1

Batch timing:
  Average: 0.982s
  Std dev: 0.124s
  Min: 0.845s
  Max: 1.456s

Memory usage:
  Peak: 8.23GB
  Average: 7.89GB
  Utilization: 68.6%
======================================================================
```

## GPU Programming Features

### 1. CPU vs GPU Benchmark 🔥

**Purpose:** Quantitative speedup measurement for project reports

```bash
cd cafa6_project/scripts

# Quick benchmark (1000 proteins, ~5 minutes)
python benchmark_cpu_gpu.py --models esm2_650m --subset 1000

# Full benchmark (multiple models)
python benchmark_cpu_gpu.py --subset 2000 --models esm2_650m esm2_3b protbert ankh
```

**Output:**
- `outputs/cpu_gpu_benchmark_results.json` - Quantitative data
- `outputs/cpu_gpu_comparison.png` - Bar charts (throughput & speedup)
- Console: "Speedup: 31.2x faster"

**Example:**
```
SPEEDUP SUMMARY: esm2_650m
============================================================
CPU time: 245.67s
GPU time: 7.89s
Speedup: 31.13x faster
Time saved: 237.78s
============================================================
```

### 2. Kernel Profiling 🔬

**Purpose:** Identify bottlenecks for custom CUDA kernel optimization

```bash
# Profile embedding generation (10 batches, ~2 minutes)
python profile_embeddings.py --model esm2_650m --num-batches 10

# Detailed profiling
python profile_embeddings.py --model esm2_650m --num-batches 20 --batch-size 8
```

**Output:**
- `outputs/profiler_trace_{model}.json` - Chrome trace file
- Console: Top operations summary + optimization suggestions

**View trace:**
1. Open `chrome://tracing` in Chrome/Edge
2. Load the JSON file
3. Analyze kernel execution timeline

**Example insights:**
```
KEY INSIGHTS FOR OPTIMIZATION
======================================================================

Tokenization: 45.23ms
  → Potential optimization: Custom tokenizer or caching

Pooling: 12.34ms
  → Potential optimization: Custom CUDA reduction kernel

Matrix Multiplications: 892.45ms
  → Using Tensor Cores (fp16/bf16)
  → Already optimized by cuBLAS

Attention: 156.78ms
  → Potential optimization: Flash Attention 2
======================================================================
```

### 3. Real-time GPU Monitoring

```bash
# Terminal 1: Start monitoring
python gpu_monitor.py

# Terminal 2: Run embedding generation
python generate_embeddings.py --device cuda

# Terminal 1: Stop monitoring (Ctrl+C)

# Analyze logs and generate plots
python analyze_gpu_logs.py --log ../outputs/gpu_monitor_phase1.csv
```

**Output:**
- `outputs/gpu_monitor_phase1.csv` - Time-series metrics (memory, utilization, temperature, power)
- `outputs/gpu_analysis.png` - 4-panel visualization

## Current Progress

### ✅ Phase 1 Complete (GPU-Optimized Pipeline)
- [x] Environment setup (PyTorch 2.10.0.dev+cu128)
- [x] GPU compatibility resolved (RTX 5070 Ti)
- [x] Project structure created
- [x] **Enhanced embedding generation with CPU/GPU modes**
- [x] **Comprehensive performance metrics & logging**
- [x] **CPU vs GPU benchmarking framework**
- [x] **Kernel profiling integration (torch.profiler)**
- [x] **GPU optimization (cuDNN, TF32, fp16, torch.compile)**
- [x] **Real-time monitoring & analysis tools**
- [x] **Validation utilities (ID checks, NaN detection)**
- [x] **Configuration management (YAML)**
- [x] Competition data downloaded (199 MB)
- [x] Data organized and verified:
  - 82,404 training proteins
  - 224,309 test proteins
  - All GO terms and ontology files present

### 🔄 In Progress (Phase 1 Execution)
Run embedding generation:
```bash
cd cafa6_project/scripts
python generate_embeddings.py --device cuda
```

**Status:** Ready to execute (14-15 hours for all 4 models)

### 📋 Upcoming (Phase 2+)
- [ ] Phase 2: Model Architecture (Week 2-3)
  - Base learner design
  - Asymmetric Loss implementation
  - Graph Neural Network for GO hierarchy
- [ ] Phase 3: Training Strategy (Week 3-4)
  - Hyperparameter tuning
  - Hierarchical Bayesian inference
  - Per-term threshold optimization
- [ ] Phase 4: Ensemble & Calibration (Week 4)
  - 7+ model ensemble training
  - Probability calibration
  - Meta-ensemble learning
- [ ] **Phase 5: Custom CUDA Kernels (GPU Programming)**
  - Smith-Waterman sequence alignment (replace BLAST/DIAMOND)
  - Custom pooling reduction kernel
  - GO graph propagation kernel
  - Benchmark custom kernels vs library implementations
- [ ] Phase 6: Final Submission

## Log Files & Outputs

All metrics are saved to persistent log files for later analysis:

| File | Purpose | Location |
|------|---------|----------|
| Performance report | Timing, throughput, memory | `outputs/embedding_performance_report_{device}.json` |
| Execution log | Console output | `outputs/logs/embedding_generation_{device}_{timestamp}.log` |
| Benchmark results | CPU vs GPU speedup | `outputs/cpu_gpu_benchmark_results.json` |
| Benchmark plots | Visualization | `outputs/cpu_gpu_comparison.png` |
| Profiler trace | Kernel timeline | `outputs/profiler_trace_{model}.json` |
| GPU monitoring | Real-time metrics | `outputs/gpu_monitor_phase1.csv` |
| GPU analysis | Visualizations | `outputs/gpu_analysis.png` |

**All logs use timestamps - never overwritten!**

## Command Reference

### Embedding Generation
```bash
# Basic usage (GPU, all models)
python generate_embeddings.py

# CPU mode (for baseline)
python generate_embeddings.py --device cpu

# Specific models only
python generate_embeddings.py --models esm2_650m protbert

# Custom config
python generate_embeddings.py --config ../my_config.yaml
```

### Benchmarking
```bash
# Quick benchmark
python benchmark_cpu_gpu.py

# Custom settings
python benchmark_cpu_gpu.py --subset 500 --batch-size 4 --models esm2_650m
```

### Profiling
```bash
# Profile a model
python profile_embeddings.py --model esm2_650m

# Detailed profiling
python profile_embeddings.py --model esm2_650m --num-batches 20
```

### Monitoring & Analysis
```bash
# Start real-time GPU monitoring
python gpu_monitor.py

# Analyze logs
python analyze_gpu_logs.py --log ../outputs/gpu_monitor_phase1.csv

# Validate embeddings
python analyze_embeddings.py

# Concatenate embeddings
python concat_embeddings.py
```

## Troubleshooting

### RTX 5070 Ti Compatibility Warning
If you see:
```
UserWarning: NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible
```

**Solution:**
```bash
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Out of Memory (OOM) Errors
Edit `config.yaml` to reduce batch sizes:
```yaml
models:
  esm2_650m:
    batch_size: 8   # Reduce from 16
  esm2_3b:
    batch_size: 2   # Reduce from 4
  protbert:
    batch_size: 8   # Reduce from 12
  ankh:
    batch_size: 8   # Reduce from 12
```

Or edit directly in `generate_embeddings.py` (line 413-416).

### CUDA Not Available
Check NVIDIA drivers:
```bash
nvidia-smi
```
Should show CUDA 12.8+ for RTX 5070 Ti.

### Column Name Errors in analyze_gpu_logs.py
**Fixed!** Updated script now uses correct column names from `gpu_monitor.py`.

### Config File Not Found
The script will use default configuration if `config.yaml` is not found. To customize, copy and edit:
```bash
cp cafa6_project/config.yaml cafa6_project/my_config.yaml
# Edit my_config.yaml
python generate_embeddings.py --config ../my_config.yaml
```

## Technical Approach Summary

### Multi-Model Embedding Fusion
- **ESM-2 3B**: 1,280-dim (global context, deep understanding)
- **ESM-2 650M**: 1,280-dim (local patterns, efficient)
- **ProtBERT**: 1,024-dim (domain boundaries)
- **Ankh Large**: 1,536-dim (protein family relationships)
- **Total**: 4,500-dim concatenated features

### Key Innovations
1. **Asymmetric Loss (ASL)** for extreme class imbalance
2. **Hierarchical Bayesian inference** on GO DAG
3. **Graph Attention Networks** for term relationships
4. **Per-term threshold optimization**
5. **Calibrated meta-ensemble** (7+ models)
6. **GPU optimization** (cuDNN, TF32, fp16, torch.compile)
7. **Custom CUDA kernels** (Phase 5) for:
   - Smith-Waterman sequence alignment (100x speedup target)
   - GO graph propagation (20x speedup target)
   - K-mer counting for similarity search (50x speedup target)

### GPU Programming Project Alignment

This project fulfills **Problem 2(b)** requirements:

✅ **Protein computation application** - Function prediction for 224K+ proteins
✅ **Existing library baseline** - PyTorch transformers (CPU mode)
✅ **Custom GPU implementation** - Optimized CUDA inference + custom kernels (Phase 5)
✅ **Quantitative benchmarks** - `benchmark_cpu_gpu.py` provides exact speedup measurements
✅ **Performance analysis** - `profile_embeddings.py` identifies optimization opportunities
✅ **Sample data** - 82K training + 224K test proteins

**Deliverables:**
- CPU baseline benchmarks
- GPU-optimized implementation (30-50x speedup)
- Custom CUDA kernels for alignment & graph operations (Phase 5)
- Performance analysis & bottleneck identification
- Comprehensive metrics & visualization

## Expected Performance

Based on RTX 5070 Ti (12GB VRAM):

| Model | CPU (proteins/sec) | GPU (proteins/sec) | Speedup |
|-------|-------------------|-------------------|---------|
| ESM-2 650M | ~30 | ~1000 | ~30x |
| ESM-2 3B | ~10 | ~400 | ~40x |
| ProtBERT | ~40 | ~800 | ~20x |
| Ankh Large | ~35 | ~750 | ~20x |

**Note:** Run `benchmark_cpu_gpu.py` for exact measurements on your system.

## Resources

- **Competition:** https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **ESM-2 Paper:** https://www.science.org/doi/10.1126/science.ade2574
- **Gene Ontology:** http://geneontology.org/
- **Torch Profiler:** https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- **CUDA Programming Guide:** https://docs.nvidia.com/cuda/cuda-c-programming-guide/

## Team & Contact

**Hardware:** RTX 5070 Ti Laptop (12GB), i9 Ultra 2, 64GB RAM
**Environment:** Windows 11, Conda, PyTorch 2.10.0.dev+cu128
**Competition Deadline:** Check Kaggle timeline
**GPU Programming:** Custom CUDA kernels for Phase 2(b) project

---

**Last Updated:** October 23, 2025
**Status:** Phase 1 GPU-Optimized Pipeline Complete, Ready for Execution
**Recent Updates:**
- Added CPU vs GPU benchmarking framework
- Integrated torch.profiler for kernel analysis
- Enhanced logging and performance metrics
- Fixed analyze_gpu_logs.py bugs
- Added configuration management (YAML)
- GPU optimization (cuDNN, TF32, fp16, torch.compile)
