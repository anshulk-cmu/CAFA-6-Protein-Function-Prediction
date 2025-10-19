# CAFA 6 Protein Function Prediction - Phase 1 Setup

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
```powershell
# Create environment
conda create -n cafa6 python=3.10 -y
conda activate cafa6
```

### 2. Install Dependencies

**For RTX 5070 Ti / 50-series (Blackwell) GPUs:**
```powershell
# Install PyTorch nightly with CUDA 12.8 support
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install remaining packages
pip install transformers==4.36.0 biopython==1.83 fair-esm scikit-learn pandas numpy tqdm sentencepiece protobuf accelerate safetensors
```

**For Other GPUs (RTX 30/40-series, etc.):**
```powershell
# Install stable PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install remaining packages
pip install transformers==4.36.0 biopython==1.83 fair-esm scikit-learn pandas numpy tqdm sentencepiece protobuf accelerate safetensors
```

### 3. Verify Installation
```powershell
cd scripts
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
│   ├── train_sequences.fasta
│   ├── train_terms.tsv
│   ├── train_taxonomy.tsv
│   ├── testsuperset.fasta
│   ├── go-basic.obo
│   └── IA.tsv
├── embeddings/                    # Generated embeddings (Phase 1 output)
├── models/                        # Trained models (Phase 2+)
├── scripts/                       # Python scripts
│   ├── test_setup.py
│   ├── generate_embeddings.py
│   └── concat_embeddings.py
├── outputs/                       # Final submissions
├── requirements.txt
└── README.md
```

## Current Progress

### ✅ Completed
- [x] Environment setup (conda + PyTorch 2.10.0.dev+cu128)
- [x] GPU compatibility resolved (RTX 5070 Ti working perfectly)
- [x] Project structure created
- [x] Scripts written and tested
- [x] Kaggle API configured
- [x] Competition data downloaded (199 MB)
- [x] Data organized and verified:
  - 82,404 training proteins
  - 224,309 test proteins
  - All GO terms and ontology files present

### 🔄 In Progress (Phase 1: Feature Engineering)
- [x] ESM-2 650M Training Set: **17% complete** (871/5,151 batches)
  - Started: Oct 19, 2025 12:50 AM
  - ETA: ~1h 37m remaining for train
  - GPU: 100% utilization, 60°C
- [ ] ESM-2 650M Test Set: ~2.5h (queued)
- [ ] ESM-2 3B: ~6-7h (queued)
- [ ] ProtBERT: ~2h (queued)
- [ ] Ankh Large: ~2h (queued)
- [ ] Concatenate embeddings: ~5 min (final step)

**Total Phase 1 ETA:** ~14-15 hours (completes ~3-4 PM Oct 19)

### 📋 Upcoming (Phase 2+)
- [ ] Phase 2: Model Architecture (Week 2-3)
- [ ] Phase 3: Training Strategy (Week 3-4)
- [ ] Phase 4: Ensemble & Calibration (Week 4)
- [ ] Phase 5: CUDA Acceleration Showcase
- [ ] Phase 6: Final Submission

## Real-time Monitoring

### GPU Usage (During Embedding Generation)
```powershell
# Monitor GPU in real-time
nvidia-smi -l 30
```

**Current Stats:**
- Utilization: 100%
- Memory: 4.5-6.5 GB / 12.2 GB
- Temperature: 58-62°C
- Power: 92-94W / 95W
- Status: ✅ Optimal Performance

### Check Embedding Progress
```powershell
# View current embeddings
ls ..\embeddings\
```

## Phase 1: Feature Engineering (Next Steps)

### 1. Download Data
Visit [competition data page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data) and download to `data/` folder.

### 2. Generate Embeddings
```powershell
cd scripts
python generate_embeddings.py  # Takes 6-8 hours
```

**Resource Usage:**
- ESM-2 3B: ~11GB VRAM, batch_size=4
- ESM-2 650M: ~8GB VRAM, batch_size=16
- ProtBERT: ~6GB VRAM, batch_size=12
- Ankh: ~6GB VRAM, batch_size=12

### 3. Concatenate Embeddings
```powershell
python concat_embeddings.py  # Takes ~5 minutes
```

**Output:** 4,500-dimensional feature vectors per protein

## Troubleshooting

### RTX 5070 Ti Compatibility Warning
If you see:
```
UserWarning: NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible
```

**Solution:**
```powershell
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

### Out of Memory (OOM) Errors
Reduce batch sizes in `generate_embeddings.py`:
```python
models = [
    ('esm2_650m', 8),   # Reduce from 16
    ('esm2_3b', 2),     # Reduce from 4
    ('protbert', 8),    # Reduce from 12
    ('ankh', 8)         # Reduce from 12
]
```

### CUDA Not Available
Check NVIDIA drivers:
```powershell
nvidia-smi
```
Should show CUDA 12.8+ for RTX 5070 Ti.

## Technical Approach Summary

### Multi-Model Embedding Fusion
- **ESM-2 3B**: 1,280-dim (global context)
- **ESM-2 650M**: 1,280-dim (local patterns)
- **ProtBERT**: 1,024-dim (domain boundaries)
- **Ankh Large**: 1,536-dim (specialized families)
- **Total**: 4,500-dim concatenated features

### Key Innovations
1. Asymmetric Loss (ASL) for extreme class imbalance
2. Hierarchical Bayesian inference on GO DAG
3. Graph Attention Networks for term relationships
4. Per-term threshold optimization
5. Calibrated meta-ensemble (7+ models)

## Resources

- **Competition:** https://www.kaggle.com/competitions/cafa-6-protein-function-prediction
- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html
- **ESM-2 Paper:** https://www.science.org/doi/10.1126/science.ade2574
- **Gene Ontology:** http://geneontology.org/

## Team & Contact

**Hardware:** RTX 5070 Ti Laptop (12GB), i9 Ultra 2, 64GB RAM  
**Environment:** Windows 11, Conda, PyTorch 2.10.0.dev+cu128  
**Competition Deadline:** Check Kaggle timeline

---

**Last Updated:** October 19, 2025  
**Status:** Phase 1 Setup Complete, Ready for Execution