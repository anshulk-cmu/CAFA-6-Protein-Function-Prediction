# CAFA 6 Protein Function Prediction

**Dual-Track Project:** GPU Programming (Track 2) + CAFA 6 Competition

Predicting protein function using state-of-the-art language models, custom CUDA kernels, and ensemble learning. This project combines cutting-edge deep learning with high-performance GPU computing to tackle the Critical Assessment of protein Function Annotation (CAFA) challenge.

---

## Project Overview

**Track A - CAFA 6 Competition (Deadline: Jan 26, 2026)**
- Goal: Predict Gene Ontology (GO) terms for 224K test proteins
- Target: F-max score 0.38-0.42 (competitive for top 3)
- Prize: $8,000-$15,000

**Track B - GPU Programming Project (Deadline: Nov 27, 2025)**
- Goal: Demonstrate GPU acceleration expertise via custom CUDA kernels
- Custom implementations: Smith-Waterman alignment, GO graph propagation
- Target speedup: 20-100x over CPU/library baselines

---

## Architecture

### Phase 1: Embedding Generation ✅ **COMPLETE**

**5 Protein Language Models:**
- **ESM2-3B** (2,560 dims) - Facebook's 3 billion parameter model
- **ESM-C-600M** (1,152 dims) - ESMplusplus architecture enhancement
- **ESM1b** (1,280 dims) - 650M parameter evolutionary model
- **ProtT5-XL** (1,024 dims) - T5 encoder for protein sequences
- **ProstT5** (1,024 dims) - Structure-aware T5 variant

**Dataset:**
- Training: 82,404 proteins
- Test: 224,309 proteins
- Total embeddings: **7,040 dimensions** (concatenated)

**Performance (Phase 1A - Nov 6):**
- Runtime: 9.2 hours on 2× NVIDIA RTX A6000 (48GB)
- GPU 0 utilization: 52.6% mean (max 100%)
- GPU 1 utilization: 38.1% mean (max 100%)
- Temperature: 64-77°C average (within safe limits)

### Phase 1B: Benchmarking & Analysis 🔄 **IN PROGRESS**

**Current Status (Nov 8, 21:43 EST):**
- CPU Baseline: Running ESM2-3B benchmark (19% complete)
- Estimated completion: ~2 hours

**What's Running:**
1. CPU benchmarks (3 models) - Establish baseline performance
2. GPU benchmarks (3 models) - Measure acceleration
3. torch.profiler analysis - Identify kernel-level bottlenecks
4. Embedding concatenation & validation - Prepare for downstream training

**Expected Results:**
- GPU speedup: 20-30x over CPU for embedding generation
- Profiling: Attention layers dominate compute (65-70%)
- Concatenated embeddings: [82K, 7040] train + [224K, 7040] test

---

## Results Achieved

### Embedding Generation Efficiency
- **Multi-GPU parallelism:** Reduced 18+ hour workload to 9.2 hours
- **Optimized batch sizes:** 50-60% GPU utilization (up from initial 20-30%)
- **Memory management:** FP16 precision, smart checkpointing, length-sorted batching
- **Total embeddings generated:** ~8.5 GB (5 models × 2 splits)

### Infrastructure
- Stratified benchmark dataset (1K proteins) for CPU/GPU comparison
- Comprehensive performance logging (timing, memory, GPU metrics)
- GPU monitoring visualization (6-panel dashboard)
- Automated SLURM pipeline for reproducible benchmarking

---

## Next Steps

### Immediate (Nov 9-10)
- Complete Phase 1B benchmarking
- Analyze CPU vs GPU speedup metrics
- Generate performance visualizations for Track 2 report

### Phase 2: Custom CUDA Kernels (Nov 10-18) 🔥 **PRIORITY**
- **Smith-Waterman alignment kernel:** Batch protein sequence alignment (target: 50-100x CPU speedup)
- **GO graph propagation kernel:** Hierarchical constraint enforcement on 40K term DAG (target: 20-50x library speedup)
- NSight Compute profiling and optimization

### Phase 3: Prediction Pipeline (Nov 19-22)
- Multi-layer transformer for GO term classification
- Integrate custom kernels as drop-in replacements
- Generate CAFA-format predictions (proof of concept)

### Phase 4: Track 2 Submission (Nov 23-27) 📝 **DEADLINE**
- 10-15 page academic report with CUDA kernel analysis
- GitHub repository with comprehensive documentation
- Presentation slides and demo video

### Phase 5-7: CAFA Competition (Dec-Jan)
- Advanced ensemble models (transformer, GNN, similarity transfer)
- Per-term threshold optimization (critical for F-max)
- Final submissions targeting top 3 finish

---

## Repository Structure

```
CAFA-6-Protein-Function-Prediction/
├── benchmark_embeddings_cpu.py       # CPU baseline benchmarking
├── benchmark_embeddings_gpu.py       # GPU benchmarking
├── create_benchmark_dataset.py       # Stratified sampling
├── generate_embeddings.py            # ESM models embedding generation
├── generate_embeddings_t5.py         # T5 models embedding generation
├── config.yaml, config_t5.yaml       # Model configurations
├── slurm_phase1b_benchmarks.sh       # Automated benchmarking pipeline
├── utils/
│   ├── performance_logger.py         # Metrics tracking
│   ├── profile_embeddings.py         # torch.profiler integration
│   ├── concatenate_embeddings.py     # Multi-model fusion
│   ├── validate_concatenated_embeddings.py
│   └── plot_gpu_monitoring.py        # Visualization
├── benchmark_results/                # Performance JSONs (for GitHub)
├── figures/                          # Visualizations (for reports)
└── docs/                             # Documentation
```

---

## Technologies

- **Deep Learning:** PyTorch 2.5, Transformers (HuggingFace)
- **Models:** ESM2, ESMplusplus, ProtT5, ProstT5
- **GPU Computing:** CUDA, cuBLAS, custom kernels
- **Profiling:** torch.profiler, NSight Compute
- **Infrastructure:** SLURM, conda, git

---

## Team & Timeline

**Developer:** Anshul Kumar (anshulk@andrew.cmu.edu)
**Institution:** Carnegie Mellon University
**Hardware:** 2× NVIDIA RTX A6000 (48GB VRAM each)

**Key Milestones:**
- ✅ Nov 6: Phase 1A complete (embedding generation)
- 🔄 Nov 8: Phase 1B in progress (benchmarking)
- 🎯 Nov 27: Track 2 GPU project due
- 🎯 Jan 26: CAFA 6 competition submission

---

## License

MIT License - See LICENSE file for details.

---

## Acknowledgments

- CAFA organizers for the protein function prediction challenge
- Facebook AI Research for ESM models
- Rostlab for ProtT5/ProstT5 models
- Synthyra for ESMplusplus architecture
