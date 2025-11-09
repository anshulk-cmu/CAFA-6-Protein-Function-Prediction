# CAFA 6 Protein Function Prediction

**Dual-Track GPU Acceleration Project** • IIT Jodhpur 
*High-Performance Protein Language Model Inference + Custom CUDA Kernels*

---

## Project Objective

This project tackles protein function prediction through two synchronized tracks: competing in the CAFA 6 challenge while demonstrating advanced GPU programming expertise through custom CUDA kernel development (academic deadline November 27, 2025). The goal is predicting Gene Ontology (GO) terms for 224,309 test proteins using state-of-the-art protein language models accelerated by GPU computing.

## Architecture & Dataset

**Five Protein Language Models Combined:**
- ESM2-3B (2,560 dimensions) - Facebook's 3 billion parameter evolutionary model
- ESM-C-600M (1,152 dimensions) - ESMplusplus architecture enhancement  
- ESM1b (1,280 dimensions) - 650 million parameter model
- ProtT5-XL (1,024 dimensions) - T5 encoder for protein sequences
- ProstT5 (1,024 dimensions) - Structure-aware T5 variant

**Total Dataset:** 82,404 training proteins + 224,309 test proteins, generating 7,040-dimensional concatenated embeddings per protein (~8.5 GB total embeddings).

## Phase 1 Achievements (November 6-9)

**Phase 1A: Multi-GPU Embedding Generation**  
Successfully generated embeddings across five models in 9.2 hours using dual NVIDIA RTX A6000 GPUs (48GB each). Optimized GPU utilization reached 52.6% (GPU 0) and 38.1% (GPU 1) mean with safe operating temperatures (64-77°C). Key optimizations included FP16 mixed precision, length-sorted batching, and smart checkpointing, reducing projected runtime from 18+ hours.

**Phase 1B: Performance Benchmarking & Profiling**  
Comprehensive CPU/GPU benchmarking across 1,000 stratified proteins revealed exceptional acceleration:

- **ESM2-3B:** 11.8x speedup (31m 15s → 2m 39s), 9.5 proteins/second throughput
- **ESM-C-600M:** 11.6x speedup (7m 55s → 41s), 35.6 proteins/second throughput  
- **ProtT5-XL:** 26.7x speedup (19m 46s → 45s), 37.8 proteins/second throughput
- **Average Speedup:** 16.7x across all models

Kernel-level profiling using `torch.profiler` identified critical optimization targets:
- GEMM matrix operations consume 15-23% of compute time (5,434ms total)
- Linear layers represent 18.8% (5,408ms) - prime candidates for kernel fusion
- Memory overhead in ProtT5-XL (344ms) from FP16/FP32 conversions
- Memory efficiency: 4.6GB (ESM-C) to 22.7GB (ProtT5-XL) peak usage

**Data Validation:**  
Concatenated embeddings [82,404 × 7,040] train and [224,309 × 7,040] test passed comprehensive validation: shape consistency, zero NaN/Inf values, proper normalization, and dimension mapping verification.

## Current Focus: Phase 2

**Custom CUDA Kernel Development - Three-Week Sprint:**

**Week 1 (Nov 10-16): Custom GEMM Kernel**  
Implementing tiled matrix multiplication with shared memory optimization targeting 2-3x speedup over cuBLAS baseline. GEMM operations consume 18.5% of compute time, making this the highest-impact optimization.

**Week 2 (Nov 17-23): Kernel Fusion & Optimization**  
Fusing linear + activation operations for 1.5-2x expected speedup and eliminating ProtT5 dtype conversion overhead (344ms savings). NSight Compute profiling will validate optimizations.

**Week 3 (Nov 24-27): Integration & Academic Deliverables**  
End-to-end benchmarking with custom kernels integrated, targeting 20-30x total speedup. Deliverables include a 10-15 page academic report with performance analysis, comprehensive GitHub documentation, and presentation materials.

## Infrastructure & Technologies

**Computing:** Dual NVIDIA RTX A6000 GPUs (48GB VRAM), Intel Xeon CPUs  
**Deep Learning:** PyTorch 2.5, HuggingFace Transformers, mixed precision training  
**GPU Programming:** CUDA, cuBLAS, custom kernels, NSight Compute profiling  
**Orchestration:** SLURM job scheduling, automated benchmarking pipelines  
**Analysis:** torch.profiler, comprehensive performance logging, 6-chart visualization suite

## Next Steps & Timeline

**Phase 2 Completion (November 10-27):** Finalize custom CUDA kernel implementation with comprehensive benchmarking demonstrating 20-30x total speedup. Academic deliverables include detailed performance analysis report, NSight Compute profiling results, and complete GitHub documentation with reproducible benchmarks.

**Competition Optimization (December-January):** Post-academic deadline, dedicate full effort to CAFA leaderboard climb. Implement advanced ensemble models combining transformer architectures with graph neural networks for GO term hierarchy. Develop per-term threshold optimization critical for F-max maximization. Target top-3 competitive ranking (F-max score 0.38-0.42) through iterative model refinement and validation.

## Repository Highlights

Six publication-ready visualizations at 300 DPI (speedup comparisons, throughput analysis, memory utilization, kernel distributions, performance dashboard). Automated analysis pipeline generates comprehensive reports with quantitative metrics. Validation suite ensures data integrity. Modular codebase supports both competitive development and academic requirements with inline CUDA documentation and reproducible benchmarks.

**Contact:** Anshul Kumar • anshulk@andrew.cmu.edu  • g24ait2048@iitj.ac.in
**License:** MIT License