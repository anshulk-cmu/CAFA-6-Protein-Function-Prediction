# Phase 1B: Profiling Analysis & Kernel Optimization Roadmap

**Tool:** `torch.profiler` with CUDA activity tracking  
**Profiled:** 3 batches per model (warmup + active profiling)  
**Output:** Chrome traces + kernel statistics

---

## Kernel-Level Bottleneck Analysis

Profiling identified **19.8 seconds total CUDA time** across 3 models, with clear optimization targets:

### ESM2-3B: Compute-Heavy Workload
- **Total CUDA Time:** 11.6 seconds
- **Top Bottleneck:** GEMM kernels (3.7s, 23.3% of time)
  - `ampere_sgemm_128x64_tn`: 2.1s across 396 calls
  - `ampere_sgemm_128x128_tn`: 1.5s across 252 calls
- **Secondary Target:** Linear layers (3.6s, 22.5%)
- **Observation:** 651 linear operations dominate—ripe for kernel fusion

### ESM-C-600M: Flash Attention Optimized
- **Total CUDA Time:** 4.3 seconds
- **Already Optimized:** Uses `fmha_cutlassF` flash attention (113ms, 8.1%)
- **Remaining Target:** Matrix operations (1.9s, 34.0%)
- **Key Insight:** Flash attention already provides 2-4x speedup—minimal further gains available

### ProtT5-XL: Memory Overhead Issue
- **Total CUDA Time:** 3.9 seconds
- **Critical Finding:** 344ms (6.8%) wasted on **dtype conversions**
  - 678 `aten::copy_` calls converting FP16↔FP32
  - Pure overhead—can be eliminated
- **Primary Target:** Matrix operations (1.8s, 35.3%)
- **Fix:** Unify precision (use FP16 throughout, eliminate conversions)

---

## Phase 2 Optimization Strategy

### Priority 1: Custom GEMM Kernels (High Impact)
**Target:** ESM2-3B's `ampere_sgemm_*` operations (3.7s → 1.2-1.8s)  
**Approach:** Tiled matrix multiplication with shared memory optimization  
**Expected Gain:** 2-3x speedup on 23% of compute = **~1.5-2.5s saved per batch**

### Priority 2: Kernel Fusion (Medium Impact)
**Target:** Linear + activation across all models (~5s total)  
**Approach:** Fuse GEMM + ReLU/GELU into single kernel  
**Expected Gain:** 1.5-2x on fused operations = **~1.5-2.5s saved**

### Priority 3: Eliminate ProtT5 Conversions (Low Effort, High Efficiency)
**Target:** 344ms dtype overhead  
**Fix:** Load model in FP16, avoid FP32 conversions  
**Expected Gain:** 100% elimination = **344ms saved + reduced memory fragmentation**

---

## Projected Phase 2 Results

**Current GPU Time (esm2_3B):** 2m 38s per 1K proteins  
**After Optimization:** <1m 30s (1.8-2.2x improvement)  
**Combined Speedup:** 11.8x → **21-26x total** (meets Track 2 20-30x target)

This analysis provides actionable kernel-level insights for Track 2 custom CUDA development.