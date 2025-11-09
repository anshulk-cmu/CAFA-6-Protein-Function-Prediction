# Phase 2: Custom CUDA Kernel Optimization Plan

**Generated:** 2025-11-09 01:57:19
**Based on:** Phase 1B profiling analysis (torch.profiler)
**Target:** Track 2 GPU Programming Project (Due: Nov 27, 2025)

---

## Current Performance Baseline

Phase 1B achieved **16.69x average speedup** using PyTorch's native GPU operations. Phase 2 will implement custom CUDA kernels to further optimize bottleneck operations identified through profiling.

**Current Speedups:**

- esm2_3B: 11.83x
- esm_c_600m: 11.58x
- prot_t5_xl: 26.67x

**Target:** Achieve 20-30x total speedup through custom kernel optimization.

---

## Profiling-Identified Bottlenecks

**Kernel Time Distribution (Average Across Models):**

- **Matrix Operations:** 31.2% (avg), 7575 ms (total)
- **Other:** 25.8% (avg), 7055 ms (total)
- **Linear Layers:** 18.8% (avg), 5408 ms (total)
- **GEMM (Matrix Multiply):** 18.5% (avg), 5434 ms (total)
- **Attention:** 8.1% (avg), 456 ms (total)
- **Memory Operations:** 6.8% (avg), 344 ms (total)

---

## Optimization Opportunities

### Opportunity 1: Custom GEMM Kernel Implementation

**Category:** GEMM (Matrix Multiply)

**Justification:**

GEMM operations consume 18.5% of average compute time (5434 ms total). Current cuBLAS kernels (`ampere_sgemm_*`) can be optimized with custom tile sizes and shared memory management.

**Implementation Approach:**

Implement tiled matrix multiplication with:
  - Optimized tile dimensions (32×32, 64×64 testing)
  - Shared memory for data reuse
  - Register blocking for reduced global memory access
  - Thread coarsening for better instruction throughput

**Expected Speedup:** 2-3x
**Implementation Difficulty:** High
**Priority:** 1

**Profiling Evidence:**

- Profiling shows GEMM kernels account for 18.5% of compute across all models

---

### Opportunity 2: Fused Linear + Activation Kernel

**Category:** Linear Layers

**Justification:**

Linear layers consume 18.8% of compute time (5408 ms). Currently implemented as separate GEMM + activation kernels, causing redundant memory transfers.

**Implementation Approach:**

Fuse matrix multiply and activation (ReLU/GELU) into single kernel:
  - Compute activation immediately after each output element
  - Eliminate intermediate memory writes
  - Reduce memory bandwidth requirements by ~50%

**Expected Speedup:** 1.5-2x
**Implementation Difficulty:** Medium
**Priority:** 2

**Profiling Evidence:**

- aten::linear appears 3 times across models with 18.8% avg time

---

### Opportunity 3: Element-wise Operation Fusion

**Category:** Element-wise Operations

**Justification:**

Element-wise operations (mul, add) consume 2.4% of time. Multiple separate kernel launches can be fused into single pass.

**Implementation Approach:**

Combine sequential element-wise operations:
  - Fuse aten::mul + aten::add → single fused kernel
  - Reduce kernel launch overhead
  - Improve memory access patterns
  - Single pass through data

**Expected Speedup:** 5-10x
**Implementation Difficulty:** Low
**Priority:** 3

**Profiling Evidence:**

- Profiling shows 381 ms spent on element-wise ops with high kernel launch overhead

---

### Opportunity 4: Eliminate ProtT5-XL FP16/FP32 Conversions

**Category:** Memory Operations

**Justification:**

ProtT5-XL wastes 6.8% of time (344 ms) on dtype conversions between FP16 and FP32. This is pure overhead with no computational benefit.

**Implementation Approach:**

Unify precision across entire inference pipeline:
  - Load model in FP16 natively
  - Remove unnecessary aten::copy_ and aten::to operations
  - Maintain FP16 throughout forward pass
  - Convert to FP32 only at final output if needed

**Expected Speedup:** Eliminate 344 ms overhead (~7% improvement)
**Implementation Difficulty:** Low
**Priority:** 4

**Profiling Evidence:**

- Profiling identifies 1221 dtype conversion operations consuming 344 ms

---

### Opportunity 5: Flash Attention for ESM2-3B

**Category:** Attention

**Justification:**

ESM2-3B uses standard attention (batched matrix multiply) while ESM-C-600M benefits from flash attention. Implementing flash attention for ESM2 will reduce memory bandwidth and improve performance.

**Implementation Approach:**

Port ESM-C's flash attention mechanism:
  - IO-aware attention computation
  - Reduce memory reads/writes
  - Kernel fusion for attention softmax
  - Based on existing fmha_cutlassF implementation

**Expected Speedup:** 1.3-1.5x
**Implementation Difficulty:** High
**Priority:** 5

**Profiling Evidence:**

- ESM-C-600M's flash attention shows 8.1% efficiency; ESM2 uses bmm pattern

---

## Implementation Priority Ranking

Based on expected impact and implementation feasibility:

1. **Custom GEMM Kernel Implementation** (Priority 1)
   - Expected Impact: 2-3x
   - Implementation Effort: Hard
   - Time Saved: 5434 ms → 2717 ms (estimated)

2. **Fused Linear + Activation Kernel** (Priority 2)
   - Expected Impact: 1.5-2x
   - Implementation Effort: Moderate
   - Time Saved: 5408 ms → 2704 ms (estimated)

3. **Element-wise Operation Fusion** (Priority 3)
   - Expected Impact: 5-10x
   - Implementation Effort: Easy
   - Time Saved: 381 ms → 190 ms (estimated)

4. **Eliminate ProtT5-XL FP16/FP32 Conversions** (Priority 4)
   - Expected Impact: Eliminate 344 ms overhead (~7% improvement)
   - Implementation Effort: Easy
   - Time Saved: 344 ms → 172 ms (estimated)

5. **Flash Attention for ESM2-3B** (Priority 5)
   - Expected Impact: 1.3-1.5x
   - Implementation Effort: Hard
   - Time Saved: 0 ms → 0 ms (estimated)

---

## Track 2 Implementation Roadmap

**Timeline:** Nov 10-27, 2025 (18 days available)

**Week 1 (Nov 10-16): Custom GEMM Kernel**

- Implement tiled matrix multiplication baseline
- Optimize tile sizes through empirical testing
- Benchmark against cuBLAS baseline
- Target: 2-3x speedup on GEMM operations

**Week 2 (Nov 17-23): Kernel Fusion**

- Fuse linear + activation operations
- Implement element-wise fusion (if time permits)
- Profile fused kernels with NSight Compute
- Target: Additional 1.5-2x on fused operations

**Week 3 (Nov 24-27): Integration & Reporting**

- Integrate custom kernels into inference pipeline
- End-to-end benchmarking and validation
- Generate Track 2 final report with performance analysis
- Target: 20-30x total speedup (current 16.7x + optimizations)

---

## Success Metrics

**Minimum Viable Product (Track 2 passing grade):**

- ✅ Implement at least 1 custom CUDA kernel (GEMM recommended)
- ✅ Demonstrate measurable speedup over PyTorch baseline
- ✅ Profile with NSight Compute showing kernel-level optimizations
- ✅ Document implementation with code walkthrough

**Stretch Goals:**

- Implement 2-3 custom kernels (GEMM + fusion)
- Achieve 20-30x total speedup (Track 2 target)
- Compare multiple tile sizes and optimization strategies

---

## Conclusion

Phase 1B profiling identified specific kernel bottlenecks with quantified time percentages and call counts. Custom GEMM kernel implementation represents the highest-impact optimization opportunity, consuming 15-23% of compute time across models. A focused 18-day implementation plan targets the Track 2 project deadline while providing clear success criteria.