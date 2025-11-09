# Phase 1B: GPU Acceleration Benchmark Results

**Date:** November 8-9, 2024  
**Hardware:** NVIDIA RTX A6000 (48GB VRAM)  
**Dataset:** 1,000 protein sequences (stratified by length)

## Performance Summary

Phase 1B benchmarking demonstrates significant GPU acceleration across three protein language models, achieving an **average speedup of 16.7x** over CPU baseline with peak throughput of **37.8 proteins/second**.

### Benchmark Results

| Model | CPU Time | GPU Time | Speedup | GPU Throughput | Peak Memory |
|-------|----------|----------|---------|----------------|-------------|
| **ESM2-3B** | 31m 15s | 2m 39s | **11.8x** | 9.5 p/s | 20.8 GB |
| **ESM-C-600M** | 7m 55s | 41s | **11.6x** | 35.6 p/s | 4.6 GB |
| **ProtT5-XL** | 19m 46s | 45s | **26.7x** | 37.8 p/s | 22.7 GB |

## Key Insights

**ProtT5-XL** delivers the highest speedup (26.7x) with exceptional throughput, making it ideal for rapid large-scale inference despite higher memory usage. **ESM-C-600M** offers the best memory efficiency at 4.6 GB while maintaining strong performance, enabling multi-model parallel processing. **ESM2-3B** provides the richest embeddings (2560-D) with manageable 20.8 GB memory footprint.

All models fit comfortably on a single A6000 GPU, with memory requirements ranging from 4.6-22.7 GB. The consistent 11-27x speedup across architectures validates GPU acceleration as essential for processing the 82K training and 224K test proteins in the CAFA competition timeframe.