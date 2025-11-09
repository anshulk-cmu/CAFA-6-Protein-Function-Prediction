# Smith-Waterman GPU-Accelerated Sequence Alignment

**Phase 2A: CAFA6 Project - Custom CUDA Kernel Implementation**

GPU-accelerated Smith-Waterman local sequence alignment for processing 12 million protein pairs (3K×3K train + 1K×3K test).

## Features

- ⚡ **3-Way Performance Comparison**: Sequential CPU → Parallel CPU (16-24 cores) → GPU CUDA
- 🎯 **Production-Ready**: Handles CAFA dataset (3,000 train + 1,000 test proteins)
- 🔧 **Custom CUDA Kernel**: Anti-diagonal parallelization with tile-based optimization
- 📊 **Competition Features**: Generates KNN similarity features for protein function prediction
- 🛠️ **Easy to Use**: Simple Python API with progress bars
- 📈 **Complete Baselines**: CPU implementations for fair performance comparison
- 🏆 **Library Comparison**: Benchmarks against CUDASW++4.0 (state-of-the-art)

## Requirements

- CUDA Toolkit 11.8+
- NVIDIA GPU with compute capability 8.0+ (RTX A6000, A100, RTX 3090, etc.)
- Python 3.9+
- PyTorch 2.0+ with CUDA support

## Installation

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm

# Build CUDA extension
cd kernels/smith_waterman
python setup.py install
```

## Quick Start

```python
from smith_waterman import align_sequences, run_phase2a_workflow

# Single alignment
score = align_sequences("ARNDCQEGHILKMFPSTWYV", "ARNDQEGHILKPSTWYV")
print(f"Similarity: {score:.3f}")

# Complete Phase 2A workflow (12M alignments)
results = run_phase2a_workflow(
    train_fasta="train_sequences_3k.fasta",
    test_fasta="test_sequences_1k.fasta",
    output_dir="similarity_matrices"
)
```

## Performance (3-Way Comparison)

**12 Million Alignments (Full CAFA Dataset):**

| Implementation | Time | Speedup vs Sequential | Speedup vs Parallel CPU |
|---|---|---|---|
| Sequential CPU (1 core) | ~33 hours | 1.0x (baseline) | - |
| Parallel CPU (24 cores) | ~2-3 hours | ~12-15x | 1.0x (best CPU) |
| **GPU CUDA Kernel** | **~40 minutes** | **~49x** | **~3-4x** |

**Key Insight**: Even compared to fully parallelized multi-core CPU, GPU provides additional 3-4x speedup, making interactive protein analysis feasible.

## Architecture

- **CUDA Kernel**: Anti-diagonal wavefront parallelization with 16×16 tiles
- **Memory**: Shared memory optimization, boundary buffers for tile communication
- **Scoring**: BLOSUM62 substitution matrix in constant memory
- **Parallelism**: 256 threads/block, thousands of concurrent alignments

## Benchmarking

### 3-Way CPU/GPU Comparison

Run the complete performance comparison:

```bash
# Run all three benchmarks (sequential sample, parallel full, GPU full)
python benchmark_comparison.py --mode all --num-workers 24 --num-pairs 10000

# Run only specific benchmarks
python benchmark_comparison.py --mode sequential  # Extrapolate from 200 samples
python benchmark_comparison.py --mode parallel    # Full parallel CPU
python benchmark_comparison.py --mode gpu         # GPU CUDA kernel

# Custom configuration
python benchmark_comparison.py \
    --num-pairs 50000 \
    --num-workers 16 \
    --sequential-samples 500 \
    --output-dir my_results
```

**Output:**
- `benchmark_results/benchmark_comparison.json`: Raw timing data
- `benchmark_results/benchmark_comparison.md`: Markdown report with tables

### Library Comparison (CUDASW++4.0)

Compare our custom kernel against the state-of-the-art CUDASW++4.0 library:

```bash
# Direct comparison against CUDASW++4.0 (if installed)
python benchmark_library_comparison.py --num-pairs 10000 --output-dir library_comparison

# Use simulated CUDASW++ benchmark (for demonstration without library)
python benchmark_library_comparison.py --use-mock --num-pairs 10000
```

**Output:**
- `library_comparison/library_comparison.json`: Performance metrics
- `library_comparison/library_comparison.md`: Feature and performance comparison
- Shows whether custom kernel matches/exceeds library performance

**Note:** If CUDASW++4.0 is not installed, the script will use simulated benchmarks based on published CUDASW++4.0 performance data (1.94-5.71 TCUPS on A100/H100 GPUs, ~2.0 TCUPS on RTX A6000).

## Output Files

**Similarity Matrices:**
- `train_similarity_3k.npz`: 3,000×3,000 similarity matrix (~36 MB)
- `test_train_similarity_1k_3k.npz`: 1,000×3,000 similarity matrix (~12 MB)
- `train_knn_features.npy`: Top-10 neighbor features for training set (~120 KB)
- `test_knn_features.npy`: Top-10 neighbor features for test set (~40 KB)

**Benchmark Results:**
- `benchmark_comparison.json`: 3-way performance metrics (JSON)
- `benchmark_comparison.md`: 3-way comparison report (Markdown)
- `library_comparison.json`: CUDASW++4.0 comparison metrics (JSON)
- `library_comparison.md`: Library comparison report (Markdown)

## License

MIT License
