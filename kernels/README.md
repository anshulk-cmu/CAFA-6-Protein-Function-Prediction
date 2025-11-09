# Custom CUDA Kernels for Protein Bioinformatics

## Phase 2A: Smith-Waterman Sequence Alignment

GPU-accelerated implementation for processing 12 million pairwise protein alignments.

## Directory Structure

```
kernels/
├── smith_waterman/
│   ├── smith_waterman_kernel.cu        # CUDA kernel implementation
│   ├── smith_waterman_wrapper.cpp      # C++ PyTorch binding
│   ├── smith_waterman.py               # Python API
│   └── setup.py                        # Build configuration
├── utils/
│   ├── prepare_sequences.py            # Extract 4K sequences from FASTA
│   └── merge_features.py               # Combine similarity + embeddings
├── tests/
│   └── test_correctness.py             # Validation against Biopython
├── benchmarks/
│   └── benchmark_comprehensive.py      # Full 12M alignment benchmark
├── demo/
│   ├── demo_visualization.py           # Heatmaps and charts
│   └── generate_report.py              # Automated report generation
├── data/
│   └── (Generated similarity matrices)
└── README.md                           # This file
```

## Build Instructions

```bash
cd kernels/smith_waterman
python setup.py install
```

## Usage

```python
import smith_waterman as sw

# Align two sequences
score = sw.align_pair(seq1, seq2)

# Batch processing
scores = sw.align_batch(sequences, batch_size=500)
```

## Performance Target

- **Input**: 12 million pairwise alignments (3K×3K + 1K×3K)
- **CPU Baseline**: ~33 hours
- **GPU Target**: ~40 minutes (20-50x speedup)
- **Memory**: 4-6 GB GPU RAM

## Timeline

- **Day 1 (Nov 10)**: Core implementation + validation
- **Day 2 (Nov 11)**: Full benchmark + profiling + documentation
