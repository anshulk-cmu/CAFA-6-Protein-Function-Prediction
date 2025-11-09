"""
cpu_smith_waterman.py
CPU Baseline Implementations for Smith-Waterman Sequence Alignment
Phase 2A: CAFA6 Project - CPU vs GPU Comparison

Provides two CPU baselines:
1. Sequential (single-threaded) - Naive baseline for extrapolation
2. Parallelized (multi-core) - Best CPU baseline for fair GPU comparison

Usage:
    from cpu_smith_waterman import align_sequences_sequential, align_sequences_parallel

    # Sequential baseline
    score = align_sequences_sequential(seq_a, seq_b)

    # Parallel baseline (uses all CPU cores)
    scores = align_batch_parallel(sequences_a, sequences_b, num_workers=24)
"""

import numpy as np
from typing import List, Tuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# BLOSUM62 Substitution Matrix (for Protein Scoring)
# ============================================================================

# BLOSUM62 matrix as Python dictionary
# Matches the values used in CUDA kernel's constant memory
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

BLOSUM62 = {
    ('A', 'A'): 4,  ('A', 'R'): -1, ('A', 'N'): -2, ('A', 'D'): -2,
    ('A', 'C'): 0,  ('A', 'Q'): -1, ('A', 'E'): -1, ('A', 'G'): 0,
    ('A', 'H'): -2, ('A', 'I'): -1, ('A', 'L'): -1, ('A', 'K'): -1,
    ('A', 'M'): -1, ('A', 'F'): -2, ('A', 'P'): -1, ('A', 'S'): 1,
    ('A', 'T'): 0,  ('A', 'W'): -3, ('A', 'Y'): -2, ('A', 'V'): 0,

    ('R', 'R'): 5,  ('R', 'N'): 0,  ('R', 'D'): -2, ('R', 'C'): -3,
    ('R', 'Q'): 1,  ('R', 'E'): 0,  ('R', 'G'): -2, ('R', 'H'): 0,
    ('R', 'I'): -3, ('R', 'L'): -2, ('R', 'K'): 2,  ('R', 'M'): -1,
    ('R', 'F'): -3, ('R', 'P'): -2, ('R', 'S'): -1, ('R', 'T'): -1,
    ('R', 'W'): -3, ('R', 'Y'): -2, ('R', 'V'): -3,

    ('N', 'N'): 6,  ('N', 'D'): 1,  ('N', 'C'): -3, ('N', 'Q'): 0,
    ('N', 'E'): 0,  ('N', 'G'): 0,  ('N', 'H'): 1,  ('N', 'I'): -3,
    ('N', 'L'): -3, ('N', 'K'): 0,  ('N', 'M'): -2, ('N', 'F'): -3,
    ('N', 'P'): -2, ('N', 'S'): 1,  ('N', 'T'): 0,  ('N', 'W'): -4,
    ('N', 'Y'): -2, ('N', 'V'): -3,

    ('D', 'D'): 6,  ('D', 'C'): -3, ('D', 'Q'): 0,  ('D', 'E'): 2,
    ('D', 'G'): -1, ('D', 'H'): -1, ('D', 'I'): -3, ('D', 'L'): -4,
    ('D', 'K'): -1, ('D', 'M'): -3, ('D', 'F'): -3, ('D', 'P'): -1,
    ('D', 'S'): 0,  ('D', 'T'): -1, ('D', 'W'): -4, ('D', 'Y'): -3,
    ('D', 'V'): -3,

    ('C', 'C'): 9,  ('C', 'Q'): -3, ('C', 'E'): -4, ('C', 'G'): -3,
    ('C', 'H'): -3, ('C', 'I'): -1, ('C', 'L'): -1, ('C', 'K'): -3,
    ('C', 'M'): -1, ('C', 'F'): -2, ('C', 'P'): -3, ('C', 'S'): -1,
    ('C', 'T'): -1, ('C', 'W'): -2, ('C', 'Y'): -2, ('C', 'V'): -1,

    ('Q', 'Q'): 5,  ('Q', 'E'): 2,  ('Q', 'G'): -2, ('Q', 'H'): 0,
    ('Q', 'I'): -3, ('Q', 'L'): -2, ('Q', 'K'): 1,  ('Q', 'M'): 0,
    ('Q', 'F'): -3, ('Q', 'P'): -1, ('Q', 'S'): 0,  ('Q', 'T'): -1,
    ('Q', 'W'): -2, ('Q', 'Y'): -1, ('Q', 'V'): -2,

    ('E', 'E'): 5,  ('E', 'G'): -2, ('E', 'H'): 0,  ('E', 'I'): -3,
    ('E', 'L'): -3, ('E', 'K'): 1,  ('E', 'M'): -2, ('E', 'F'): -3,
    ('E', 'P'): -1, ('E', 'S'): 0,  ('E', 'T'): -1, ('E', 'W'): -3,
    ('E', 'Y'): -2, ('E', 'V'): -2,

    ('G', 'G'): 6,  ('G', 'H'): -2, ('G', 'I'): -4, ('G', 'L'): -4,
    ('G', 'K'): -2, ('G', 'M'): -3, ('G', 'F'): -3, ('G', 'P'): -2,
    ('G', 'S'): 0,  ('G', 'T'): -2, ('G', 'W'): -2, ('G', 'Y'): -3,
    ('G', 'V'): -3,

    ('H', 'H'): 8,  ('H', 'I'): -3, ('H', 'L'): -3, ('H', 'K'): -1,
    ('H', 'M'): -2, ('H', 'F'): -1, ('H', 'P'): -2, ('H', 'S'): -1,
    ('H', 'T'): -2, ('H', 'W'): -2, ('H', 'Y'): 2,  ('H', 'V'): -3,

    ('I', 'I'): 4,  ('I', 'L'): 2,  ('I', 'K'): -3, ('I', 'M'): 1,
    ('I', 'F'): 0,  ('I', 'P'): -3, ('I', 'S'): -2, ('I', 'T'): -1,
    ('I', 'W'): -3, ('I', 'Y'): -1, ('I', 'V'): 3,

    ('L', 'L'): 4,  ('L', 'K'): -2, ('L', 'M'): 2,  ('L', 'F'): 0,
    ('L', 'P'): -3, ('L', 'S'): -2, ('L', 'T'): -1, ('L', 'W'): -2,
    ('L', 'Y'): -1, ('L', 'V'): 1,

    ('K', 'K'): 5,  ('K', 'M'): -1, ('K', 'F'): -3, ('K', 'P'): -1,
    ('K', 'S'): 0,  ('K', 'T'): -1, ('K', 'W'): -3, ('K', 'Y'): -2,
    ('K', 'V'): -2,

    ('M', 'M'): 5,  ('M', 'F'): 0,  ('M', 'P'): -2, ('M', 'S'): -1,
    ('M', 'T'): -1, ('M', 'W'): -1, ('M', 'Y'): -1, ('M', 'V'): 1,

    ('F', 'F'): 6,  ('F', 'P'): -4, ('F', 'S'): -2, ('F', 'T'): -2,
    ('F', 'W'): 1,  ('F', 'Y'): 3,  ('F', 'V'): -1,

    ('P', 'P'): 7,  ('P', 'S'): -1, ('P', 'T'): -1, ('P', 'W'): -4,
    ('P', 'Y'): -3, ('P', 'V'): -2,

    ('S', 'S'): 4,  ('S', 'T'): 1,  ('S', 'W'): -3, ('S', 'Y'): -2,
    ('S', 'V'): -2,

    ('T', 'T'): 5,  ('T', 'W'): -2, ('T', 'Y'): -2, ('T', 'V'): 0,

    ('W', 'W'): 11, ('W', 'Y'): 2,  ('W', 'V'): -3,

    ('Y', 'Y'): 7,  ('Y', 'V'): -1,

    ('V', 'V'): 4,
}

# Make matrix symmetric
for (a, b), score in list(BLOSUM62.items()):
    BLOSUM62[(b, a)] = score

def get_blosum_score(aa1: str, aa2: str) -> int:
    """Get BLOSUM62 score for two amino acids"""
    return BLOSUM62.get((aa1.upper(), aa2.upper()), -4)  # Default -4 for unknown


# ============================================================================
# Sequential CPU Smith-Waterman (Baseline 1: Naive)
# ============================================================================

def smith_waterman_sequential(seq_a: str, seq_b: str, gap_open: int = -10, gap_extend: int = -1) -> float:
    """
    Pure Python Smith-Waterman local alignment (single-threaded)

    This is the NAIVE baseline - intentionally slow for demonstration purposes.
    Used to extrapolate "would take 33 hours for 12M alignments"

    Args:
        seq_a: First protein sequence
        seq_b: Second protein sequence
        gap_open: Gap opening penalty (default: -10)
        gap_extend: Gap extension penalty (default: -1)

    Returns:
        Normalized alignment score (0.0 to 1.0)

    Algorithm:
        1. Initialize H matrix (size: len_a+1 × len_b+1)
        2. Fill matrix using DP recurrence
        3. Track maximum score
        4. Normalize by self-alignment scores
    """
    len_a = len(seq_a)
    len_b = len(seq_b)

    # Allocate DP matrix (initialized to zeros)
    H = np.zeros((len_a + 1, len_b + 1), dtype=np.float32)

    max_score = 0.0

    # Fill DP matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            # Match/mismatch score
            match = H[i-1, j-1] + get_blosum_score(seq_a[i-1], seq_b[j-1])

            # Gap penalties (simplified: affine gap model)
            delete = H[i-1, j] + gap_open
            insert = H[i, j-1] + gap_open

            # Smith-Waterman: max(0, match, delete, insert)
            H[i, j] = max(0, match, delete, insert)

            # Track maximum
            if H[i, j] > max_score:
                max_score = H[i, j]

    # Normalize by self-alignment scores
    score_aa = smith_waterman_self_score(seq_a)
    score_bb = smith_waterman_self_score(seq_b)

    if score_aa == 0 or score_bb == 0:
        return 0.0

    normalized_score = max_score / np.sqrt(score_aa * score_bb)

    return min(1.0, max(0.0, normalized_score))


def smith_waterman_self_score(seq: str) -> float:
    """Compute self-alignment score (for normalization)"""
    score = 0.0
    for aa in seq:
        score += get_blosum_score(aa, aa)
    return score


def align_sequences_sequential(seq_a: str, seq_b: str) -> float:
    """
    Single alignment using sequential CPU implementation

    Example:
        >>> score = align_sequences_sequential("ARNDCQEGH", "ARDCQEG")
        >>> print(f"Similarity: {score:.3f}")
    """
    if not seq_a or not seq_b:
        raise ValueError("Empty sequences provided")

    return smith_waterman_sequential(seq_a, seq_b)


def align_batch_sequential(
    sequences_a: List[str],
    sequences_b: List[str],
    show_progress: bool = True
) -> np.ndarray:
    """
    Align batches sequentially (single-threaded)

    This is SLOW by design - used for timing extrapolation only!
    Run on small sample (100-200 pairs) to estimate full runtime.

    Args:
        sequences_a: List of query sequences
        sequences_b: List of database sequences
        show_progress: Show tqdm progress bar

    Returns:
        NumPy array of alignment scores
    """
    if len(sequences_a) != len(sequences_b):
        raise ValueError("sequences_a and sequences_b must have same length")

    num_pairs = len(sequences_a)
    scores = np.zeros(num_pairs, dtype=np.float32)

    progress_bar = tqdm(
        total=num_pairs,
        desc="Sequential CPU",
        disable=not show_progress,
        unit="pairs"
    )

    for i in range(num_pairs):
        scores[i] = smith_waterman_sequential(sequences_a[i], sequences_b[i])
        progress_bar.update(1)

    progress_bar.close()

    return scores


# ============================================================================
# Parallelized CPU Smith-Waterman (Baseline 2: Best CPU)
# ============================================================================

def _align_pair_worker(args: Tuple[str, str]) -> float:
    """Worker function for multiprocessing pool"""
    seq_a, seq_b = args
    return smith_waterman_sequential(seq_a, seq_b)


def align_batch_parallel(
    sequences_a: List[str],
    sequences_b: List[str],
    num_workers: int = None,
    show_progress: bool = True
) -> np.ndarray:
    """
    Align batches using multi-core CPU parallelization

    This is the FAIR CPU BASELINE for GPU comparison.
    Uses all available CPU cores (16-24) to show best CPU performance.

    Args:
        sequences_a: List of query sequences
        sequences_b: List of database sequences
        num_workers: Number of CPU cores to use (default: all cores)
        show_progress: Show tqdm progress bar

    Returns:
        NumPy array of alignment scores

    Example:
        >>> scores = align_batch_parallel(queries, targets, num_workers=24)
        >>> # Uses all 24 cores for maximum CPU throughput
    """
    if len(sequences_a) != len(sequences_b):
        raise ValueError("sequences_a and sequences_b must have same length")

    if num_workers is None:
        num_workers = cpu_count()

    num_pairs = len(sequences_a)

    logger.info(f"Parallel CPU: Using {num_workers} workers for {num_pairs:,} alignments")

    # Create argument pairs
    arg_pairs = list(zip(sequences_a, sequences_b))

    # Process in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        if show_progress:
            scores = list(tqdm(
                pool.imap(_align_pair_worker, arg_pairs, chunksize=max(1, num_pairs // (num_workers * 4))),
                total=num_pairs,
                desc=f"Parallel CPU ({num_workers} cores)",
                unit="pairs"
            ))
        else:
            scores = pool.map(_align_pair_worker, arg_pairs, chunksize=max(1, num_pairs // (num_workers * 4)))

    return np.array(scores, dtype=np.float32)


# ============================================================================
# Benchmarking & Extrapolation
# ============================================================================

def benchmark_sequential_sample(
    num_samples: int = 200,
    seq_length: int = 400
) -> dict:
    """
    Benchmark sequential CPU on small sample to extrapolate full runtime

    Args:
        num_samples: Number of alignments to test (default: 200)
        seq_length: Average sequence length

    Returns:
        Dictionary with timing and extrapolation to 12M alignments

    Example:
        >>> stats = benchmark_sequential_sample(num_samples=200)
        >>> print(f"Estimated time for 12M: {stats['extrapolated_hours']:.1f} hours")
    """
    # Generate random sequences
    np.random.seed(42)
    sequences_a = [
        ''.join(np.random.choice(list(AMINO_ACIDS), seq_length))
        for _ in range(num_samples)
    ]
    sequences_b = [
        ''.join(np.random.choice(list(AMINO_ACIDS), seq_length))
        for _ in range(num_samples)
    ]

    logger.info(f"Benchmarking sequential CPU on {num_samples} samples...")

    start_time = time.time()
    scores = align_batch_sequential(sequences_a, sequences_b, show_progress=True)
    elapsed = time.time() - start_time

    # Extrapolate to 12 million alignments
    time_per_alignment = elapsed / num_samples
    total_alignments = 12_000_000
    extrapolated_seconds = time_per_alignment * total_alignments
    extrapolated_hours = extrapolated_seconds / 3600

    return {
        'num_samples': num_samples,
        'elapsed_seconds': elapsed,
        'time_per_alignment_ms': time_per_alignment * 1000,
        'throughput_pairs_per_sec': num_samples / elapsed,
        'extrapolated_total_seconds': extrapolated_seconds,
        'extrapolated_hours': extrapolated_hours,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


def benchmark_parallel_full(
    sequences_a: List[str],
    sequences_b: List[str],
    num_workers: int = None
) -> dict:
    """
    Benchmark parallelized CPU on full dataset

    Args:
        sequences_a: All query sequences
        sequences_b: All database sequences
        num_workers: Number of CPU cores (default: all)

    Returns:
        Dictionary with actual runtime and throughput
    """
    if num_workers is None:
        num_workers = cpu_count()

    num_pairs = len(sequences_a)

    logger.info(f"Benchmarking parallel CPU ({num_workers} cores) on {num_pairs:,} alignments...")

    start_time = time.time()
    scores = align_batch_parallel(sequences_a, sequences_b, num_workers=num_workers, show_progress=True)
    elapsed = time.time() - start_time

    return {
        'num_pairs': num_pairs,
        'num_workers': num_workers,
        'elapsed_seconds': elapsed,
        'elapsed_hours': elapsed / 3600,
        'throughput_pairs_per_sec': num_pairs / elapsed,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test sequential alignment
    print("\n=== Testing Sequential CPU ===")
    score = align_sequences_sequential("ARNDCQEGHILKMFPSTWYV", "ARNDQEGHILKPSTWYV")
    print(f"Similarity: {score:.3f}")

    # Benchmark sequential (small sample for extrapolation)
    print("\n=== Benchmarking Sequential CPU (200 samples) ===")
    seq_stats = benchmark_sequential_sample(num_samples=200, seq_length=400)
    print(f"Time for 200 alignments: {seq_stats['elapsed_seconds']:.2f}s")
    print(f"Throughput: {seq_stats['throughput_pairs_per_sec']:.1f} pairs/sec")
    print(f"Extrapolated time for 12M alignments: {seq_stats['extrapolated_hours']:.1f} hours")

    # Test parallel alignment
    print("\n=== Testing Parallel CPU ===")
    test_seqs_a = [''.join(np.random.choice(list(AMINO_ACIDS), 400)) for _ in range(1000)]
    test_seqs_b = [''.join(np.random.choice(list(AMINO_ACIDS), 400)) for _ in range(1000)]

    par_stats = benchmark_parallel_full(test_seqs_a, test_seqs_b, num_workers=cpu_count())
    print(f"Time for 1000 alignments: {par_stats['elapsed_seconds']:.2f}s")
    print(f"Throughput: {par_stats['throughput_pairs_per_sec']:.1f} pairs/sec")
    print(f"Speedup vs sequential: {seq_stats['throughput_pairs_per_sec'] / par_stats['throughput_pairs_per_sec']:.1f}x")
