"""
smith_waterman.py
Python API for GPU-Accelerated Smith-Waterman Sequence Alignment
Phase 2A: CAFA6 Project - User-Friendly Interface

Usage:
    from smith_waterman import align_sequences, compute_train_similarity_matrix

    # Single alignment
    score = align_sequences(protein1, protein2)

    # All-vs-all for training data
    similarity_matrix = compute_train_similarity_matrix(train_sequences)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm
import pickle
import logging

# Import compiled CUDA extension (will be built by setup.py)
try:
    import smith_waterman_cuda
    CUDA_AVAILABLE = True
except ImportError as e:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(
        f"CUDA extension not found: {e}\n"
        "Please build the extension by running:\n"
        "  cd kernels/smith_waterman\n"
        "  python setup.py install"
    )

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Core Alignment Functions
# ============================================================================

def align_sequences(seq_a: str, seq_b: str) -> float:
    """
    Align two protein sequences using GPU-accelerated Smith-Waterman

    Args:
        seq_a: First protein sequence (amino acid string)
        seq_b: Second protein sequence

    Returns:
        Normalized alignment score (0.0 to 1.0)

    Example:
        >>> score = align_sequences("ARNDCQEGH", "ARDCQEG")
        >>> print(f"Similarity: {score:.3f}")
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available. Please build first.")

    # Validate sequences
    if not seq_a or not seq_b:
        raise ValueError("Empty sequences provided")

    # Call CUDA kernel via C++ wrapper
    scores = smith_waterman_cuda.align_batch([seq_a], [seq_b])
    return float(scores[0])


def align_batch(
    sequences_a: List[str],
    sequences_b: List[str],
    batch_size: int = 10000,
    show_progress: bool = True
) -> np.ndarray:
    """
    Align batches of sequence pairs using GPU

    Args:
        sequences_a: List of query sequences
        sequences_b: List of database sequences (must match length)
        batch_size: Number of alignments per GPU batch (default: 10000)
        show_progress: Show tqdm progress bar

    Returns:
        NumPy array of alignment scores [num_pairs]

    Example:
        >>> queries = ["ARND", "QEGH", "ILKM"]
        >>> targets = ["ARDC", "QEGS", "ILKV"]
        >>> scores = align_batch(queries, targets)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    if len(sequences_a) != len(sequences_b):
        raise ValueError("sequences_a and sequences_b must have same length")

    num_pairs = len(sequences_a)

    if num_pairs == 0:
        return np.array([])

    # Process in chunks to avoid GPU memory overflow
    all_scores = []

    progress_bar = tqdm(
        total=num_pairs,
        desc="Aligning sequences",
        disable=not show_progress,
        unit="pairs"
    )

    for start_idx in range(0, num_pairs, batch_size):
        end_idx = min(start_idx + batch_size, num_pairs)

        # Extract chunk
        chunk_a = sequences_a[start_idx:end_idx]
        chunk_b = sequences_b[start_idx:end_idx]

        # Process on GPU
        chunk_scores = smith_waterman_cuda.align_batch(chunk_a, chunk_b)

        # Convert to NumPy and accumulate
        all_scores.append(chunk_scores.numpy())

        progress_bar.update(end_idx - start_idx)

    progress_bar.close()

    return np.concatenate(all_scores)


# ============================================================================
# Phase 2A: CAFA Dataset Processing
# ============================================================================

def compute_train_similarity_matrix(
    sequences: List[str],
    batch_size: int = 10000,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute N×N all-vs-all similarity matrix for training proteins

    Phase 2A Use Case: 3,000 training proteins → 3,000×3,000 = 9M alignments

    Args:
        sequences: List of N protein sequences
        batch_size: Alignments per GPU batch
        save_path: Optional path to save result as .npz file

    Returns:
        Symmetric N×N similarity matrix as NumPy array

    Example:
        >>> train_seqs = load_fasta("train_sequences_3k.fasta")
        >>> similarity = compute_train_similarity_matrix(train_seqs)
        >>> print(f"Shape: {similarity.shape}")  # (3000, 3000)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    n = len(sequences)
    logger.info(f"Computing {n}×{n} similarity matrix ({n*(n+1)//2:,} alignments)")

    # Use optimized all-vs-all function from C++ wrapper
    similarity_matrix = smith_waterman_cuda.align_all_vs_all(sequences, batch_size)
    similarity_matrix = similarity_matrix.numpy()

    # Validate symmetry
    assert np.allclose(similarity_matrix, similarity_matrix.T), "Matrix not symmetric!"

    # Save if requested
    if save_path:
        save_similarity_matrix(similarity_matrix, save_path)
        logger.info(f"Saved similarity matrix to {save_path}")

    return similarity_matrix


def compute_test_train_similarity(
    test_sequences: List[str],
    train_sequences: List[str],
    batch_size: int = 10000,
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Compute M×N similarity matrix (test vs training proteins)

    Phase 2A Use Case: 1,000 test × 3,000 train = 3M alignments

    Args:
        test_sequences: List of M test protein sequences
        train_sequences: List of N training protein sequences
        batch_size: Alignments per GPU batch
        save_path: Optional save path

    Returns:
        M×N similarity matrix (test rows, train columns)

    Example:
        >>> test_seqs = load_fasta("test_sequences_1k.fasta")
        >>> train_seqs = load_fasta("train_sequences_3k.fasta")
        >>> similarity = compute_test_train_similarity(test_seqs, train_seqs)
        >>> print(f"Shape: {similarity.shape}")  # (1000, 3000)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA extension not available")

    m = len(test_sequences)
    n = len(train_sequences)

    logger.info(f"Computing {m}×{n} test-train similarity matrix ({m*n:,} alignments)")

    # Generate all pairs
    pairs_a = []
    pairs_b = []

    for test_seq in test_sequences:
        for train_seq in train_sequences:
            pairs_a.append(test_seq)
            pairs_b.append(train_seq)

    # Align all pairs
    scores = align_batch(pairs_a, pairs_b, batch_size=batch_size)

    # Reshape into matrix
    similarity_matrix = scores.reshape(m, n)

    # Save if requested
    if save_path:
        save_similarity_matrix(similarity_matrix, save_path)
        logger.info(f"Saved test-train similarity to {save_path}")

    return similarity_matrix


# ============================================================================
# KNN-based Feature Extraction (for CAFA Competition)
# ============================================================================

def get_top_k_neighbors(
    similarity_matrix: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract top-K most similar proteins for each query

    Args:
        similarity_matrix: N×M similarity matrix
        k: Number of nearest neighbors to retrieve

    Returns:
        indices: [N, k] array of neighbor indices
        scores: [N, k] array of similarity scores

    Example:
        >>> indices, scores = get_top_k_neighbors(similarity, k=10)
        >>> print(f"Protein 0 most similar to: {indices[0]}")
    """
    n = similarity_matrix.shape[0]

    # Get top-k indices (excluding self if diagonal exists)
    top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :k]

    # Get corresponding scores
    top_k_scores = np.take_along_axis(
        similarity_matrix, top_k_indices, axis=1
    )

    return top_k_indices, top_k_scores


def create_knn_features(
    similarity_matrix: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    Create KNN-based feature vectors from similarity matrix

    Feature vector per protein: [top-k similarity scores]
    Can be concatenated with ESM/ProtBERT embeddings

    Args:
        similarity_matrix: N×M similarity matrix
        k: Number of neighbors

    Returns:
        Feature matrix [N, k]
    """
    _, top_k_scores = get_top_k_neighbors(similarity_matrix, k)
    return top_k_scores


# ============================================================================
# File I/O Utilities
# ============================================================================

def load_fasta(fasta_path: str) -> List[str]:
    """
    Load protein sequences from FASTA file

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of protein sequences (amino acid strings)
    """
    sequences = []
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                current_seq = []
            elif line:
                current_seq.append(line)

        if current_seq:
            sequences.append(''.join(current_seq))

    logger.info(f"Loaded {len(sequences)} sequences from {fasta_path}")
    return sequences


def save_similarity_matrix(matrix: np.ndarray, save_path: str):
    """Save similarity matrix to disk (compressed NumPy format)"""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, similarity=matrix)


def load_similarity_matrix(save_path: str) -> np.ndarray:
    """Load similarity matrix from disk"""
    data = np.load(save_path)
    return data['similarity']


# ============================================================================
# Benchmarking & Statistics
# ============================================================================

def compute_alignment_statistics(
    sequences_a: List[str],
    sequences_b: List[str]
) -> dict:
    """
    Compute statistics about sequence pairs (for benchmark analysis)

    Returns:
        Dictionary with length statistics, GC content, etc.
    """
    lengths_a = [len(s) for s in sequences_a]
    lengths_b = [len(s) for s in sequences_b]

    return {
        'num_pairs': len(sequences_a),
        'length_a_mean': np.mean(lengths_a),
        'length_a_median': np.median(lengths_a),
        'length_a_min': np.min(lengths_a),
        'length_a_max': np.max(lengths_a),
        'length_b_mean': np.mean(lengths_b),
        'length_b_median': np.median(lengths_b),
        'length_b_min': np.min(lengths_b),
        'length_b_max': np.max(lengths_b),
        'total_amino_acids': sum(lengths_a) + sum(lengths_b)
    }


def benchmark_throughput(
    num_pairs: int = 1000,
    seq_length: int = 400
) -> dict:
    """
    Benchmark alignment throughput on synthetic data

    Args:
        num_pairs: Number of sequence pairs to test
        seq_length: Average sequence length

    Returns:
        Performance metrics (time, throughput, etc.)
    """
    import time

    # Generate random protein sequences
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    np.random.seed(42)

    sequences_a = [
        ''.join(np.random.choice(list(amino_acids), seq_length))
        for _ in range(num_pairs)
    ]
    sequences_b = [
        ''.join(np.random.choice(list(amino_acids), seq_length))
        for _ in range(num_pairs)
    ]

    # Benchmark
    logger.info(f"Benchmarking {num_pairs} alignments (seq_len={seq_length})...")

    start_time = time.time()
    scores = align_batch(sequences_a, sequences_b, show_progress=True)
    elapsed = time.time() - start_time

    throughput = num_pairs / elapsed

    return {
        'num_pairs': num_pairs,
        'seq_length': seq_length,
        'elapsed_seconds': elapsed,
        'throughput_pairs_per_sec': throughput,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }


# ============================================================================
# Phase 2A Main Workflow
# ============================================================================

def run_phase2a_workflow(
    train_fasta: str,
    test_fasta: str,
    output_dir: str = "similarity_matrices",
    k_neighbors: int = 10
) -> dict:
    """
    Complete Phase 2A workflow: Generate all similarity matrices

    Processes:
    1. Train-vs-train (3K×3K = 9M alignments)
    2. Test-vs-train (1K×3K = 3M alignments)
    3. Extract top-K neighbors for KNN features

    Args:
        train_fasta: Path to training sequences (3K proteins)
        test_fasta: Path to test sequences (1K proteins)
        output_dir: Directory to save similarity matrices
        k_neighbors: Number of neighbors for KNN features

    Returns:
        Dictionary with paths to saved files and statistics
    """
    logger.info("=" * 80)
    logger.info("Phase 2A: Smith-Waterman Similarity Matrix Generation")
    logger.info("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sequences
    logger.info("\n[1/4] Loading sequences...")
    train_seqs = load_fasta(train_fasta)
    test_seqs = load_fasta(test_fasta)

    # Compute train-train similarity
    logger.info(f"\n[2/4] Computing train-train similarity ({len(train_seqs)}×{len(train_seqs)})...")
    train_similarity = compute_train_similarity_matrix(
        train_seqs,
        save_path=str(output_path / "train_similarity_3k.npz")
    )

    # Compute test-train similarity
    logger.info(f"\n[3/4] Computing test-train similarity ({len(test_seqs)}×{len(train_seqs)})...")
    test_train_similarity = compute_test_train_similarity(
        test_seqs,
        train_seqs,
        save_path=str(output_path / "test_train_similarity_1k_3k.npz")
    )

    # Extract KNN features
    logger.info(f"\n[4/4] Extracting top-{k_neighbors} neighbors for KNN features...")
    train_knn_features = create_knn_features(train_similarity, k=k_neighbors)
    test_knn_features = create_knn_features(test_train_similarity, k=k_neighbors)

    np.save(output_path / "train_knn_features.npy", train_knn_features)
    np.save(output_path / "test_knn_features.npy", test_knn_features)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Phase 2A Complete!")
    logger.info("=" * 80)
    logger.info(f"Train similarity matrix: {train_similarity.shape}")
    logger.info(f"Test-train similarity: {test_train_similarity.shape}")
    logger.info(f"KNN features (train): {train_knn_features.shape}")
    logger.info(f"KNN features (test): {test_knn_features.shape}")
    logger.info(f"\nOutput directory: {output_path.absolute()}")

    return {
        'train_similarity_path': str(output_path / "train_similarity_3k.npz"),
        'test_train_similarity_path': str(output_path / "test_train_similarity_1k_3k.npz"),
        'train_knn_features_path': str(output_path / "train_knn_features.npy"),
        'test_knn_features_path': str(output_path / "test_knn_features.npy"),
        'train_shape': train_similarity.shape,
        'test_shape': test_train_similarity.shape,
        'num_alignments': len(train_seqs)**2 + len(test_seqs)*len(train_seqs)
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Benchmark throughput
    print("Running benchmark...")
    stats = benchmark_throughput(num_pairs=1000, seq_length=400)
    print(f"\nThroughput: {stats['throughput_pairs_per_sec']:.1f} alignments/sec")
    print(f"Time: {stats['elapsed_seconds']:.2f} seconds")
