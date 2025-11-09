"""
smith_waterman.py
Python API for GPU-accelerated Smith-Waterman alignment
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm

# Import compiled CUDA extension (will be built by setup.py)
try:
    import smith_waterman_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA extension not found. Run 'python setup.py install' first.")


class SmithWatermanGPU:
    """GPU-accelerated Smith-Waterman sequence alignment"""

    def __init__(self, batch_size: int = 500):
        """
        Initialize Smith-Waterman GPU processor

        Args:
            batch_size: Number of alignments to process per GPU batch
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available")

        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def align_pair(self, seq_a: str, seq_b: str) -> float:
        """
        Align two sequences

        Args:
            seq_a: First protein sequence
            seq_b: Second protein sequence

        Returns:
            Alignment score
        """
        scores = self.align_batch([seq_a], [seq_b])
        return scores[0]

    def align_batch(self, sequences_a: List[str], sequences_b: List[str]) -> np.ndarray:
        """
        Align batches of sequence pairs

        Args:
            sequences_a: List of first sequences
            sequences_b: List of second sequences

        Returns:
            Array of alignment scores [num_pairs]
        """
        assert len(sequences_a) == len(sequences_b), "Sequence lists must have same length"

        num_pairs = len(sequences_a)
        all_scores = []

        # Process in batches to avoid GPU memory overflow
        for i in tqdm(range(0, num_pairs, self.batch_size), desc="Aligning sequences"):
            batch_a = sequences_a[i:i + self.batch_size]
            batch_b = sequences_b[i:i + self.batch_size]

            # Call CUDA kernel
            scores_tensor = smith_waterman_cuda.align_batch(batch_a, batch_b)
            scores = scores_tensor.cpu().numpy()
            all_scores.append(scores)

        return np.concatenate(all_scores)

    def align_all_vs_all(self, sequences: List[str]) -> np.ndarray:
        """
        Compute all-vs-all alignment matrix

        Args:
            sequences: List of protein sequences

        Returns:
            Similarity matrix [N x N]
        """
        n = len(sequences)
        similarity_matrix = np.zeros((n, n), dtype=np.float32)

        # Generate all pairs (upper triangle)
        pairs_a = []
        pairs_b = []
        indices = []

        for i in range(n):
            for j in range(i, n):
                pairs_a.append(sequences[i])
                pairs_b.append(sequences[j])
                indices.append((i, j))

        # Compute alignments
        scores = self.align_batch(pairs_a, pairs_b)

        # Fill matrix (symmetric)
        for (i, j), score in zip(indices, scores):
            similarity_matrix[i, j] = score
            similarity_matrix[j, i] = score

        return similarity_matrix


def align_sequences(seq_a: str, seq_b: str) -> float:
    """
    Convenience function: align two sequences

    Args:
        seq_a: First protein sequence
        seq_b: Second protein sequence

    Returns:
        Alignment score
    """
    aligner = SmithWatermanGPU()
    return aligner.align_pair(seq_a, seq_b)
