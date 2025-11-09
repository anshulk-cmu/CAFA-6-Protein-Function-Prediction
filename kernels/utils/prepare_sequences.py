"""
prepare_sequences.py
Extract 4,000 sequences (3K train + 1K test) from CAFA dataset
"""

import random
from pathlib import Path
from typing import List, Tuple
import pickle

def read_fasta(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Read FASTA file and return (id, sequence) tuples

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of (protein_id, sequence) tuples
    """
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def stratified_sample(sequences: List[Tuple[str, str]], n: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample sequences stratified by length

    Args:
        sequences: List of (id, sequence) tuples
        n: Number of sequences to sample
        seed: Random seed

    Returns:
        Stratified sample
    """
    random.seed(seed)

    # Sort by length
    sorted_seqs = sorted(sequences, key=lambda x: len(x[1]))

    # Divide into bins
    bin_size = len(sorted_seqs) // n
    sampled = []

    for i in range(n):
        start = i * bin_size
        end = start + bin_size
        if i == n - 1:
            end = len(sorted_seqs)

        if start < len(sorted_seqs):
            sampled.append(random.choice(sorted_seqs[start:end]))

    return sampled


def main():
    """Extract and save 4K sequences for Smith-Waterman benchmarking"""

    # Paths (adjust based on your data location)
    data_dir = Path("/data/user_data/anshulk/cafa6/data")
    train_fasta = data_dir / "train_sequences.fasta"
    test_fasta = data_dir / "train_sequences_benchmark_1k.fasta"  # Your Phase 1B test set

    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)

    print("Reading FASTA files...")
    train_sequences = read_fasta(train_fasta)
    print(f"Total training sequences: {len(train_sequences)}")

    # Sample 3,000 training sequences (stratified by length)
    print("Sampling 3,000 training sequences...")
    train_sample = stratified_sample(train_sequences, 3000)

    # Read existing 1K test set from Phase 1B
    print("Reading 1,000 test sequences...")
    test_sequences = read_fasta(test_fasta)[:1000]

    # Save
    print("Saving sequences...")
    with open(output_dir / "sw_train_3k.pkl", 'wb') as f:
        pickle.dump(train_sample, f)

    with open(output_dir / "sw_test_1k.pkl", 'wb') as f:
        pickle.dump(test_sequences, f)

    # Summary
    train_lens = [len(seq) for _, seq in train_sample]
    test_lens = [len(seq) for _, seq in test_sequences]

    print("\n=== Summary ===")
    print(f"Train: {len(train_sample)} sequences")
    print(f"  Length range: {min(train_lens)} - {max(train_lens)}")
    print(f"  Mean length: {sum(train_lens) / len(train_lens):.0f}")

    print(f"\nTest: {len(test_sequences)} sequences")
    print(f"  Length range: {min(test_lens)} - {max(test_lens)}")
    print(f"  Mean length: {sum(test_lens) / len(test_lens):.0f}")

    print(f"\nTotal alignments to compute:")
    print(f"  Train-vs-train: 3,000 × 3,000 = 9,000,000")
    print(f"  Test-vs-train:  1,000 × 3,000 = 3,000,000")
    print(f"  TOTAL: 12,000,000 alignments")


if __name__ == "__main__":
    main()
