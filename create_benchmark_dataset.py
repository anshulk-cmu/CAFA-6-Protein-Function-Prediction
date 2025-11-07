#!/usr/bin/env python3
"""
Create benchmark dataset for Phase 1B CPU/GPU comparison.

Extracts a stratified sample of 1000 proteins from the training set
with diverse length distribution.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import random


def read_fasta(fasta_path: str) -> List[Tuple[str, str, int]]:
    """
    Read FASTA file and return list of (id, sequence, length).

    Args:
        fasta_path: Path to FASTA file

    Returns:
        List of tuples (protein_id, sequence, length)
    """
    proteins = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous protein if exists
                if current_id is not None:
                    seq = ''.join(current_seq)
                    proteins.append((current_id, seq, len(seq)))

                # Start new protein
                current_id = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Save last protein
        if current_id is not None:
            seq = ''.join(current_seq)
            proteins.append((current_id, seq, len(seq)))

    return proteins


def stratified_sample(proteins: List[Tuple[str, str, int]],
                      n_total: int = 1000,
                      n_short: int = 200,
                      n_medium: int = 600,
                      n_long: int = 200,
                      short_threshold: int = 200,
                      long_threshold: int = 500,
                      seed: int = 42) -> List[Tuple[str, str, int]]:
    """
    Create stratified sample by sequence length.

    Args:
        proteins: List of (id, sequence, length) tuples
        n_total: Total number of proteins to sample
        n_short: Number of short sequences (<short_threshold)
        n_medium: Number of medium sequences (between thresholds)
        n_long: Number of long sequences (>long_threshold)
        short_threshold: Upper bound for short sequences
        long_threshold: Lower bound for long sequences
        seed: Random seed

    Returns:
        Stratified sample of proteins
    """
    random.seed(seed)

    # Separate into buckets
    short = [p for p in proteins if p[2] < short_threshold]
    medium = [p for p in proteins if short_threshold <= p[2] <= long_threshold]
    long_seqs = [p for p in proteins if p[2] > long_threshold]

    print(f"Available proteins:")
    print(f"  Short (<{short_threshold} aa): {len(short)}")
    print(f"  Medium ({short_threshold}-{long_threshold} aa): {len(medium)}")
    print(f"  Long (>{long_threshold} aa): {len(long_seqs)}")

    # Sample from each bucket
    sampled_short = random.sample(short, min(n_short, len(short)))
    sampled_medium = random.sample(medium, min(n_medium, len(medium)))
    sampled_long = random.sample(long_seqs, min(n_long, len(long_seqs)))

    # Combine
    sampled = sampled_short + sampled_medium + sampled_long

    print(f"\nSampled proteins:")
    print(f"  Short: {len(sampled_short)}")
    print(f"  Medium: {len(sampled_medium)}")
    print(f"  Long: {len(sampled_long)}")
    print(f"  Total: {len(sampled)}")

    return sampled


def write_fasta(proteins: List[Tuple[str, str, int]], output_path: str):
    """
    Write proteins to FASTA file.

    Args:
        proteins: List of (id, sequence, length) tuples
        output_path: Output FASTA path
    """
    with open(output_path, 'w') as f:
        for protein_id, sequence, _ in proteins:
            f.write(f">{protein_id}\n")
            # Write sequence in lines of 80 characters
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i:i+80]}\n")


def calculate_statistics(proteins: List[Tuple[str, str, int]]) -> dict:
    """
    Calculate statistics for protein sequences.

    Args:
        proteins: List of (id, sequence, length) tuples

    Returns:
        Dictionary with statistics
    """
    lengths = [p[2] for p in proteins]
    lengths.sort()

    n = len(lengths)
    return {
        'count': n,
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_length': sum(lengths) / n,
        'median_length': lengths[n // 2] if n % 2 == 1 else (lengths[n // 2 - 1] + lengths[n // 2]) / 2,
        'total_amino_acids': sum(lengths)
    }


def main():
    parser = argparse.ArgumentParser(description='Create benchmark dataset for Phase 1B')
    parser.add_argument('--input', type=str,
                       default='/data/user_data/anshulk/cafa6/data/train_sequences.fasta',
                       help='Input FASTA file')
    parser.add_argument('--output', type=str,
                       default='data/train_sequences_benchmark_1k.fasta',
                       help='Output FASTA file')
    parser.add_argument('--metadata', type=str,
                       default='data/train_sequences_benchmark_1k_metadata.json',
                       help='Output metadata JSON file')
    parser.add_argument('--n-total', type=int, default=1000,
                       help='Total proteins to sample')
    parser.add_argument('--n-short', type=int, default=200,
                       help='Number of short sequences')
    parser.add_argument('--n-medium', type=int, default=600,
                       help='Number of medium sequences')
    parser.add_argument('--n-long', type=int, default=200,
                       help='Number of long sequences')
    parser.add_argument('--short-threshold', type=int, default=200,
                       help='Upper bound for short sequences (amino acids)')
    parser.add_argument('--long-threshold', type=int, default=500,
                       help='Lower bound for long sequences (amino acids)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print("Phase 1B: Creating Benchmark Dataset")
    print("=" * 70)

    # Read full training set
    print(f"\nReading input FASTA: {args.input}")
    proteins = read_fasta(args.input)
    print(f"Loaded {len(proteins)} proteins")

    # Calculate full dataset statistics
    full_stats = calculate_statistics(proteins)
    print(f"\nFull dataset statistics:")
    print(f"  Count: {full_stats['count']}")
    print(f"  Length range: {full_stats['min_length']}-{full_stats['max_length']} aa")
    print(f"  Mean length: {full_stats['mean_length']:.1f} aa")
    print(f"  Median length: {full_stats['median_length']:.1f} aa")

    # Create stratified sample
    print(f"\nCreating stratified sample...")
    sampled = stratified_sample(
        proteins,
        n_total=args.n_total,
        n_short=args.n_short,
        n_medium=args.n_medium,
        n_long=args.n_long,
        short_threshold=args.short_threshold,
        long_threshold=args.long_threshold,
        seed=args.seed
    )

    # Calculate sample statistics
    sample_stats = calculate_statistics(sampled)
    print(f"\nSample statistics:")
    print(f"  Count: {sample_stats['count']}")
    print(f"  Length range: {sample_stats['min_length']}-{sample_stats['max_length']} aa")
    print(f"  Mean length: {sample_stats['mean_length']:.1f} aa")
    print(f"  Median length: {sample_stats['median_length']:.1f} aa")
    print(f"  Total amino acids: {sample_stats['total_amino_acids']:,}")

    # Write output FASTA
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting benchmark FASTA: {args.output}")
    write_fasta(sampled, args.output)

    # Write metadata
    metadata = {
        'source_file': args.input,
        'output_file': args.output,
        'sampling_method': 'stratified_by_length',
        'parameters': {
            'n_total': args.n_total,
            'n_short': args.n_short,
            'n_medium': args.n_medium,
            'n_long': args.n_long,
            'short_threshold': args.short_threshold,
            'long_threshold': args.long_threshold,
            'seed': args.seed
        },
        'full_dataset_stats': full_stats,
        'sample_stats': sample_stats,
        'protein_ids': [p[0] for p in sampled]
    }

    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing metadata: {args.metadata}")
    with open(args.metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 70)
    print("Benchmark dataset created successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
