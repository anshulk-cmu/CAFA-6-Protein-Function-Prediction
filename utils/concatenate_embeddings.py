#!/usr/bin/env python3
"""
Concatenate embeddings from multiple models into a single tensor.

Combines embeddings from ESM2-3B, ESM-C-600M, ESM1b, ProtT5-XL, and ProstT5
into a unified feature vector for downstream training.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np


def setup_logging():
    """Setup logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def normalize_embeddings(embeddings: torch.Tensor, method: str = 'none') -> torch.Tensor:
    """
    Normalize embeddings.

    Args:
        embeddings: Tensor of shape [N, D]
        method: 'none', 'l2', or 'standardize'

    Returns:
        Normalized embeddings
    """
    if method == 'none':
        return embeddings

    elif method == 'l2':
        # L2 normalize each embedding to unit norm
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-9)  # Avoid division by zero
        return embeddings / norms

    elif method == 'standardize':
        # Standardize to mean=0, std=1
        mean = embeddings.mean(dim=0, keepdim=True)
        std = embeddings.std(dim=0, keepdim=True)
        std = torch.clamp(std, min=1e-9)  # Avoid division by zero
        return (embeddings - mean) / std

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_statistics(embeddings: torch.Tensor) -> dict:
    """
    Compute statistics for embeddings.

    Args:
        embeddings: Tensor of shape [N, D]

    Returns:
        Dictionary with statistics
    """
    embeddings_np = embeddings.numpy() if isinstance(embeddings, torch.Tensor) else embeddings

    return {
        'shape': list(embeddings.shape),
        'mean': float(embeddings_np.mean()),
        'std': float(embeddings_np.std()),
        'min': float(embeddings_np.min()),
        'max': float(embeddings_np.max()),
        'sparsity': float((embeddings_np == 0).mean()),
        'nan_count': int(np.isnan(embeddings_np).sum()),
        'inf_count': int(np.isinf(embeddings_np).sum())
    }


def main():
    parser = argparse.ArgumentParser(description='Concatenate embeddings from multiple models')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--embeddings-dir', type=str,
                       default='/data/user_data/anshulk/cafa6/embeddings',
                       help='Directory containing embedding files')
    parser.add_argument('--output-dir', type=str,
                       default='/data/user_data/anshulk/cafa6/embeddings',
                       help='Output directory')
    parser.add_argument('--normalize', type=str, default='l2',
                       choices=['none', 'l2', 'standardize'],
                       help='Normalization method')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['esm2_3b', 'esm_c_600m', 'esm1b', 'prot_t5_xl', 'prost_t5'],
                       help='Models to concatenate (in order)')

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 70)
    logging.info("Phase 1B: Embedding Concatenation")
    logging.info("=" * 70)
    logging.info(f"Split: {args.split}")
    logging.info(f"Models: {', '.join(args.models)}")
    logging.info(f"Normalization: {args.normalize}")

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model dimension mapping
    model_dims = {
        'esm2_3b': 2560,
        'esm_c_600m': 1152,
        'esm1b': 1280,
        'prot_t5_xl': 1024,
        'prost_t5': 1024
    }

    # Load all embeddings
    all_embeddings = []
    model_info = []

    logging.info(f"\n{'Loading embeddings':^70}")
    logging.info("-" * 70)

    for model_name in args.models:
        # Construct filename
        filename = f"{args.split}_embeddings_{model_name}.pt"
        filepath = embeddings_dir / filename

        logging.info(f"\nLoading {model_name}:")
        logging.info(f"  File: {filepath}")

        if not filepath.exists():
            logging.error(f"  ERROR: File not found!")
            logging.error(f"  Skipping {model_name}")
            continue

        # Load embeddings
        try:
            embeddings = torch.load(filepath)
            logging.info(f"  Shape: {embeddings.shape}")

            # Verify shape
            expected_dim = model_dims[model_name]
            if embeddings.shape[1] != expected_dim:
                logging.warning(f"  WARNING: Expected dim {expected_dim}, got {embeddings.shape[1]}")

            # Compute statistics before normalization
            stats_before = compute_statistics(embeddings)
            logging.info(f"  Mean: {stats_before['mean']:.4f}, Std: {stats_before['std']:.4f}")
            logging.info(f"  Range: [{stats_before['min']:.4f}, {stats_before['max']:.4f}]")
            logging.info(f"  Sparsity: {stats_before['sparsity']*100:.2f}%")

            # Check for NaN/Inf
            if stats_before['nan_count'] > 0:
                logging.error(f"  ERROR: Found {stats_before['nan_count']} NaN values!")
            if stats_before['inf_count'] > 0:
                logging.error(f"  ERROR: Found {stats_before['inf_count']} Inf values!")

            # Normalize if requested
            if args.normalize != 'none':
                logging.info(f"  Normalizing with method: {args.normalize}")
                embeddings = normalize_embeddings(embeddings, args.normalize)
                stats_after = compute_statistics(embeddings)
                logging.info(f"  After normalization - Mean: {stats_after['mean']:.4f}, Std: {stats_after['std']:.4f}")

            all_embeddings.append(embeddings)

            # Record model info
            model_info.append({
                'model_name': model_name,
                'dim': embeddings.shape[1],
                'dim_start': sum([m['dim'] for m in model_info]),
                'dim_end': sum([m['dim'] for m in model_info]) + embeddings.shape[1],
                'stats_before': stats_before,
                'stats_after': compute_statistics(embeddings) if args.normalize != 'none' else stats_before
            })

            logging.info(f"  ✓ Loaded successfully")

        except Exception as e:
            logging.error(f"  ERROR loading embeddings: {e}")
            continue

    if len(all_embeddings) == 0:
        logging.error("\nNo embeddings loaded! Exiting.")
        return

    # Verify all embeddings have same number of proteins
    num_proteins = [emb.shape[0] for emb in all_embeddings]
    if len(set(num_proteins)) > 1:
        logging.error(f"\nERROR: Inconsistent number of proteins: {num_proteins}")
        return

    logging.info(f"\n{'Concatenating embeddings':^70}")
    logging.info("-" * 70)
    logging.info(f"Number of proteins: {num_proteins[0]}")
    logging.info(f"Number of models: {len(all_embeddings)}")

    # Concatenate along feature dimension
    concatenated = torch.cat(all_embeddings, dim=1)
    logging.info(f"Concatenated shape: {concatenated.shape}")

    # Verify concatenated dimensions
    expected_total_dim = sum([info['dim'] for info in model_info])
    if concatenated.shape[1] != expected_total_dim:
        logging.error(f"ERROR: Expected total dim {expected_total_dim}, got {concatenated.shape[1]}")

    # Make contiguous for better performance
    concatenated = concatenated.contiguous()
    logging.info(f"Memory contiguous: {concatenated.is_contiguous()}")

    # Compute final statistics
    final_stats = compute_statistics(concatenated)
    logging.info(f"\nFinal statistics:")
    logging.info(f"  Mean: {final_stats['mean']:.4f}")
    logging.info(f"  Std: {final_stats['std']:.4f}")
    logging.info(f"  Range: [{final_stats['min']:.4f}, {final_stats['max']:.4f}]")

    # Save concatenated embeddings
    output_filename = f"{args.split}_embeddings_concatenated.pt"
    output_path = output_dir / output_filename

    logging.info(f"\nSaving concatenated embeddings:")
    logging.info(f"  Path: {output_path}")
    logging.info(f"  Size: {concatenated.element_size() * concatenated.nelement() / (1024**3):.2f} GB")

    torch.save(concatenated, output_path)
    logging.info(f"  ✓ Saved successfully")

    # Save metadata
    metadata = {
        'split': args.split,
        'num_proteins': int(num_proteins[0]),
        'total_dimensions': int(concatenated.shape[1]),
        'normalization_method': args.normalize,
        'models': model_info,
        'final_stats': final_stats,
        'output_file': str(output_path),
        'creation_date': datetime.now().isoformat()
    }

    metadata_filename = f"{args.split}_embeddings_metadata.json"
    metadata_path = output_dir / metadata_filename

    logging.info(f"\nSaving metadata:")
    logging.info(f"  Path: {metadata_path}")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"  ✓ Saved successfully")

    # Print summary table
    logging.info(f"\n{'Model Dimension Summary':^70}")
    logging.info("-" * 70)
    logging.info(f"{'Model':<20} {'Dimensions':<15} {'Range':<20}")
    logging.info("-" * 70)

    for info in model_info:
        dim_range = f"[{info['dim_start']}:{info['dim_end']}]"
        logging.info(f"{info['model_name']:<20} {info['dim']:<15} {dim_range:<20}")

    logging.info("-" * 70)
    logging.info(f"{'TOTAL':<20} {concatenated.shape[1]:<15}")
    logging.info("-" * 70)

    logging.info("\n" + "=" * 70)
    logging.info("Concatenation complete!")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
