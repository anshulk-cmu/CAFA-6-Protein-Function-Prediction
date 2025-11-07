#!/usr/bin/env python3
"""
Validate concatenated embeddings for correctness and data integrity.

Ensures that concatenation was performed correctly and no data corruption occurred.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import numpy as np


def setup_logging():
    """Setup logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def main():
    parser = argparse.ArgumentParser(description='Validate concatenated embeddings')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'test'],
                       help='Dataset split to validate')
    parser.add_argument('--embeddings-dir', type=str,
                       default='/data/user_data/anshulk/cafa6/embeddings',
                       help='Directory containing embedding files')

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 70)
    logging.info("Phase 1B: Validating Concatenated Embeddings")
    logging.info("=" * 70)
    logging.info(f"Split: {args.split}")

    embeddings_dir = Path(args.embeddings_dir)

    # Load metadata
    metadata_path = embeddings_dir / f"{args.split}_embeddings_metadata.json"
    logging.info(f"\nLoading metadata: {metadata_path}")

    if not metadata_path.exists():
        logging.error(f"ERROR: Metadata file not found!")
        logging.error(f"  Expected: {metadata_path}")
        logging.error(f"  Run concatenate_embeddings.py first")
        return

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    logging.info(f"  ✓ Metadata loaded")
    logging.info(f"  Models: {len(metadata['models'])}")
    logging.info(f"  Total dimensions: {metadata['total_dimensions']}")
    logging.info(f"  Normalization: {metadata['normalization_method']}")

    # Load concatenated embeddings
    concat_path = embeddings_dir / f"{args.split}_embeddings_concatenated.pt"
    logging.info(f"\nLoading concatenated embeddings: {concat_path}")

    if not concat_path.exists():
        logging.error(f"ERROR: Concatenated file not found!")
        return

    concatenated = torch.load(concat_path)
    logging.info(f"  ✓ Loaded")
    logging.info(f"  Shape: {concatenated.shape}")

    # Track validation results
    all_checks_passed = True
    validation_results = []

    # Check 1: Shape consistency
    logging.info(f"\n{'Check 1: Shape Consistency':^70}")
    logging.info("-" * 70)

    expected_shape = (metadata['num_proteins'], metadata['total_dimensions'])
    actual_shape = tuple(concatenated.shape)

    if expected_shape == actual_shape:
        logging.info(f"  ✓ PASS: Shape matches expected {expected_shape}")
        validation_results.append(('Shape Consistency', 'PASS', None))
    else:
        logging.error(f"  ✗ FAIL: Expected {expected_shape}, got {actual_shape}")
        validation_results.append(('Shape Consistency', 'FAIL', f"Shape mismatch"))
        all_checks_passed = False

    # Check 2: Data quality (NaN/Inf)
    logging.info(f"\n{'Check 2: Data Quality (NaN/Inf)':^70}")
    logging.info("-" * 70)

    concatenated_np = concatenated.numpy()
    nan_count = np.isnan(concatenated_np).sum()
    inf_count = np.isinf(concatenated_np).sum()

    if nan_count == 0 and inf_count == 0:
        logging.info(f"  ✓ PASS: No NaN or Inf values found")
        validation_results.append(('Data Quality', 'PASS', None))
    else:
        logging.error(f"  ✗ FAIL: Found {nan_count} NaN and {inf_count} Inf values")
        validation_results.append(('Data Quality', 'FAIL', f"NaN: {nan_count}, Inf: {inf_count}"))
        all_checks_passed = False

    # Check 3: Memory contiguity
    logging.info(f"\n{'Check 3: Memory Contiguity':^70}")
    logging.info("-" * 70)

    if concatenated.is_contiguous():
        logging.info(f"  ✓ PASS: Tensor is contiguous")
        validation_results.append(('Memory Contiguity', 'PASS', None))
    else:
        logging.warning(f"  ⚠ WARNING: Tensor is not contiguous (may hurt performance)")
        validation_results.append(('Memory Contiguity', 'WARNING', "Not contiguous"))

    # Check 4: Value preservation (if no normalization was applied)
    logging.info(f"\n{'Check 4: Value Preservation':^70}")
    logging.info("-" * 70)

    if metadata['normalization_method'] == 'none':
        logging.info("  Testing value preservation (no normalization applied)...")

        # Load individual embeddings and verify slices match
        all_match = True
        for model_info in metadata['models']:
            model_name = model_info['model_name']
            dim_start = model_info['dim_start']
            dim_end = model_info['dim_end']

            # Load original embedding
            original_path = embeddings_dir / f"{args.split}_embeddings_{model_name}.pt"

            if not original_path.exists():
                logging.warning(f"    ⚠ Skipping {model_name}: Original file not found")
                continue

            original = torch.load(original_path)

            # Extract slice from concatenated
            concat_slice = concatenated[:, dim_start:dim_end]

            # Compare
            if torch.allclose(original, concat_slice, rtol=1e-5, atol=1e-7):
                logging.info(f"    ✓ {model_name}: Values match")
            else:
                logging.error(f"    ✗ {model_name}: Values DO NOT match!")
                max_diff = torch.max(torch.abs(original - concat_slice)).item()
                logging.error(f"      Max difference: {max_diff}")
                all_match = False
                all_checks_passed = False

        if all_match:
            validation_results.append(('Value Preservation', 'PASS', None))
        else:
            validation_results.append(('Value Preservation', 'FAIL', "Slice mismatch"))

    else:
        logging.info(f"  Skipped (normalization was applied: {metadata['normalization_method']})")
        validation_results.append(('Value Preservation', 'SKIP', 'Normalization applied'))

    # Check 5: Dimension mapping correctness
    logging.info(f"\n{'Check 5: Dimension Mapping':^70}")
    logging.info("-" * 70)

    total_expected = sum([m['dim'] for m in metadata['models']])
    total_actual = metadata['total_dimensions']

    if total_expected == total_actual:
        logging.info(f"  ✓ PASS: Total dimensions match ({total_actual})")
        validation_results.append(('Dimension Mapping', 'PASS', None))
    else:
        logging.error(f"  ✗ FAIL: Expected {total_expected}, got {total_actual}")
        validation_results.append(('Dimension Mapping', 'FAIL', f"Dimension mismatch"))
        all_checks_passed = False

    # Verify dimension ranges don't overlap
    dim_ranges = [(m['dim_start'], m['dim_end']) for m in metadata['models']]
    for i, (start1, end1) in enumerate(dim_ranges):
        for j, (start2, end2) in enumerate(dim_ranges[i+1:], i+1):
            if start1 < end2 and start2 < end1:
                logging.error(f"  ✗ FAIL: Dimension ranges overlap!")
                logging.error(f"    Model {i}: [{start1}, {end1})")
                logging.error(f"    Model {j}: [{start2}, {end2})")
                all_checks_passed = False
                break

    # Check 6: Statistics sanity
    logging.info(f"\n{'Check 6: Statistics Sanity':^70}")
    logging.info("-" * 70)

    stats = metadata['final_stats']
    logging.info(f"  Mean: {stats['mean']:.4f}")
    logging.info(f"  Std: {stats['std']:.4f}")
    logging.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # Check for degenerate embeddings (all zeros, unreasonable values)
    if stats['min'] == 0 and stats['max'] == 0:
        logging.error(f"  ✗ FAIL: All embeddings are zero!")
        validation_results.append(('Statistics Sanity', 'FAIL', 'All zeros'))
        all_checks_passed = False
    elif abs(stats['max']) > 1e6 or abs(stats['min']) > 1e6:
        logging.warning(f"  ⚠ WARNING: Unusually large values detected")
        validation_results.append(('Statistics Sanity', 'WARNING', 'Large values'))
    else:
        logging.info(f"  ✓ PASS: Statistics look reasonable")
        validation_results.append(('Statistics Sanity', 'PASS', None))

    # Print summary
    logging.info(f"\n{'Validation Summary':^70}")
    logging.info("=" * 70)

    for check_name, status, details in validation_results:
        symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠" if status == "WARNING" else "⊗"
        line = f"  {symbol} {check_name}: {status}"
        if details:
            line += f" ({details})"
        logging.info(line)

    logging.info("=" * 70)

    if all_checks_passed:
        logging.info("\n✓ ALL CHECKS PASSED")
        logging.info("Concatenated embeddings are valid and ready for use!")
    else:
        logging.error("\n✗ VALIDATION FAILED")
        logging.error("Please review errors above and re-run concatenation")

    # Save validation report
    report_path = embeddings_dir / f"validation_report_{args.split}.txt"
    logging.info(f"\nSaving validation report: {report_path}")

    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Validation Report: {args.split}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Concatenated file: {concat_path}\n")
        f.write(f"Metadata file: {metadata_path}\n\n")

        f.write("Validation Results:\n")
        f.write("-" * 70 + "\n")

        for check_name, status, details in validation_results:
            f.write(f"{check_name}: {status}")
            if details:
                f.write(f" ({details})")
            f.write("\n")

        f.write("-" * 70 + "\n")
        f.write(f"\nOverall: {'PASS' if all_checks_passed else 'FAIL'}\n")

    logging.info(f"  ✓ Report saved")
    logging.info("\n" + "=" * 70)


if __name__ == '__main__':
    main()
