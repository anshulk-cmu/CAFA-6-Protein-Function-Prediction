"""
benchmark_comparison.py
3-Way Performance Comparison: Sequential CPU vs Parallel CPU vs GPU
Phase 2A: CAFA6 Project - Complete Benchmarking Suite

Compares three implementations:
1. Sequential CPU (Naive Baseline) - 200 samples → extrapolate to 33 hours
2. Parallel CPU (Best CPU) - Full dataset on all cores → 2-3 hours
3. GPU CUDA Kernel - Full dataset → 40-45 minutes

Usage:
    python benchmark_comparison.py --mode all --num-workers 24

Output:
    - Timing results for all three approaches
    - Speedup calculations
    - JSON report and markdown table
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

# CPU implementations
from cpu_smith_waterman import (
    align_batch_sequential,
    align_batch_parallel,
    benchmark_sequential_sample,
    AMINO_ACIDS
)

# GPU implementation (conditionally import)
try:
    from smith_waterman import align_batch as align_batch_gpu
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    import warnings
    warnings.warn("GPU implementation not available. Build with: python setup.py install")

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Synthetic Data Generation
# ============================================================================

def generate_test_sequences(num_sequences: int, avg_length: int = 400, seed: int = 42) -> List[str]:
    """
    Generate random protein sequences for benchmarking

    Args:
        num_sequences: Number of sequences to generate
        avg_length: Average sequence length (will vary ±20%)
        seed: Random seed for reproducibility

    Returns:
        List of protein sequences
    """
    np.random.seed(seed)
    sequences = []

    for i in range(num_sequences):
        # Vary length by ±20%
        length = int(avg_length * (0.8 + 0.4 * np.random.random()))
        seq = ''.join(np.random.choice(list(AMINO_ACIDS), length))
        sequences.append(seq)

    logger.info(f"Generated {num_sequences} sequences (avg length: {avg_length})")
    return sequences


# ============================================================================
# Benchmark 1: Sequential CPU (Naive Baseline)
# ============================================================================

def run_sequential_benchmark(num_samples: int = 200, seq_length: int = 400) -> Dict:
    """
    Run sequential CPU benchmark on small sample

    Strategy: Run 200 samples → extrapolate to 12M alignments

    Returns:
        Dictionary with timing and extrapolation
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK 1: Sequential CPU (Naive Baseline)")
    logger.info("=" * 80)
    logger.info(f"Running {num_samples} sample alignments to extrapolate full runtime...")

    # Generate test data
    np.random.seed(42)
    sequences_a = [
        ''.join(np.random.choice(list(AMINO_ACIDS), seq_length))
        for _ in range(num_samples)
    ]
    sequences_b = [
        ''.join(np.random.choice(list(AMINO_ACIDS), seq_length))
        for _ in range(num_samples)
    ]

    # Benchmark
    start_time = time.time()
    scores = align_batch_sequential(sequences_a, sequences_b, show_progress=True)
    elapsed = time.time() - start_time

    # Extrapolate to 12M alignments
    time_per_alignment = elapsed / num_samples
    total_alignments = 12_000_000
    extrapolated_seconds = time_per_alignment * total_alignments
    extrapolated_hours = extrapolated_seconds / 3600

    results = {
        'name': 'Sequential CPU',
        'num_samples': num_samples,
        'actual_elapsed_seconds': elapsed,
        'time_per_alignment_ms': time_per_alignment * 1000,
        'throughput_pairs_per_sec': num_samples / elapsed,
        'extrapolated_total_seconds': extrapolated_seconds,
        'extrapolated_hours': extrapolated_hours,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }

    logger.info(f"✓ Sequential CPU: {num_samples} alignments in {elapsed:.2f}s")
    logger.info(f"  Throughput: {results['throughput_pairs_per_sec']:.1f} pairs/sec")
    logger.info(f"  Extrapolated for 12M: {extrapolated_hours:.1f} hours ({extrapolated_hours/24:.1f} days)")
    logger.info("")

    return results


# ============================================================================
# Benchmark 2: Parallel CPU (Best CPU Baseline)
# ============================================================================

def run_parallel_benchmark(
    sequences_a: List[str],
    sequences_b: List[str],
    num_workers: int = None
) -> Dict:
    """
    Run parallelized CPU benchmark on full dataset

    Strategy: Use all CPU cores (16-24) for maximum CPU performance

    Returns:
        Dictionary with actual timing
    """
    logger.info("=" * 80)
    logger.info("BENCHMARK 2: Parallel CPU (Best CPU Baseline)")
    logger.info("=" * 80)

    import multiprocessing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    num_pairs = len(sequences_a)

    logger.info(f"Running {num_pairs:,} alignments on {num_workers} CPU cores...")

    start_time = time.time()
    scores = align_batch_parallel(sequences_a, sequences_b, num_workers=num_workers, show_progress=True)
    elapsed = time.time() - start_time

    results = {
        'name': 'Parallel CPU',
        'num_pairs': num_pairs,
        'num_workers': num_workers,
        'elapsed_seconds': elapsed,
        'elapsed_hours': elapsed / 3600,
        'throughput_pairs_per_sec': num_pairs / elapsed,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }

    logger.info(f"✓ Parallel CPU ({num_workers} cores): {num_pairs:,} alignments in {elapsed:.2f}s ({elapsed/3600:.2f}h)")
    logger.info(f"  Throughput: {results['throughput_pairs_per_sec']:.1f} pairs/sec")
    logger.info("")

    return results


# ============================================================================
# Benchmark 3: GPU CUDA Kernel
# ============================================================================

def run_gpu_benchmark(
    sequences_a: List[str],
    sequences_b: List[str]
) -> Dict:
    """
    Run GPU CUDA benchmark on full dataset

    Strategy: Single GPU processes all alignments with custom kernel

    Returns:
        Dictionary with actual timing
    """
    if not GPU_AVAILABLE:
        logger.warning("GPU implementation not available - skipping GPU benchmark")
        return None

    logger.info("=" * 80)
    logger.info("BENCHMARK 3: GPU CUDA Kernel")
    logger.info("=" * 80)

    num_pairs = len(sequences_a)

    logger.info(f"Running {num_pairs:,} alignments on GPU...")

    start_time = time.time()
    scores = align_batch_gpu(sequences_a, sequences_b, batch_size=10000, show_progress=True)
    elapsed = time.time() - start_time

    results = {
        'name': 'GPU CUDA',
        'num_pairs': num_pairs,
        'elapsed_seconds': elapsed,
        'elapsed_minutes': elapsed / 60,
        'throughput_pairs_per_sec': num_pairs / elapsed,
        'mean_score': float(np.mean(scores)),
        'std_score': float(np.std(scores))
    }

    logger.info(f"✓ GPU CUDA: {num_pairs:,} alignments in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    logger.info(f"  Throughput: {results['throughput_pairs_per_sec']:.1f} pairs/sec")
    logger.info("")

    return results


# ============================================================================
# Comparison & Report Generation
# ============================================================================

def generate_comparison_report(
    sequential_results: Dict,
    parallel_results: Dict,
    gpu_results: Dict = None,
    output_dir: str = "benchmark_results"
) -> Dict:
    """
    Generate comprehensive comparison report

    Creates:
    - JSON file with raw results
    - Markdown table
    - Speedup calculations

    Returns:
        Summary dictionary
    """
    logger.info("=" * 80)
    logger.info("PERFORMANCE COMPARISON SUMMARY")
    logger.info("=" * 80)

    # Calculate speedups
    seq_throughput = sequential_results['throughput_pairs_per_sec']
    par_throughput = parallel_results['throughput_pairs_per_sec']

    par_speedup_vs_seq = par_throughput / seq_throughput

    summary = {
        'sequential_cpu': sequential_results,
        'parallel_cpu': parallel_results,
        'speedups': {
            'parallel_vs_sequential': par_speedup_vs_seq
        }
    }

    # GPU comparison (if available)
    if gpu_results:
        gpu_throughput = gpu_results['throughput_pairs_per_sec']
        gpu_speedup_vs_seq = gpu_throughput / seq_throughput
        gpu_speedup_vs_par = gpu_throughput / par_throughput

        summary['gpu_cuda'] = gpu_results
        summary['speedups']['gpu_vs_sequential'] = gpu_speedup_vs_seq
        summary['speedups']['gpu_vs_parallel'] = gpu_speedup_vs_par

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS:")
    print("=" * 80)
    print(f"\n1. Sequential CPU (Naive Baseline)")
    print(f"   - Sample: {sequential_results['num_samples']} alignments in {sequential_results['actual_elapsed_seconds']:.2f}s")
    print(f"   - Throughput: {seq_throughput:.1f} pairs/sec")
    print(f"   - Extrapolated 12M: {sequential_results['extrapolated_hours']:.1f} hours")

    print(f"\n2. Parallel CPU ({parallel_results['num_workers']} cores)")
    print(f"   - Actual: {parallel_results['num_pairs']:,} alignments in {parallel_results['elapsed_hours']:.2f}h")
    print(f"   - Throughput: {par_throughput:.1f} pairs/sec")
    print(f"   - Speedup vs Sequential: {par_speedup_vs_seq:.1f}x")

    if gpu_results:
        print(f"\n3. GPU CUDA Kernel")
        print(f"   - Actual: {gpu_results['num_pairs']:,} alignments in {gpu_results['elapsed_minutes']:.2f} min")
        print(f"   - Throughput: {gpu_throughput:.1f} pairs/sec")
        print(f"   - Speedup vs Sequential: {gpu_speedup_vs_seq:.1f}x")
        print(f"   - Speedup vs Parallel CPU: {gpu_speedup_vs_par:.1f}x")

    print("\n" + "=" * 80)

    # Save JSON report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "benchmark_comparison.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"✓ Saved JSON report: {json_path}")

    # Generate markdown table
    markdown_table = generate_markdown_table(summary)
    md_path = output_path / "benchmark_comparison.md"
    with open(md_path, 'w') as f:
        f.write(markdown_table)

    logger.info(f"✓ Saved Markdown report: {md_path}")

    return summary


def generate_markdown_table(summary: Dict) -> str:
    """Generate markdown table for report"""
    seq = summary['sequential_cpu']
    par = summary['parallel_cpu']
    speedups = summary['speedups']

    has_gpu = 'gpu_cuda' in summary
    if has_gpu:
        gpu = summary['gpu_cuda']

    md = "# Smith-Waterman Performance Comparison\n\n"
    md += "## Phase 2A: CPU vs GPU Benchmark Results\n\n"

    # Main comparison table
    md += "| Implementation | Alignments | Time | Throughput (pairs/sec) | Speedup vs Sequential |\n"
    md += "|---|---|---|---|---|\n"

    # Sequential row
    md += f"| Sequential CPU (1 core) | {seq['num_samples']} samples | {seq['actual_elapsed_seconds']:.2f}s | {seq['throughput_pairs_per_sec']:.1f} | 1.0x (baseline) |\n"

    # Parallel row
    par_speedup = speedups['parallel_vs_sequential']
    md += f"| Parallel CPU ({par['num_workers']} cores) | {par['num_pairs']:,} | {par['elapsed_hours']:.2f}h | {par['throughput_pairs_per_sec']:.1f} | **{par_speedup:.1f}x** |\n"

    # GPU row (if available)
    if has_gpu:
        gpu_speedup_seq = speedups['gpu_vs_sequential']
        gpu_speedup_par = speedups['gpu_vs_parallel']
        md += f"| GPU CUDA Kernel | {gpu['num_pairs']:,} | {gpu['elapsed_minutes']:.2f} min | {gpu['throughput_pairs_per_sec']:.1f} | **{gpu_speedup_seq:.1f}x** |\n"

    md += "\n"

    # Extrapolation section
    md += "## Extrapolation to Full CAFA Dataset (12M alignments)\n\n"
    md += "| Implementation | Estimated Time | Notes |\n"
    md += "|---|---|---|\n"
    md += f"| Sequential CPU | {seq['extrapolated_hours']:.1f} hours ({seq['extrapolated_hours']/24:.1f} days) | Impractical |\n"

    # Calculate parallel extrapolation
    par_rate = par['throughput_pairs_per_sec']
    par_12m_seconds = 12_000_000 / par_rate
    par_12m_hours = par_12m_seconds / 3600
    md += f"| Parallel CPU ({par['num_workers']} cores) | {par_12m_hours:.1f} hours | Practical but slow |\n"

    if has_gpu:
        gpu_rate = gpu['throughput_pairs_per_sec']
        gpu_12m_seconds = 12_000_000 / gpu_rate
        gpu_12m_minutes = gpu_12m_seconds / 60
        md += f"| GPU CUDA | {gpu_12m_minutes:.1f} minutes ({gpu_12m_minutes/60:.2f} hours) | **{gpu_speedup_par:.1f}x faster than best CPU** |\n"

    md += "\n"

    # Key findings
    md += "## Key Findings\n\n"
    md += f"1. **Parallel CPU Speedup**: {par_speedup:.1f}x faster than sequential (using {par['num_workers']} cores)\n"

    if has_gpu:
        md += f"2. **GPU Speedup vs Sequential**: {gpu_speedup_seq:.1f}x faster than single-core CPU\n"
        md += f"3. **GPU Speedup vs Best CPU**: {gpu_speedup_par:.1f}x faster than fully parallelized CPU\n"
        md += f"4. **Real-world Impact**: Reduces 12M alignments from {par_12m_hours:.1f}h (CPU) to {gpu_12m_minutes:.1f} min (GPU)\n"

    return md


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="3-way Smith-Waterman performance comparison")

    parser.add_argument('--mode', type=str, choices=['all', 'sequential', 'parallel', 'gpu'],
                        default='all', help='Which benchmarks to run')

    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU workers for parallel (default: all cores)')

    parser.add_argument('--num-pairs', type=int, default=10000,
                        help='Number of alignment pairs for full benchmarks (default: 10000)')

    parser.add_argument('--seq-length', type=int, default=400,
                        help='Average sequence length (default: 400)')

    parser.add_argument('--sequential-samples', type=int, default=200,
                        help='Number of samples for sequential benchmark (default: 200)')

    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Output directory for results')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Smith-Waterman 3-Way Performance Comparison")
    logger.info("Phase 2A: CPU Sequential vs CPU Parallel vs GPU CUDA")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Num pairs (full benchmarks): {args.num_pairs:,}")
    logger.info(f"  Sequence length: {args.seq_length}")
    logger.info(f"  Sequential samples: {args.sequential_samples}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info("")

    # Generate test data for full benchmarks
    if args.mode in ['all', 'parallel', 'gpu']:
        logger.info(f"Generating {args.num_pairs:,} test sequence pairs...")
        sequences_a = generate_test_sequences(args.num_pairs, args.seq_length, seed=42)
        sequences_b = generate_test_sequences(args.num_pairs, args.seq_length, seed=43)

    # Run benchmarks
    sequential_results = None
    parallel_results = None
    gpu_results = None

    if args.mode in ['all', 'sequential']:
        sequential_results = run_sequential_benchmark(
            num_samples=args.sequential_samples,
            seq_length=args.seq_length
        )

    if args.mode in ['all', 'parallel']:
        parallel_results = run_parallel_benchmark(
            sequences_a, sequences_b, num_workers=args.num_workers
        )

    if args.mode in ['all', 'gpu']:
        if GPU_AVAILABLE:
            gpu_results = run_gpu_benchmark(sequences_a, sequences_b)
        else:
            logger.warning("GPU benchmark requested but GPU not available - skipping")

    # Generate comparison report
    if args.mode == 'all' and sequential_results and parallel_results:
        generate_comparison_report(
            sequential_results,
            parallel_results,
            gpu_results,
            output_dir=args.output_dir
        )

    logger.info("✓ Benchmarking complete!")


if __name__ == "__main__":
    main()
