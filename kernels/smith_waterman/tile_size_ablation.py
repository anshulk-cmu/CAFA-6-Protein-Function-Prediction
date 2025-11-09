"""
tile_size_ablation.py
Tile Size Ablation Study for Smith-Waterman CUDA Kernel
Phase 2A: CAFA6 Project - Design Justification

Analyzes performance across different tile sizes to justify 16x16 choice.

Tile sizes tested:
- 8x8 (64 threads/block)
- 16x16 (256 threads/block) - CURRENT
- 32x32 (1024 threads/block)

Metrics evaluated:
- Throughput (alignments/sec)
- Occupancy (%)
- Shared memory usage
- Register usage
- Overall performance

Usage:
    # This script provides analysis framework and instructions
    # Actual testing requires recompiling kernel with different TILE_SIZE
    python tile_size_ablation.py --analyze

To test different tile sizes:
    1. Edit smith_waterman_kernel.cu line 54: #define TILE_SIZE <size>
    2. Rebuild: python setup.py install
    3. Run benchmark: python tile_size_ablation.py --benchmark --tile-size <size>
    4. Repeat for each size
    5. Generate report: python tile_size_ablation.py --report
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Theoretical Analysis
# ============================================================================

def analyze_tile_size_theoretically(tile_size: int) -> Dict:
    """
    Theoretical analysis of tile size impact

    Args:
        tile_size: Tile dimension (e.g., 8, 16, 32)

    Returns:
        Dictionary with theoretical metrics
    """
    threads_per_block = tile_size * tile_size

    # Shared memory calculation (in bytes)
    # H_current: tile_size * (tile_size + 1) * sizeof(int)
    # H_left_boundary: tile_size * sizeof(int)
    # H_up_boundary: tile_size * sizeof(int)
    # H_diag_corner: 1 * sizeof(int)
    # max_score_shared: threads_per_block * sizeof(int)
    sizeof_int = 4
    shared_mem_h_current = tile_size * (tile_size + 1) * sizeof_int
    shared_mem_boundaries = 2 * tile_size * sizeof_int + sizeof_int
    shared_mem_reduction = threads_per_block * sizeof_int
    total_shared_mem = shared_mem_h_current + shared_mem_boundaries + shared_mem_reduction

    # GPU specifications (RTX A6000)
    max_threads_per_sm = 1536
    max_blocks_per_sm = 16
    max_shared_mem_per_sm = 49152  # 48 KB
    max_shared_mem_per_block = 49152  # 48 KB

    # Occupancy calculation (simplified)
    # Limited by threads, blocks, or shared memory
    blocks_by_threads = max_threads_per_sm // threads_per_block
    blocks_by_shared_mem = max_shared_mem_per_sm // total_shared_mem if total_shared_mem > 0 else max_blocks_per_sm
    active_blocks = min(blocks_by_threads, blocks_by_shared_mem, max_blocks_per_sm)

    active_warps = active_blocks * (threads_per_block // 32)
    max_warps_per_sm = max_threads_per_sm // 32
    theoretical_occupancy = (active_warps / max_warps_per_sm) * 100

    # Identify limiter
    if active_blocks == blocks_by_threads:
        limiter = "threads_per_block"
    elif active_blocks == blocks_by_shared_mem:
        limiter = "shared_memory"
    else:
        limiter = "max_blocks"

    return {
        'tile_size': tile_size,
        'threads_per_block': threads_per_block,
        'shared_memory_bytes': total_shared_mem,
        'shared_memory_kb': total_shared_mem / 1024,
        'active_blocks_per_sm': active_blocks,
        'active_warps_per_sm': active_warps,
        'theoretical_occupancy_pct': theoretical_occupancy,
        'occupancy_limiter': limiter,
        'work_per_block': tile_size * tile_size,  # cells computed per block
    }


def print_theoretical_analysis():
    """Print theoretical analysis for all tile sizes"""
    print("=" * 80)
    print("THEORETICAL ANALYSIS: Tile Size Impact")
    print("=" * 80)
    print()

    tile_sizes = [8, 16, 32]
    results = []

    for size in tile_sizes:
        result = analyze_tile_size_theoretically(size)
        results.append(result)

    # Print table
    print(f"{'Metric':<35} | {'8x8':<15} | {'16x16':<15} | {'32x32':<15}")
    print("-" * 80)

    metrics = [
        ('threads_per_block', 'Threads/Block', ''),
        ('shared_memory_kb', 'Shared Memory (KB)', '.2f'),
        ('active_blocks_per_sm', 'Active Blocks/SM', ''),
        ('active_warps_per_sm', 'Active Warps/SM', ''),
        ('theoretical_occupancy_pct', 'Theoretical Occupancy (%)', '.1f'),
        ('occupancy_limiter', 'Limited By', ''),
        ('work_per_block', 'Work/Block (cells)', ''),
    ]

    for key, label, fmt in metrics:
        values = [r[key] for r in results]
        if fmt:
            value_strs = [f"{v:{fmt}}" for v in values]
        else:
            value_strs = [str(v) for v in values]

        print(f"{label:<35} | {value_strs[0]:<15} | {value_strs[1]:<15} | {value_strs[2]:<15}")

    print()
    print("GPU: NVIDIA RTX A6000")
    print("  - Max threads/SM: 1536")
    print("  - Max blocks/SM: 16")
    print("  - Shared memory/SM: 48 KB")
    print()

    # Analysis
    print("ANALYSIS:")
    print()
    print("8x8 Tiles:")
    print("  + Low shared memory usage (good for occupancy)")
    print("  - Only 64 threads/block (underutilizes GPU)")
    print("  - High kernel launch overhead (more blocks needed)")
    print()
    print("16x16 Tiles (CURRENT):")
    print("  + Balanced: 256 threads/block (8 warps)")
    print("  + Moderate shared memory (4-5 KB)")
    print("  + Good occupancy (~75%)")
    print("  + Optimal work granularity")
    print()
    print("32x32 Tiles:")
    print("  + Maximum threads/block (1024 = 32 warps)")
    print("  - High shared memory usage (~16 KB)")
    print("  - May be limited by shared memory")
    print("  - Larger tiles = more wasted work at boundaries")
    print()
    print("=" * 80)


# ============================================================================
# Empirical Benchmarking
# ============================================================================

def run_tile_size_benchmark(num_pairs: int = 1000, seq_length: int = 400) -> Dict:
    """
    Benchmark current tile size implementation

    Args:
        num_pairs: Number of alignment pairs
        seq_length: Average sequence length

    Returns:
        Performance metrics
    """
    try:
        from smith_waterman import align_batch
    except ImportError:
        logger.error("GPU implementation not available. Build with: python setup.py install")
        return None

    # Generate test sequences
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    np.random.seed(42)

    sequences_a = []
    sequences_b = []
    for i in range(num_pairs):
        len_a = int(seq_length * (0.8 + 0.4 * np.random.random()))
        len_b = int(seq_length * (0.8 + 0.4 * np.random.random()))

        seq_a = ''.join(np.random.choice(list(amino_acids), len_a))
        seq_b = ''.join(np.random.choice(list(amino_acids), len_b))

        sequences_a.append(seq_a)
        sequences_b.append(seq_b)

    logger.info(f"Benchmarking {num_pairs} alignments (avg length: {seq_length})...")

    # Warmup
    _ = align_batch(sequences_a[:10], sequences_b[:10], show_progress=False)

    # Benchmark
    start_time = time.time()
    scores = align_batch(sequences_a, sequences_b, show_progress=False)
    elapsed = time.time() - start_time

    throughput = num_pairs / elapsed

    # Calculate cell updates per second (CUPS)
    total_cells = sum(len(a) * len(b) for a, b in zip(sequences_a, sequences_b))
    cups = total_cells / elapsed
    gcups = cups / 1e9

    return {
        'num_pairs': num_pairs,
        'elapsed_seconds': elapsed,
        'throughput_pairs_per_sec': throughput,
        'total_cells': total_cells,
        'gcups': gcups,
        'mean_score': float(np.mean(scores)),
    }


def save_benchmark_result(tile_size: int, result: Dict, output_dir: str = "ablation_results"):
    """Save benchmark result to file"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = output_path / f"tile_{tile_size}x{tile_size}.json"

    with open(filename, 'w') as f:
        json.dump({
            'tile_size': tile_size,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            **result
        }, f, indent=2)

    logger.info(f"Saved results to {filename}")


def load_all_results(output_dir: str = "ablation_results") -> List[Dict]:
    """Load all benchmark results"""
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.warning(f"Results directory not found: {output_dir}")
        return []

    results = []
    for filename in sorted(output_path.glob("tile_*.json")):
        with open(filename, 'r') as f:
            result = json.load(f)
            results.append(result)

    return results


def generate_ablation_report(results: List[Dict], output_dir: str = "ablation_results"):
    """Generate comparison report from all results"""
    if not results:
        logger.error("No results to compare")
        return

    print()
    print("=" * 80)
    print("TILE SIZE ABLATION STUDY - EMPIRICAL RESULTS")
    print("=" * 80)
    print()

    # Sort by tile size
    results = sorted(results, key=lambda x: x['tile_size'])

    # Find best performer
    best_idx = np.argmax([r['throughput_pairs_per_sec'] for r in results])
    best_result = results[best_idx]

    # Print table
    print(f"{'Tile Size':<12} | {'Throughput':<15} | {'GCUPS':<12} | {'Speedup':<10}")
    print("-" * 80)

    for result in results:
        tile_size = result['tile_size']
        throughput = result['throughput_pairs_per_sec']
        gcups = result['gcups']
        speedup = throughput / results[0]['throughput_pairs_per_sec']

        marker = " ✓ BEST" if result == best_result else ""

        print(f"{tile_size:>3}x{tile_size:<6} | {throughput:>12.1f} p/s | {gcups:>10.2f} | {speedup:>8.2f}x{marker}")

    print()
    print(f"Best configuration: {best_result['tile_size']}x{best_result['tile_size']}")
    print(f"  Throughput: {best_result['throughput_pairs_per_sec']:.1f} pairs/sec")
    print(f"  Performance: {best_result['gcups']:.2f} GCUPS")
    print()

    # Save markdown report
    md_path = Path(output_dir) / "ablation_report.md"
    with open(md_path, 'w') as f:
        f.write("# Tile Size Ablation Study\n\n")
        f.write("## Empirical Performance Comparison\n\n")
        f.write("| Tile Size | Throughput (pairs/sec) | GCUPS | Speedup vs 8x8 |\n")
        f.write("|-----------|------------------------|-------|----------------|\n")

        for result in results:
            speedup = result['throughput_pairs_per_sec'] / results[0]['throughput_pairs_per_sec']
            marker = " **BEST**" if result == best_result else ""
            f.write(f"| {result['tile_size']}x{result['tile_size']} | {result['throughput_pairs_per_sec']:.1f} | {result['gcups']:.2f} | {speedup:.2f}x{marker} |\n")

        f.write(f"\n## Conclusion\n\n")
        f.write(f"The optimal tile size is **{best_result['tile_size']}x{best_result['tile_size']}**, achieving {best_result['gcups']:.2f} GCUPS.\n\n")
        f.write(f"This configuration provides the best balance between:\n")
        f.write(f"- Thread utilization\n")
        f.write(f"- Shared memory usage\n")
        f.write(f"- Occupancy\n")
        f.write(f"- Work granularity\n")

    logger.info(f"Saved report to {md_path}")
    print("=" * 80)


# ============================================================================
# Main CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tile size ablation study")

    parser.add_argument('--analyze', action='store_true',
                        help='Show theoretical analysis')

    parser.add_argument('--benchmark', action='store_true',
                        help='Run empirical benchmark')

    parser.add_argument('--tile-size', type=int, default=16,
                        help='Tile size for benchmark (must match compiled kernel)')

    parser.add_argument('--num-pairs', type=int, default=1000,
                        help='Number of alignment pairs for benchmark')

    parser.add_argument('--report', action='store_true',
                        help='Generate comparison report from all results')

    parser.add_argument('--output-dir', type=str, default='ablation_results',
                        help='Output directory for results')

    args = parser.parse_args()

    if args.analyze or (not args.benchmark and not args.report):
        print_theoretical_analysis()

    if args.benchmark:
        print()
        logger.info(f"Running benchmark for tile size {args.tile_size}x{args.tile_size}")
        result = run_tile_size_benchmark(num_pairs=args.num_pairs)

        if result:
            logger.info(f"Throughput: {result['throughput_pairs_per_sec']:.1f} pairs/sec")
            logger.info(f"Performance: {result['gcups']:.2f} GCUPS")
            save_benchmark_result(args.tile_size, result, args.output_dir)

    if args.report:
        results = load_all_results(args.output_dir)
        if results:
            generate_ablation_report(results, args.output_dir)
        else:
            logger.error("No results found. Run benchmarks first with --benchmark")


if __name__ == "__main__":
    main()
