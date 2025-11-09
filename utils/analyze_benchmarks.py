#!/usr/bin/env python3
"""
Benchmark Results Aggregation and Analysis Script.

Analyzes CPU and GPU benchmark results to calculate speedups,
throughput comparisons, and memory efficiency metrics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
import sys


def load_benchmark(filepath: Path) -> Dict[str, Any]:
    """Load a benchmark JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_metrics(benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from benchmark data."""
    metadata = benchmark.get('metadata', {})
    batch_stats = benchmark.get('batch_stats', {})
    timers = benchmark.get('timers', {})
    memory_snapshots = benchmark.get('memory_snapshots', [])

    # Extract timing information
    total_time = benchmark.get('total_time', 0)

    # Extract throughput
    throughput = batch_stats.get('throughput', {})
    mean_throughput = throughput.get('mean', 0)

    # Extract batch time
    batch_time = batch_stats.get('batch_time', {})
    mean_batch_time = batch_time.get('mean', 0)

    # Extract forward time
    forward_time = batch_stats.get('forward_time', {})
    mean_forward_time = forward_time.get('mean', 0)

    # Extract memory info
    peak_memory = 0
    memory_snapshots = benchmark.get('memory_snapshots', [])
    for snapshot in memory_snapshots:
        if snapshot.get('label') == 'final':
            gpu_mem = snapshot.get('gpu_memory', {})
            peak_memory = gpu_mem.get('max_allocated_gb', 0)
            break

    # Get total proteins processed
    total_proteins = batch_stats.get('total_items', metadata.get('num_sequences', 0))

    return {
        'total_time_sec': total_time,
        'mean_batch_time_sec': mean_batch_time,
        'mean_forward_time_sec': mean_forward_time,
        'mean_throughput_proteins_per_sec': mean_throughput,
        'peak_memory_gb': peak_memory,
        'total_proteins': total_proteins,
        'device': metadata.get('device', 'unknown'),
        'model': metadata.get('model', 'unknown')
    }


def calculate_speedup(cpu_metrics: Dict, gpu_metrics: Dict) -> Dict[str, Any]:
    """Calculate speedup metrics comparing GPU vs CPU."""
    speedup_time = cpu_metrics['total_time_sec'] / gpu_metrics['total_time_sec'] if gpu_metrics['total_time_sec'] > 0 else 0
    speedup_throughput = gpu_metrics['mean_throughput_proteins_per_sec'] / cpu_metrics['mean_throughput_proteins_per_sec'] if cpu_metrics['mean_throughput_proteins_per_sec'] > 0 else 0
    speedup_batch = cpu_metrics['mean_batch_time_sec'] / gpu_metrics['mean_batch_time_sec'] if gpu_metrics['mean_batch_time_sec'] > 0 else 0

    return {
        'speedup_total_time': speedup_time,
        'speedup_throughput': speedup_throughput,
        'speedup_batch_time': speedup_batch,
        'cpu_time_sec': cpu_metrics['total_time_sec'],
        'gpu_time_sec': gpu_metrics['total_time_sec'],
        'cpu_throughput': cpu_metrics['mean_throughput_proteins_per_sec'],
        'gpu_throughput': gpu_metrics['mean_throughput_proteins_per_sec'],
        'gpu_peak_memory_gb': gpu_metrics['peak_memory_gb'],
        'memory_per_protein_mb': (gpu_metrics['peak_memory_gb'] * 1024) / gpu_metrics['total_proteins'] if gpu_metrics['total_proteins'] > 0 else 0
    }


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def generate_markdown_table(results: Dict[str, Dict]) -> str:
    """Generate markdown table from results."""
    lines = []
    lines.append("## Benchmark Results Summary\n")
    lines.append("| Model | CPU Time | GPU Time | Speedup | CPU Throughput | GPU Throughput | Peak GPU Memory |")
    lines.append("|-------|----------|----------|---------|----------------|----------------|-----------------|")

    for model, data in results.items():
        cpu_time = format_time(data['cpu_time_sec'])
        gpu_time = format_time(data['gpu_time_sec'])
        speedup = f"{data['speedup_total_time']:.1f}x"
        cpu_thr = f"{data['cpu_throughput']:.2f} p/s"
        gpu_thr = f"{data['gpu_throughput']:.2f} p/s"
        memory = f"{data['gpu_peak_memory_gb']:.2f} GB"

        lines.append(f"| {model} | {cpu_time} | {gpu_time} | {speedup} | {cpu_thr} | {gpu_thr} | {memory} |")

    lines.append("")
    lines.append("**Legend:** p/s = proteins per second")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--benchmark-dir', type=str, default='benchmark_results',
                       help='Directory containing benchmark JSON files')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for analysis results')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['esm2_3B', 'esm_c_600m', 'prot_t5_xl'],
                       help='Models to analyze')

    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1B: Benchmark Results Analysis")
    print("=" * 70)
    print(f"Benchmark directory: {benchmark_dir}")
    print(f"Models: {', '.join(args.models)}")
    print("")

    results = {}

    for model in args.models:
        print(f"Analyzing {model}...")

        # Load CPU and GPU benchmarks
        cpu_file = benchmark_dir / f"{model}_cpu_1k.json"
        gpu_file = benchmark_dir / f"{model}_gpu_1k.json"

        if not cpu_file.exists():
            print(f"  ⚠ Warning: {cpu_file} not found, skipping")
            continue
        if not gpu_file.exists():
            print(f"  ⚠ Warning: {gpu_file} not found, skipping")
            continue

        cpu_data = load_benchmark(cpu_file)
        gpu_data = load_benchmark(gpu_file)

        cpu_metrics = extract_metrics(cpu_data)
        gpu_metrics = extract_metrics(gpu_data)

        speedup_data = calculate_speedup(cpu_metrics, gpu_metrics)

        results[model] = speedup_data

        print(f"  ✓ CPU time: {format_time(speedup_data['cpu_time_sec'])}")
        print(f"  ✓ GPU time: {format_time(speedup_data['gpu_time_sec'])}")
        print(f"  ✓ Speedup: {speedup_data['speedup_total_time']:.2f}x")
        print(f"  ✓ GPU throughput: {speedup_data['gpu_throughput']:.2f} proteins/sec")
        print("")

    if not results:
        print("❌ No benchmark results found!")
        sys.exit(1)

    # Calculate overall statistics
    avg_speedup = sum(r['speedup_total_time'] for r in results.values()) / len(results)
    max_speedup = max(r['speedup_total_time'] for r in results.values())
    min_speedup = min(r['speedup_total_time'] for r in results.values())

    print("=" * 70)
    print("Overall Statistics")
    print("=" * 70)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")
    print(f"Minimum speedup: {min_speedup:.2f}x")
    print("")

    # Save results
    summary_json = output_dir / "benchmark_summary.json"
    with open(summary_json, 'w') as f:
        json.dump({
            'models': results,
            'overall': {
                'avg_speedup': avg_speedup,
                'max_speedup': max_speedup,
                'min_speedup': min_speedup,
                'num_models': len(results)
            }
        }, f, indent=2)
    print(f"✓ Saved JSON summary: {summary_json}")

    # Generate markdown table
    markdown_table = generate_markdown_table(results)
    table_file = output_dir / "benchmark_comparison.md"
    with open(table_file, 'w') as f:
        f.write(markdown_table)
    print(f"✓ Saved markdown table: {table_file}")

    print("")
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
