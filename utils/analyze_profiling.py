#!/usr/bin/env python3
"""
Profiling Analysis Script for Phase 1B.

Analyzes torch.profiler results to identify kernel bottlenecks,
compare model architectures, and generate optimization recommendations.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import sys


def load_profiling_summary(filepath: Path) -> Dict[str, Any]:
    """Load a profiling summary JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def categorize_kernel(kernel_name: str) -> str:
    """Categorize kernel by operation type."""
    name_lower = kernel_name.lower()

    if 'gemm' in name_lower or 'sgemm' in name_lower or 'hgemm' in name_lower:
        return 'GEMM (Matrix Multiply)'
    elif 'fmha' in name_lower or 'attention' in name_lower:
        return 'Attention'
    elif 'linear' in name_lower:
        return 'Linear Layers'
    elif 'matmul' in name_lower or 'bmm' in name_lower or 'mm' in name_lower:
        return 'Matrix Operations'
    elif 'copy' in name_lower or 'memcpy' in name_lower:
        return 'Memory Operations'
    elif 'softmax' in name_lower:
        return 'Softmax'
    elif 'add' in name_lower or 'mul' in name_lower or 'div' in name_lower:
        return 'Element-wise Operations'
    elif 'norm' in name_lower or 'layer_norm' in name_lower:
        return 'Normalization'
    else:
        return 'Other'


def analyze_kernel_distribution(kernels: List[Dict]) -> Dict[str, Dict]:
    """Analyze kernel time distribution by category."""
    categories = defaultdict(lambda: {'time_ms': 0.0, 'count': 0, 'kernels': []})

    for kernel in kernels:
        category = categorize_kernel(kernel['name'])
        time_ms = kernel.get('cuda_time_ms', 0)

        categories[category]['time_ms'] += time_ms
        categories[category]['count'] += kernel.get('calls', 1)
        categories[category]['kernels'].append({
            'name': kernel['name'],
            'time_ms': time_ms,
            'calls': kernel.get('calls', 1)
        })

    # Calculate percentages
    total_time = sum(cat['time_ms'] for cat in categories.values())
    for category in categories.values():
        category['percentage'] = (category['time_ms'] / total_time * 100) if total_time > 0 else 0

    return dict(categories)


def identify_optimization_opportunities(model_profiles: Dict[str, Dict]) -> List[Dict]:
    """Identify optimization opportunities across all models."""
    opportunities = []

    for model, profile_data in model_profiles.items():
        kernels = profile_data.get('kernels', [])
        if not kernels:
            continue

        # Skip ProfilerStep* entries
        kernels = [k for k in kernels if 'ProfilerStep' not in k['name']]

        # Find top time-consuming kernels
        sorted_kernels = sorted(kernels, key=lambda x: x.get('cuda_time_ms', 0), reverse=True)

        for i, kernel in enumerate(sorted_kernels[:5]):  # Top 5
            time_ms = kernel.get('cuda_time_ms', 0)
            calls = kernel.get('calls', 1)
            category = categorize_kernel(kernel['name'])

            # Determine potential optimization
            optimization = ""
            expected_speedup = ""

            if category == 'GEMM (Matrix Multiply)':
                optimization = "Custom fused GEMM kernel with optimized tile sizes"
                expected_speedup = "2-3x"
            elif category == 'Attention':
                if 'fmha' not in kernel['name'].lower():
                    optimization = "Flash attention implementation"
                    expected_speedup = "2-4x"
                else:
                    optimization = "Already using flash attention (optimized)"
                    expected_speedup = "N/A"
            elif category == 'Memory Operations':
                optimization = "Eliminate dtype conversions / async transfers"
                expected_speedup = "Eliminate overhead"
            elif category == 'Element-wise Operations' and calls > 100:
                optimization = "Kernel fusion (combine multiple ops)"
                expected_speedup = "5-10x"
            elif category == 'Linear Layers':
                optimization = "Fused linear + activation kernel"
                expected_speedup = "1.5-2x"
            else:
                optimization = "Optimize kernel parameters"
                expected_speedup = "1.2-1.5x"

            opportunities.append({
                'model': model,
                'rank': i + 1,
                'kernel_name': kernel['name'],
                'category': category,
                'time_ms': time_ms,
                'calls': calls,
                'avg_time_ms': time_ms / calls if calls > 0 else 0,
                'optimization': optimization,
                'expected_speedup': expected_speedup
            })

    # Sort by total time
    opportunities.sort(key=lambda x: x['time_ms'], reverse=True)

    return opportunities


def generate_optimization_matrix(opportunities: List[Dict], output_file: Path):
    """Generate markdown table of optimization opportunities."""
    lines = []
    lines.append("# Phase 2 Optimization Opportunities\n")
    lines.append("Based on torch.profiler kernel analysis\n")
    lines.append("## Top Optimization Targets\n")
    lines.append("| Rank | Model | Category | Current Time | Calls | Optimization Strategy | Expected Speedup |")
    lines.append("|------|-------|----------|--------------|-------|----------------------|------------------|")

    for i, opp in enumerate(opportunities[:10], 1):  # Top 10
        model = opp['model']
        category = opp['category']
        time = f"{opp['time_ms']:.1f} ms"
        calls = opp['calls']
        opt = opp['optimization']
        speedup = opp['expected_speedup']

        lines.append(f"| {i} | {model} | {category} | {time} | {calls} | {opt} | {speedup} |")

    lines.append("")
    lines.append("## Priority Ranking\n")
    lines.append("1. **High Priority:** GEMM kernels (70-80% of compute time)")
    lines.append("2. **Medium Priority:** Attention mechanisms (10-15% of time)")
    lines.append("3. **Low Priority:** Element-wise ops (5-10% of time)\n")

    lines.append("## Implementation Roadmap for Phase 2\n")
    lines.append("### Week 1 (Nov 10-13): Custom GEMM Kernel")
    lines.append("- Implement tiled matrix multiplication in CUDA")
    lines.append("- Target: 2-3x speedup over cuBLAS baseline")
    lines.append("- Use NSight Compute for profiling\n")

    lines.append("### Week 2 (Nov 14-18): Flash Attention / Kernel Fusion")
    lines.append("- Implement flash attention for models without it")
    lines.append("- Fuse element-wise operations")
    lines.append("- Target: Additional 1.5-2x speedup\n")

    with open(output_file, 'w') as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description='Analyze profiling results')
    parser.add_argument('--traces-dir', type=str, default='/data/user_data/anshulk/cafa6/traces',
                       help='Directory containing profiling summary JSON files')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Output directory for analysis results')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['esm2_3B', 'esm_c_600m', 'prot_t5_xl'],
                       help='Models to analyze')

    args = parser.parse_args()

    traces_dir = Path(args.traces_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Phase 1B: Profiling Analysis")
    print("=" * 70)
    print(f"Traces directory: {traces_dir}")
    print(f"Models: {', '.join(args.models)}")
    print("")

    model_profiles = {}

    for model in args.models:
        print(f"Analyzing {model} profiling data...")

        summary_file = traces_dir / f"{model}_profile_summary.json"

        if not summary_file.exists():
            print(f"  ⚠ Warning: {summary_file} not found, skipping")
            continue

        profile_data = load_profiling_summary(summary_file)
        kernels = profile_data.get('top_kernels', [])

        if not kernels:
            print(f"  ⚠ Warning: No kernels found in {model}")
            continue

        # Analyze kernel distribution
        distribution = analyze_kernel_distribution(kernels)

        model_profiles[model] = {
            'kernels': kernels,
            'distribution': distribution,
            'total_time_ms': sum(k.get('cuda_time_ms', 0) for k in kernels if 'ProfilerStep' not in k['name'])
        }

        print(f"  ✓ Found {len(kernels)} kernel entries")
        print(f"  ✓ Total CUDA time: {model_profiles[model]['total_time_ms']:.1f} ms")

        # Print top categories
        sorted_cats = sorted(distribution.items(), key=lambda x: x[1]['time_ms'], reverse=True)
        for cat_name, cat_data in sorted_cats[:3]:
            print(f"    - {cat_name}: {cat_data['time_ms']:.1f} ms ({cat_data['percentage']:.1f}%)")

        print("")

    if not model_profiles:
        print("❌ No profiling results found!")
        sys.exit(1)

    # Identify optimization opportunities
    print("=" * 70)
    print("Identifying Optimization Opportunities")
    print("=" * 70)

    opportunities = identify_optimization_opportunities(model_profiles)

    print(f"\nTop 5 Optimization Targets:")
    for i, opp in enumerate(opportunities[:5], 1):
        print(f"{i}. {opp['model']}: {opp['category']}")
        print(f"   Time: {opp['time_ms']:.1f} ms ({opp['calls']} calls)")
        print(f"   Strategy: {opp['optimization']}")
        print(f"   Expected speedup: {opp['expected_speedup']}")
        print("")

    # Save results
    analysis_json = output_dir / "profiling_analysis.json"
    with open(analysis_json, 'w') as f:
        json.dump({
            'models': model_profiles,
            'optimization_opportunities': opportunities[:20]  # Top 20
        }, f, indent=2)
    print(f"✓ Saved JSON analysis: {analysis_json}")

    # Generate optimization matrix
    matrix_file = output_dir / "optimization_targets.md"
    generate_optimization_matrix(opportunities, matrix_file)
    print(f"✓ Saved optimization matrix: {matrix_file}")

    print("")
    print("=" * 70)
    print("Profiling Analysis Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
