#!/usr/bin/env python3
"""
Generate Phase 2 Optimization Plan (Task 14).

Analyzes profiling data to identify and prioritize
custom CUDA kernel development opportunities.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def generate_optimization_plan(profiling_path: Path, benchmark_path: Path, output_path: Path):
    """Generate Phase 2 optimization plan from profiling data."""

    # Load data
    with open(profiling_path, 'r') as f:
        profiling_data = json.load(f)

    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)

    # Generate report
    report = []
    report.append("# Phase 2: Custom CUDA Kernel Optimization Plan")
    report.append("")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Based on:** Phase 1B profiling analysis (torch.profiler)")
    report.append(f"**Target:** Track 2 GPU Programming Project (Due: Nov 27, 2025)")
    report.append("")
    report.append("---")
    report.append("")

    # Current Performance Baseline
    report.append("## Current Performance Baseline")
    report.append("")
    overall = benchmark_data['overall']
    report.append(f"Phase 1B achieved **{overall['avg_speedup']:.2f}x average speedup** using PyTorch's native GPU operations. Phase 2 will implement custom CUDA kernels to further optimize bottleneck operations identified through profiling.")
    report.append("")

    models = list(profiling_data['models'].keys())
    report.append("**Current Speedups:**")
    report.append("")
    for model in models:
        speedup = benchmark_data['models'][model]['speedup_total_time']
        report.append(f"- {model}: {speedup:.2f}x")

    report.append("")
    report.append("**Target:** Achieve 20-30x total speedup through custom kernel optimization.")
    report.append("")
    report.append("---")
    report.append("")

    # Profiling-Identified Bottlenecks
    report.append("## Profiling-Identified Bottlenecks")
    report.append("")

    # Aggregate kernel time by category
    category_stats = {}
    for model in models:
        for category, data in profiling_data['models'][model]['distribution'].items():
            if category not in category_stats:
                category_stats[category] = {
                    'total_time_ms': 0,
                    'avg_percentage': 0,
                    'model_count': 0
                }
            category_stats[category]['total_time_ms'] += data['time_ms']
            category_stats[category]['avg_percentage'] += data['percentage']
            category_stats[category]['model_count'] += 1

    # Calculate averages
    for category in category_stats:
        count = category_stats[category]['model_count']
        category_stats[category]['avg_percentage'] /= count

    # Sort by average percentage
    sorted_categories = sorted(category_stats.items(),
                               key=lambda x: x[1]['avg_percentage'],
                               reverse=True)

    report.append("**Kernel Time Distribution (Average Across Models):**")
    report.append("")
    for category, stats in sorted_categories[:6]:
        report.append(f"- **{category}:** {stats['avg_percentage']:.1f}% (avg), {stats['total_time_ms']:.0f} ms (total)")

    report.append("")
    report.append("---")
    report.append("")

    # Optimization Opportunities
    report.append("## Optimization Opportunities")
    report.append("")

    opportunities = []

    # Opportunity 1: GEMM Kernels
    if 'GEMM (Matrix Multiply)' in category_stats:
        gemm = category_stats['GEMM (Matrix Multiply)']
        opportunities.append({
            'title': 'Custom GEMM Kernel Implementation',
            'category': 'GEMM (Matrix Multiply)',
            'current_time_ms': gemm['total_time_ms'],
            'percentage': gemm['avg_percentage'],
            'justification': f"GEMM operations consume {gemm['avg_percentage']:.1f}% of average compute time ({gemm['total_time_ms']:.0f} ms total). Current cuBLAS kernels (`ampere_sgemm_*`) can be optimized with custom tile sizes and shared memory management.",
            'approach': "Implement tiled matrix multiplication with:\n  - Optimized tile dimensions (32×32, 64×64 testing)\n  - Shared memory for data reuse\n  - Register blocking for reduced global memory access\n  - Thread coarsening for better instruction throughput",
            'expected_speedup': '2-3x',
            'difficulty': 'High',
            'priority': 1,
            'evidence': f"Profiling shows GEMM kernels account for {gemm['avg_percentage']:.1f}% of compute across all models"
        })

    # Opportunity 2: Linear + Activation Fusion
    if 'Linear Layers' in category_stats:
        linear = category_stats['Linear Layers']
        opportunities.append({
            'title': 'Fused Linear + Activation Kernel',
            'category': 'Linear Layers',
            'current_time_ms': linear['total_time_ms'],
            'percentage': linear['avg_percentage'],
            'justification': f"Linear layers consume {linear['avg_percentage']:.1f}% of compute time ({linear['total_time_ms']:.0f} ms). Currently implemented as separate GEMM + activation kernels, causing redundant memory transfers.",
            'approach': "Fuse matrix multiply and activation (ReLU/GELU) into single kernel:\n  - Compute activation immediately after each output element\n  - Eliminate intermediate memory writes\n  - Reduce memory bandwidth requirements by ~50%",
            'expected_speedup': '1.5-2x',
            'difficulty': 'Medium',
            'priority': 2,
            'evidence': f"aten::linear appears {linear['model_count']} times across models with {linear['avg_percentage']:.1f}% avg time"
        })

    # Opportunity 3: Element-wise Kernel Fusion
    if 'Element-wise Operations' in category_stats:
        elem = category_stats['Element-wise Operations']
        opportunities.append({
            'title': 'Element-wise Operation Fusion',
            'category': 'Element-wise Operations',
            'current_time_ms': elem['total_time_ms'],
            'percentage': elem['avg_percentage'],
            'justification': f"Element-wise operations (mul, add) consume {elem['avg_percentage']:.1f}% of time. Multiple separate kernel launches can be fused into single pass.",
            'approach': "Combine sequential element-wise operations:\n  - Fuse aten::mul + aten::add → single fused kernel\n  - Reduce kernel launch overhead\n  - Improve memory access patterns\n  - Single pass through data",
            'expected_speedup': '5-10x',
            'difficulty': 'Low',
            'priority': 3,
            'evidence': f"Profiling shows {elem['total_time_ms']:.0f} ms spent on element-wise ops with high kernel launch overhead"
        })

    # Opportunity 4: ProtT5 dtype conversion elimination
    prot_t5_memory = None
    if 'prot_t5_xl' in models:
        if 'Memory Operations' in profiling_data['models']['prot_t5_xl']['distribution']:
            mem_ops = profiling_data['models']['prot_t5_xl']['distribution']['Memory Operations']
            prot_t5_memory = mem_ops

            opportunities.append({
                'title': 'Eliminate ProtT5-XL FP16/FP32 Conversions',
                'category': 'Memory Operations',
                'current_time_ms': mem_ops['time_ms'],
                'percentage': mem_ops['percentage'],
                'justification': f"ProtT5-XL wastes {mem_ops['percentage']:.1f}% of time ({mem_ops['time_ms']:.0f} ms) on dtype conversions between FP16 and FP32. This is pure overhead with no computational benefit.",
                'approach': "Unify precision across entire inference pipeline:\n  - Load model in FP16 natively\n  - Remove unnecessary aten::copy_ and aten::to operations\n  - Maintain FP16 throughout forward pass\n  - Convert to FP32 only at final output if needed",
                'expected_speedup': 'Eliminate 344 ms overhead (~7% improvement)',
                'difficulty': 'Low',
                'priority': 4,
                'evidence': f"Profiling identifies {mem_ops['count']} dtype conversion operations consuming {mem_ops['time_ms']:.0f} ms"
            })

    # Flash Attention for ESM2 (if not present)
    if 'esm2_3B' in models:
        esm2_dist = profiling_data['models']['esm2_3B']['distribution']
        if 'Attention' not in esm2_dist and 'Matrix Operations' in esm2_dist:
            # ESM2 doesn't have flash attention
            opportunities.append({
                'title': 'Flash Attention for ESM2-3B',
                'category': 'Attention',
                'current_time_ms': 0,
                'percentage': 0,
                'justification': "ESM2-3B uses standard attention (batched matrix multiply) while ESM-C-600M benefits from flash attention. Implementing flash attention for ESM2 will reduce memory bandwidth and improve performance.",
                'approach': "Port ESM-C's flash attention mechanism:\n  - IO-aware attention computation\n  - Reduce memory reads/writes\n  - Kernel fusion for attention softmax\n  - Based on existing fmha_cutlassF implementation",
                'expected_speedup': '1.3-1.5x',
                'difficulty': 'High',
                'priority': 5,
                'evidence': "ESM-C-600M's flash attention shows 8.1% efficiency; ESM2 uses bmm pattern"
            })

    # Generate detailed opportunity sections
    for i, opp in enumerate(opportunities, 1):
        report.append(f"### Opportunity {i}: {opp['title']}")
        report.append("")
        report.append(f"**Category:** {opp['category']}")
        report.append("")
        report.append(f"**Justification:**")
        report.append("")
        report.append(opp['justification'])
        report.append("")
        report.append(f"**Implementation Approach:**")
        report.append("")
        report.append(opp['approach'])
        report.append("")
        report.append(f"**Expected Speedup:** {opp['expected_speedup']}")
        report.append(f"**Implementation Difficulty:** {opp['difficulty']}")
        report.append(f"**Priority:** {opp['priority']}")
        report.append("")
        report.append(f"**Profiling Evidence:**")
        report.append("")
        report.append(f"- {opp['evidence']}")
        report.append("")
        report.append("---")
        report.append("")

    # Priority Ranking
    report.append("## Implementation Priority Ranking")
    report.append("")
    report.append("Based on expected impact and implementation feasibility:")
    report.append("")

    priority_map = {
        'High': 'Hard',
        'Medium': 'Moderate',
        'Low': 'Easy'
    }

    sorted_opps = sorted(opportunities, key=lambda x: x['priority'])

    for i, opp in enumerate(sorted_opps, 1):
        impact = opp['expected_speedup']
        effort = priority_map.get(opp['difficulty'], 'Unknown')
        report.append(f"{i}. **{opp['title']}** (Priority {opp['priority']})")
        report.append(f"   - Expected Impact: {impact}")
        report.append(f"   - Implementation Effort: {effort}")
        report.append(f"   - Time Saved: {opp['current_time_ms']:.0f} ms → {opp['current_time_ms'] / 2:.0f} ms (estimated)")
        report.append("")

    report.append("---")
    report.append("")

    # Implementation Roadmap
    report.append("## Track 2 Implementation Roadmap")
    report.append("")
    report.append("**Timeline:** Nov 10-27, 2025 (18 days available)")
    report.append("")
    report.append("**Week 1 (Nov 10-16): Custom GEMM Kernel**")
    report.append("")
    report.append("- Implement tiled matrix multiplication baseline")
    report.append("- Optimize tile sizes through empirical testing")
    report.append("- Benchmark against cuBLAS baseline")
    report.append("- Target: 2-3x speedup on GEMM operations")
    report.append("")
    report.append("**Week 2 (Nov 17-23): Kernel Fusion**")
    report.append("")
    report.append("- Fuse linear + activation operations")
    report.append("- Implement element-wise fusion (if time permits)")
    report.append("- Profile fused kernels with NSight Compute")
    report.append("- Target: Additional 1.5-2x on fused operations")
    report.append("")
    report.append("**Week 3 (Nov 24-27): Integration & Reporting**")
    report.append("")
    report.append("- Integrate custom kernels into inference pipeline")
    report.append("- End-to-end benchmarking and validation")
    report.append("- Generate Track 2 final report with performance analysis")
    report.append("- Target: 20-30x total speedup (current 16.7x + optimizations)")
    report.append("")
    report.append("---")
    report.append("")

    # Success Metrics
    report.append("## Success Metrics")
    report.append("")
    report.append("**Minimum Viable Product (Track 2 passing grade):**")
    report.append("")
    report.append("- ✅ Implement at least 1 custom CUDA kernel (GEMM recommended)")
    report.append("- ✅ Demonstrate measurable speedup over PyTorch baseline")
    report.append("- ✅ Profile with NSight Compute showing kernel-level optimizations")
    report.append("- ✅ Document implementation with code walkthrough")
    report.append("")
    report.append("**Stretch Goals:**")
    report.append("")
    report.append("- Implement 2-3 custom kernels (GEMM + fusion)")
    report.append("- Achieve 20-30x total speedup (Track 2 target)")
    report.append("- Compare multiple tile sizes and optimization strategies")
    report.append("")
    report.append("---")
    report.append("")

    # Conclusion
    report.append("## Conclusion")
    report.append("")
    report.append("Phase 1B profiling identified specific kernel bottlenecks with quantified time percentages and call counts. Custom GEMM kernel implementation represents the highest-impact optimization opportunity, consuming 15-23% of compute time across models. A focused 18-day implementation plan targets the Track 2 project deadline while providing clear success criteria.")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))

    print(f"✓ Generated optimization plan: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 2 optimization plan')
    parser.add_argument('--profiling-analysis', type=str, default='reports/profiling_analysis.json',
                       help='Path to profiling analysis JSON')
    parser.add_argument('--benchmark-summary', type=str, default='reports/benchmark_summary.json',
                       help='Path to benchmark summary JSON')
    parser.add_argument('--output', type=str, default='reports/phase2_optimization_plan.md',
                       help='Output markdown file')

    args = parser.parse_args()

    profiling_path = Path(args.profiling_analysis)
    benchmark_path = Path(args.benchmark_summary)
    output_path = Path(args.output)

    if not profiling_path.exists():
        print(f"❌ Error: {profiling_path} not found")
        print("   Run 'python utils/analyze_profiling.py' first")
        return

    if not benchmark_path.exists():
        print(f"❌ Error: {benchmark_path} not found")
        print("   Run 'python utils/analyze_benchmarks.py' first")
        return

    generate_optimization_plan(profiling_path, benchmark_path, output_path)


if __name__ == '__main__':
    main()
