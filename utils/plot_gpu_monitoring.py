#!/usr/bin/env python3
"""
Visualize GPU monitoring data from Phase 1A embedding generation.

Generates comprehensive plots showing GPU utilization, memory, temperature,
and power consumption over time.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def setup_plot_style():
    """Setup publication-quality plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 10)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 14


def load_monitoring_data(csv_path: str) -> pd.DataFrame:
    """Load and preprocess GPU monitoring CSV."""
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Convert elapsed seconds to minutes for better readability
    df['elapsed_minutes'] = df['elapsed_seconds'] / 60

    return df


def plot_gpu_utilization(df: pd.DataFrame, ax):
    """Plot GPU utilization over time."""
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        ax.plot(gpu_data['elapsed_minutes'],
                gpu_data['gpu_util_pct'],
                label=f'GPU {gpu_id}',
                linewidth=2,
                alpha=0.8)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    # Add horizontal line at target utilization (50-60%)
    ax.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='Target Min (50%)')
    ax.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Target Max (60%)')


def plot_gpu_memory(df: pd.DataFrame, ax):
    """Plot GPU memory usage over time."""
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        ax.plot(gpu_data['elapsed_minutes'],
                gpu_data['gpu_mem_used_mb'] / 1024,  # Convert to GB
                label=f'GPU {gpu_id}',
                linewidth=2,
                alpha=0.8)

    # Add total memory line
    total_mem_gb = df['gpu_mem_total_mb'].iloc[0] / 1024
    ax.axhline(y=total_mem_gb, color='red', linestyle='--',
               alpha=0.5, label=f'Total Memory ({total_mem_gb:.0f} GB)')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('GPU Memory Used (GB)')
    ax.set_title('GPU Memory Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, total_mem_gb * 1.1])


def plot_gpu_temperature(df: pd.DataFrame, ax):
    """Plot GPU temperature over time."""
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        ax.plot(gpu_data['elapsed_minutes'],
                gpu_data['gpu_temp_c'],
                label=f'GPU {gpu_id}',
                linewidth=2,
                alpha=0.8)

    # Add safe temperature range
    ax.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Warning (80°C)')
    ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Critical (90°C)')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('GPU Temperature Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_gpu_power(df: pd.DataFrame, ax):
    """Plot GPU power consumption over time."""
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        ax.plot(gpu_data['elapsed_minutes'],
                gpu_data['gpu_power_draw_w'],
                label=f'GPU {gpu_id}',
                linewidth=2,
                alpha=0.8)

    # Add power limit line
    power_limit = df['gpu_power_limit_w'].iloc[0]
    ax.axhline(y=power_limit, color='red', linestyle='--',
               alpha=0.5, label=f'Power Limit ({power_limit:.0f}W)')

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Power Draw (W)')
    ax.set_title('GPU Power Consumption Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, power_limit * 1.1])


def plot_memory_utilization(df: pd.DataFrame, ax):
    """Plot GPU memory utilization percentage."""
    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]
        ax.plot(gpu_data['elapsed_minutes'],
                gpu_data['gpu_mem_util_pct'],
                label=f'GPU {gpu_id}',
                linewidth=2,
                alpha=0.8)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Memory Utilization (%)')
    ax.set_title('GPU Memory Utilization Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])


def plot_cpu_ram(df: pd.DataFrame, ax):
    """Plot CPU and RAM usage."""
    # Use first GPU's data (system metrics are same for all)
    system_data = df[df['gpu_id'] == 0]

    ax2 = ax.twinx()

    # CPU utilization on left axis
    line1 = ax.plot(system_data['elapsed_minutes'],
                    system_data['cpu_util_pct'],
                    label='CPU Utilization',
                    color='blue',
                    linewidth=2,
                    alpha=0.8)

    # RAM utilization on right axis
    line2 = ax2.plot(system_data['elapsed_minutes'],
                     system_data['ram_util_pct'],
                     label='RAM Utilization',
                     color='green',
                     linewidth=2,
                     alpha=0.8)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('CPU Utilization (%)', color='blue')
    ax2.set_ylabel('RAM Utilization (%)', color='green')
    ax.set_title('System Resource Usage')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.grid(True, alpha=0.3)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calculate summary statistics for the monitoring data."""
    stats = {}

    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]

        stats[f'gpu_{gpu_id}'] = {
            'mean_util': gpu_data['gpu_util_pct'].mean(),
            'max_util': gpu_data['gpu_util_pct'].max(),
            'mean_mem_util': gpu_data['gpu_mem_util_pct'].mean(),
            'max_mem_gb': (gpu_data['gpu_mem_used_mb'].max() / 1024),
            'mean_temp': gpu_data['gpu_temp_c'].mean(),
            'max_temp': gpu_data['gpu_temp_c'].max(),
            'mean_power': gpu_data['gpu_power_draw_w'].mean(),
            'max_power': gpu_data['gpu_power_draw_w'].max(),
        }

    # System stats (from GPU 0 data)
    system_data = df[df['gpu_id'] == 0]
    stats['system'] = {
        'mean_cpu_util': system_data['cpu_util_pct'].mean(),
        'mean_ram_util': system_data['ram_util_pct'].mean(),
        'duration_minutes': system_data['elapsed_minutes'].max(),
    }

    return stats


def print_statistics(stats: dict):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("GPU Monitoring Statistics")
    print("=" * 70)

    for gpu_key, gpu_stats in stats.items():
        if gpu_key.startswith('gpu_'):
            gpu_id = gpu_key.split('_')[1]
            print(f"\nGPU {gpu_id}:")
            print(f"  Utilization:       Mean={gpu_stats['mean_util']:.1f}%, Max={gpu_stats['max_util']:.1f}%")
            print(f"  Memory Utilization: Mean={gpu_stats['mean_mem_util']:.1f}%, Max={gpu_stats['max_mem_gb']:.1f} GB")
            print(f"  Temperature:       Mean={gpu_stats['mean_temp']:.1f}°C, Max={gpu_stats['max_temp']:.1f}°C")
            print(f"  Power Draw:        Mean={gpu_stats['mean_power']:.1f}W, Max={gpu_stats['max_power']:.1f}W")

    print(f"\nSystem:")
    print(f"  CPU Utilization:   Mean={stats['system']['mean_cpu_util']:.1f}%")
    print(f"  RAM Utilization:   Mean={stats['system']['mean_ram_util']:.1f}%")
    print(f"  Total Duration:    {stats['system']['duration_minutes']:.1f} minutes")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Visualize GPU monitoring data')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file from GPU monitoring')
    parser.add_argument('--output', type=str, default='figures/gpu_monitoring.png',
                       help='Output figure path')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Figure DPI (default: 300)')

    args = parser.parse_args()

    print(f"Loading data from: {args.input}")
    df = load_monitoring_data(args.input)

    print(f"Loaded {len(df)} rows, {len(df['gpu_id'].unique())} GPUs")
    print(f"Time range: {df['elapsed_minutes'].min():.1f} - {df['elapsed_minutes'].max():.1f} minutes")

    # Calculate and print statistics
    stats = calculate_statistics(df)
    print_statistics(stats)

    # Setup plot style
    setup_plot_style()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('GPU Monitoring Dashboard - Phase 1A Embedding Generation',
                 fontsize=16, fontweight='bold')

    # Plot each metric
    plot_gpu_utilization(df, axes[0, 0])
    plot_gpu_memory(df, axes[0, 1])
    plot_memory_utilization(df, axes[1, 0])
    plot_gpu_temperature(df, axes[1, 1])
    plot_gpu_power(df, axes[2, 0])
    plot_cpu_ram(df, axes[2, 1])

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    # Save figure
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving figure to: {args.output}")
    plt.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
    print(f"✓ Figure saved successfully ({args.dpi} DPI)")

    # Also save as lower resolution for quick viewing
    quick_view_path = output_path.parent / (output_path.stem + '_preview.png')
    plt.savefig(quick_view_path, dpi=150, bbox_inches='tight')
    print(f"✓ Preview saved: {quick_view_path}")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
