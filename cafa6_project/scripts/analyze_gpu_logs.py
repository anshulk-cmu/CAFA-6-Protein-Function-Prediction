import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Analyze GPU monitoring logs')
parser.add_argument('--log', type=str, default='../outputs/gpu_monitor_phase1.csv',
                   help='Path to GPU monitor CSV log file')
parser.add_argument('--output', type=str, default='../outputs/gpu_analysis.png',
                   help='Output path for visualization')
parser.add_argument('--interval', type=int, default=5,
                   help='Monitoring interval in seconds (default: 5)')
args = parser.parse_args()

# Check if log file exists
log_path = Path(args.log)
if not log_path.exists():
    print(f"ERROR: Log file not found: {log_path}")
    print(f"\nMake sure gpu_monitor.py is running and generating logs.")
    sys.exit(1)

# Read CSV
try:
    df = pd.read_csv(log_path)
except Exception as e:
    print(f"ERROR: Failed to read log file: {e}")
    sys.exit(1)

# Check for required columns
required_cols = ['timestamp', 'gpu_mem_gb', 'gpu_util_pct', 'gpu_temp_c']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing required columns: {missing_cols}")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['elapsed_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

print("=" * 50)
print("GPU MONITORING SUMMARY")
print("=" * 50)
print(f"Log file: {log_path}")
print(f"\nDuration: {df.shape[0] * args.interval / 3600:.2f} hours")
print(f"Total Samples: {df.shape[0]:,}")
print(f"\n--- GPU Memory ---")
print(f"Average: {df['gpu_mem_gb'].mean():.2f} GB")
print(f"Peak: {df['gpu_mem_gb'].max():.2f} GB")
print(f"Min: {df['gpu_mem_gb'].min():.2f} GB")
print(f"\n--- GPU Utilization ---")
print(f"Average: {df['gpu_util_pct'].mean():.1f}%")
print(f"Peak: {df['gpu_util_pct'].max():.1f}%")
print(f"Min: {df['gpu_util_pct'].min():.1f}%")

# Detect thermal throttling
thermal_threshold = 80
thermal_warnings = df[df['gpu_temp_c'] > thermal_threshold]
print(f"\n--- Temperature ---")
print(f"Average: {df['gpu_temp_c'].mean():.1f}°C")
print(f"Peak: {df['gpu_temp_c'].max():.1f}°C")
print(f"Min: {df['gpu_temp_c'].min():.1f}°C")
if len(thermal_warnings) > 0:
    print(f"⚠ WARNING: Temperature exceeded {thermal_threshold}°C for {len(thermal_warnings)} samples ({len(thermal_warnings)/len(df)*100:.1f}%)")

# Check for optional columns
if 'cpu_pct' in df.columns:
    print(f"\n--- System Resources ---")
    print(f"Avg CPU: {df['cpu_pct'].mean():.1f}%")
    if 'ram_gb' in df.columns:
        print(f"Avg RAM: {df['ram_gb'].mean():.1f} GB")

# Memory efficiency
if df['gpu_mem_gb'].max() > 0:
    print(f"\n--- Memory Efficiency ---")
    print(f"Memory utilization: {df['gpu_mem_gb'].mean() / df['gpu_mem_gb'].max() * 100:.1f}% (avg/peak)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df['elapsed_hours'], df['gpu_mem_gb'], color='blue', linewidth=1)
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('GPU Memory (GB)')
axes[0, 0].set_title('GPU Memory Usage Over Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=df['gpu_mem_gb'].mean(), color='red', linestyle='--', alpha=0.5, label=f"Avg: {df['gpu_mem_gb'].mean():.1f}GB")
axes[0, 0].legend()

axes[0, 1].plot(df['elapsed_hours'], df['gpu_util_pct'], color='green', linewidth=1)
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('GPU Utilization (%)')
axes[0, 1].set_title('GPU Utilization Over Time')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=df['gpu_util_pct'].mean(), color='red', linestyle='--', alpha=0.5, label=f"Avg: {df['gpu_util_pct'].mean():.0f}%")
axes[0, 1].legend()

axes[1, 0].plot(df['elapsed_hours'], df['gpu_temp_c'], color='orange', linewidth=1)
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Temperature (°C)')
axes[1, 0].set_title('GPU Temperature Over Time')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Thermal Warning (80°C)')
axes[1, 0].legend()

# CPU vs GPU comparison (optional columns)
if 'cpu_pct' in df.columns:
    axes[1, 1].plot(df['elapsed_hours'], df['cpu_pct'], color='purple', alpha=0.7, label='CPU %')
    axes[1, 1].plot(df['elapsed_hours'], df['gpu_util_pct'], color='green', alpha=0.7, label='GPU %')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Utilization (%)')
    axes[1, 1].set_title('CPU vs GPU Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
else:
    # If no CPU data, show power consumption instead
    if 'gpu_power_w' in df.columns:
        axes[1, 1].plot(df['elapsed_hours'], df['gpu_power_w'], color='red', linewidth=1)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Power (W)')
        axes[1, 1].set_title('GPU Power Consumption Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=df['gpu_power_w'].mean(), color='blue', linestyle='--', alpha=0.5, label=f"Avg: {df['gpu_power_w'].mean():.0f}W")
        axes[1, 1].legend()

plt.tight_layout()
output_path = Path(args.output)
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n{'=' * 50}")
print(f"Charts saved to {output_path}")
print("=" * 50)

estimated_cpu_hours = df.shape[0] * args.interval / 3600 * 50
actual_gpu_hours = df.shape[0] * args.interval / 3600
speedup = estimated_cpu_hours / actual_gpu_hours if actual_gpu_hours > 0 else 0

print(f"\n--- CUDA Acceleration Impact (Estimated) ---")
print(f"Estimated CPU time: {estimated_cpu_hours:.1f} hours (50x slower estimate)")
print(f"Actual GPU time: {actual_gpu_hours:.1f} hours")
print(f"Speedup: {speedup:.0f}x faster with CUDA")
print(f"Time saved: {estimated_cpu_hours - actual_gpu_hours:.1f} hours")
print(f"\nNote: For accurate speedup, run benchmark_cpu_gpu.py")