import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../outputs/gpu_monitor.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['elapsed_hours'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600

print("=" * 50)
print("GPU MONITORING SUMMARY")
print("=" * 50)
print(f"\nDuration: {df.shape[0] * 5 / 3600:.2f} hours")
print(f"Total Samples: {df.shape[0]:,}")
print(f"\n--- GPU Memory ---")
print(f"Average: {df['gpu_mem_used_gb'].mean():.2f} GB")
print(f"Peak: {df['gpu_mem_used_gb'].max():.2f} GB")
print(f"Min: {df['gpu_mem_used_gb'].min():.2f} GB")
print(f"\n--- GPU Utilization ---")
print(f"Average: {df['gpu_utilization_pct'].mean():.1f}%")
print(f"Peak: {df['gpu_utilization_pct'].max():.1f}%")
print(f"Min: {df['gpu_utilization_pct'].min():.1f}%")
print(f"\n--- Temperature ---")
print(f"Average: {df['gpu_temp_c'].mean():.1f}°C")
print(f"Peak: {df['gpu_temp_c'].max():.1f}°C")
print(f"Min: {df['gpu_temp_c'].min():.1f}°C")
print(f"\n--- System Resources ---")
print(f"Avg CPU: {df['cpu_percent'].mean():.1f}%")
print(f"Avg RAM: {df['ram_used_gb'].mean():.1f} GB")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df['elapsed_hours'], df['gpu_mem_used_gb'], color='blue', linewidth=1)
axes[0, 0].set_xlabel('Time (hours)')
axes[0, 0].set_ylabel('GPU Memory (GB)')
axes[0, 0].set_title('GPU Memory Usage Over Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=df['gpu_mem_used_gb'].mean(), color='red', linestyle='--', alpha=0.5, label=f"Avg: {df['gpu_mem_used_gb'].mean():.1f}GB")
axes[0, 0].legend()

axes[0, 1].plot(df['elapsed_hours'], df['gpu_utilization_pct'], color='green', linewidth=1)
axes[0, 1].set_xlabel('Time (hours)')
axes[0, 1].set_ylabel('GPU Utilization (%)')
axes[0, 1].set_title('GPU Utilization Over Time')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=df['gpu_utilization_pct'].mean(), color='red', linestyle='--', alpha=0.5, label=f"Avg: {df['gpu_utilization_pct'].mean():.0f}%")
axes[0, 1].legend()

axes[1, 0].plot(df['elapsed_hours'], df['gpu_temp_c'], color='orange', linewidth=1)
axes[1, 0].set_xlabel('Time (hours)')
axes[1, 0].set_ylabel('Temperature (°C)')
axes[1, 0].set_title('GPU Temperature Over Time')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Thermal Warning (80°C)')
axes[1, 0].legend()

axes[1, 1].plot(df['elapsed_hours'], df['cpu_percent'], color='purple', alpha=0.7, label='CPU %')
axes[1, 1].set_xlabel('Time (hours)')
axes[1, 1].set_ylabel('Utilization (%)')
axes[1, 1].set_title('CPU vs GPU Comparison')
axes[1, 1].plot(df['elapsed_hours'], df['gpu_utilization_pct'], color='green', alpha=0.7, label='GPU %')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('../outputs/gpu_analysis_full.png', dpi=150, bbox_inches='tight')
print(f"\n{'=' * 50}")
print("Charts saved to outputs/gpu_analysis_full.png")
print("=" * 50)

estimated_cpu_hours = df.shape[0] * 5 / 3600 * 50
actual_gpu_hours = df.shape[0] * 5 / 3600
speedup = estimated_cpu_hours / actual_gpu_hours if actual_gpu_hours > 0 else 0

print(f"\n--- CUDA Acceleration Impact ---")
print(f"Estimated CPU time: {estimated_cpu_hours:.1f} hours (50x slower)")
print(f"Actual GPU time: {actual_gpu_hours:.1f} hours")
print(f"Speedup: {speedup:.0f}x faster with CUDA")
print(f"Time saved: {estimated_cpu_hours - actual_gpu_hours:.1f} hours")