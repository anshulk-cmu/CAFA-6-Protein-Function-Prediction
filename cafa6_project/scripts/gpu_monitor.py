import subprocess
import time
import csv
from datetime import datetime
import os
import psutil

def get_gpu_stats_nvidia_smi():
    try:
        result = subprocess.run([
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        gpu_mem_used, gpu_mem_total, gpu_util, gpu_temp = result.stdout.strip().split(', ')
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_mem_used_mb': int(gpu_mem_used),
            'gpu_mem_total_mb': int(gpu_mem_total),
            'gpu_mem_used_gb': round(int(gpu_mem_used) / 1024, 2),
            'gpu_utilization_pct': int(gpu_util),
            'gpu_temp_c': int(gpu_temp),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'ram_used_gb': round(psutil.virtual_memory().used / 1e9, 2),
            'ram_total_gb': round(psutil.virtual_memory().total / 1e9, 2)
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def monitor_gpu(log_file='../outputs/gpu_monitor.csv', interval=5):
    print("GPU Monitor Started (nvidia-smi)")
    print(f"Logging to: {log_file}")
    print(f"Update interval: {interval}s")
    print("Press Ctrl+C to stop\n")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    with open(log_file, 'w', newline='') as f:
        writer = None
        
        try:
            while True:
                stats = get_gpu_stats_nvidia_smi()
                
                if stats:
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=stats.keys())
                        writer.writeheader()
                    
                    writer.writerow(stats)
                    f.flush()
                    
                    print(f"[{stats['timestamp']}] "
                          f"GPU: {stats['gpu_mem_used_gb']:.1f}GB "
                          f"({stats['gpu_utilization_pct']}%) "
                          f"{stats['gpu_temp_c']}°C | "
                          f"CPU: {stats['cpu_percent']:.1f}% | "
                          f"RAM: {stats['ram_used_gb']:.1f}GB")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped")
            print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    monitor_gpu()