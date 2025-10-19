import subprocess
import time
import csv
from datetime import datetime
import os
import psutil
import signal
import sys

class GPUMonitor:
    def __init__(self, log_file='../outputs/gpu_monitor_phase1.csv', interval=5):
        self.log_file = log_file
        self.interval = interval
        self.running = True
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        if os.path.exists(log_file):
            print(f"⚠ Appending to existing log: {log_file}")
            self.mode = 'a'
        else:
            print(f"✓ Creating new log: {log_file}")
            self.mode = 'w'
        
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, sig, frame):
        print("\n\n✓ Monitoring stopped gracefully")
        print(f"✓ Log: {self.log_file}")
        self.running = False
        sys.exit(0)
    
    def get_stats(self):
        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            parts = result.stdout.strip().split(', ')
            gpu_mem_used, gpu_mem_total, gpu_util, gpu_temp = parts[:4]
            gpu_power = parts[4] if len(parts) > 4 else '0'
            
            return {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'gpu_mem_gb': round(int(gpu_mem_used) / 1024, 2),
                'gpu_util_pct': int(gpu_util),
                'gpu_temp_c': int(gpu_temp),
                'gpu_power_w': float(gpu_power),
                'cpu_pct': round(psutil.cpu_percent(interval=0.5), 1),
                'ram_gb': round(psutil.virtual_memory().used / 1e9, 1)
            }
        except Exception as e:
            return None
    
    def run(self):
        print("="*60)
        print("GPU MONITOR ACTIVE")
        print("="*60)
        print(f"Interval: {self.interval}s | Press Ctrl+C to stop\n")
        
        with open(self.log_file, self.mode, newline='') as f:
            writer = None
            
            while self.running:
                stats = self.get_stats()
                
                if stats:
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=stats.keys())
                        if self.mode == 'w':
                            writer.writeheader()
                    
                    writer.writerow(stats)
                    f.flush()
                    
                    print(f"[{stats['timestamp']}] "
                          f"GPU: {stats['gpu_mem_gb']:.1f}GB "
                          f"{stats['gpu_util_pct']:3d}% "
                          f"{stats['gpu_temp_c']:2d}°C "
                          f"{stats['gpu_power_w']:4.0f}W | "
                          f"CPU: {stats['cpu_pct']:4.1f}% | "
                          f"RAM: {stats['ram_gb']:.1f}GB")
                
                time.sleep(self.interval)

if __name__ == "__main__":
    monitor = GPUMonitor()
    monitor.run()