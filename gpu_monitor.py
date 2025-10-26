import subprocess
import time
import csv
from datetime import datetime
import os
import psutil
import signal
import sys

class GPUMonitor:
    def __init__(self, log_dir='/data/user_data/anshulk/cafa6/logs/gpu_monitoring', interval=2):
        self.interval = interval
        self.running = True
        self.log_dir = log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'gpu_monitor_{timestamp}.csv')
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.fieldnames = [
            'timestamp',
            'elapsed_seconds',
            'gpu_id',
            'gpu_name',
            'gpu_mem_used_mb',
            'gpu_mem_free_mb',
            'gpu_mem_total_mb',
            'gpu_mem_util_pct',
            'gpu_util_pct',
            'gpu_temp_c',
            'gpu_power_draw_w',
            'gpu_power_limit_w',
            'gpu_power_util_pct',
            'gpu_clock_sm_mhz',
            'gpu_clock_mem_mhz',
            'gpu_fan_speed_pct',
            'pcie_tx_mb',
            'pcie_rx_mb',
            'cpu_util_pct',
            'ram_used_gb',
            'ram_total_gb',
            'ram_util_pct'
        ]
        
        self.start_time = time.time()
    
    def signal_handler(self, sig, frame):
        print(f"\n\nMonitoring stopped")
        print(f"Log saved: {self.log_file}")
        self.running = False
        sys.exit(0)
    
    def get_gpu_stats(self):
        try:
            query = (
                'index,name,memory.used,memory.free,memory.total,'
                'utilization.gpu,utilization.memory,temperature.gpu,'
                'power.draw,power.limit,clocks.sm,clocks.mem,fan.speed,'
                'pcie.link.gen.current,pcie.link.width.current'
            )
            
            result = subprocess.run(
                ['nvidia-smi', f'--query-gpu={query}', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode != 0:
                return []
            
            gpu_stats = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                
                try:
                    gpu_id = int(parts[0])
                    gpu_name = parts[1]
                    mem_used = int(parts[2])
                    mem_free = int(parts[3])
                    mem_total = int(parts[4])
                    gpu_util = int(parts[5])
                    mem_util = int(parts[6])
                    temp = int(parts[7])
                    power_draw = float(parts[8])
                    power_limit = float(parts[9])
                    clock_sm = int(parts[10])
                    clock_mem = int(parts[11])
                    fan_speed = int(parts[12]) if parts[12] != 'N/A' else 0
                    
                    gpu_stats.append({
                        'gpu_id': gpu_id,
                        'gpu_name': gpu_name,
                        'gpu_mem_used_mb': mem_used,
                        'gpu_mem_free_mb': mem_free,
                        'gpu_mem_total_mb': mem_total,
                        'gpu_mem_util_pct': mem_util,
                        'gpu_util_pct': gpu_util,
                        'gpu_temp_c': temp,
                        'gpu_power_draw_w': power_draw,
                        'gpu_power_limit_w': power_limit,
                        'gpu_power_util_pct': round((power_draw / power_limit) * 100, 1) if power_limit > 0 else 0,
                        'gpu_clock_sm_mhz': clock_sm,
                        'gpu_clock_mem_mhz': clock_mem,
                        'gpu_fan_speed_pct': fan_speed
                    })
                except (ValueError, IndexError):
                    continue
            
            return gpu_stats
            
        except Exception:
            return []
    
    def get_pcie_stats(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', 'dmon', '-c', '1', '-s', 'pcit'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            pcie_data = {}
            for line in result.stdout.strip().split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        gpu_id = int(parts[0])
                        pcie_tx = int(parts[1])
                        pcie_rx = int(parts[2])
                        pcie_data[gpu_id] = {'tx': pcie_tx, 'rx': pcie_rx}
                    except (ValueError, IndexError):
                        continue
            
            return pcie_data
        except:
            return {}
    
    def get_system_stats(self):
        cpu_util = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        
        return {
            'cpu_util_pct': round(cpu_util, 1),
            'ram_used_gb': round(ram.used / 1e9, 2),
            'ram_total_gb': round(ram.total / 1e9, 2),
            'ram_util_pct': round(ram.percent, 1)
        }
    
    def format_row(self, gpu_stat, pcie_stat, sys_stat, timestamp, elapsed):
        row = {
            'timestamp': timestamp,
            'elapsed_seconds': elapsed,
            'pcie_tx_mb': pcie_stat.get('tx', 0),
            'pcie_rx_mb': pcie_stat.get('rx', 0)
        }
        row.update(gpu_stat)
        row.update(sys_stat)
        return row
    
    def run(self):
        print("="*80)
        print("GPU MONITORING ACTIVE")
        print("="*80)
        print(f"Log file: {self.log_file}")
        print(f"Interval: {self.interval}s")
        print(f"Press Ctrl+C to stop")
        print("="*80)
        print()
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            
            while self.running:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                elapsed = round(time.time() - self.start_time, 2)
                
                gpu_stats = self.get_gpu_stats()
                pcie_stats = self.get_pcie_stats()
                sys_stats = self.get_system_stats()
                
                if gpu_stats:
                    for gpu_stat in gpu_stats:
                        gpu_id = gpu_stat['gpu_id']
                        pcie_stat = pcie_stats.get(gpu_id, {'tx': 0, 'rx': 0})
                        
                        row = self.format_row(gpu_stat, pcie_stat, sys_stats, timestamp, elapsed)
                        writer.writerow(row)
                        
                        print(f"[{timestamp}] GPU{gpu_id}: "
                              f"{gpu_stat['gpu_mem_used_mb']:5d}MB "
                              f"{gpu_stat['gpu_util_pct']:3d}% "
                              f"{gpu_stat['gpu_temp_c']:2d}C "
                              f"{gpu_stat['gpu_power_draw_w']:5.1f}W "
                              f"| CPU: {sys_stats['cpu_util_pct']:4.1f}% "
                              f"| RAM: {sys_stats['ram_util_pct']:4.1f}%")
                    
                    f.flush()
                
                time.sleep(self.interval)

if __name__ == "__main__":
    monitor = GPUMonitor(interval=2)
    monitor.run()
