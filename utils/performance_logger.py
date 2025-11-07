#!/usr/bin/env python3
"""
Performance logging utilities for benchmarking and profiling.

Provides structured logging for timing, memory usage, and system metrics.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, pynvml.NVMLError):
    PYNVML_AVAILABLE = False


class PerformanceLogger:
    """
    Logger for performance metrics including timing, memory, and system stats.
    """

    def __init__(self, name: str = "benchmark"):
        """
        Initialize performance logger.

        Args:
            name: Name of this logging session
        """
        self.name = name
        self.timers = {}
        self.batch_stats = []
        self.memory_snapshots = []
        self.metadata = {}
        self.start_time = time.time()

    @contextmanager
    def timer(self, label: str):
        """
        Context manager for timing code blocks.

        Args:
            label: Label for this timing

        Example:
            with logger.timer("forward_pass"):
                output = model(input)
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if label not in self.timers:
                self.timers[label] = []
            self.timers[label].append(elapsed)

    def add_batch_stat(self, batch_time: float, batch_size: int,
                       max_seq_length: Optional[int] = None,
                       **kwargs):
        """
        Record statistics for a single batch.

        Args:
            batch_time: Time to process this batch (seconds)
            batch_size: Number of items in batch
            max_seq_length: Maximum sequence length in batch
            **kwargs: Additional metrics to record
        """
        stat = {
            'batch_time': batch_time,
            'batch_size': batch_size,
            'throughput': batch_size / batch_time if batch_time > 0 else 0
        }

        if max_seq_length is not None:
            stat['max_seq_length'] = max_seq_length

        stat.update(kwargs)
        self.batch_stats.append(stat)

    def log_memory(self, device: Optional[Any] = None, label: str = ""):
        """
        Log current memory usage.

        Args:
            device: Torch device (for GPU memory) or None (for CPU memory)
            label: Label for this snapshot
        """
        snapshot = {
            'timestamp': time.time() - self.start_time,
            'label': label
        }

        # CPU memory
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            mem_info = process.memory_info()
            snapshot['cpu_memory'] = {
                'rss': mem_info.rss,  # Resident Set Size
                'vms': mem_info.vms,  # Virtual Memory Size
                'rss_mb': mem_info.rss / (1024 ** 2),
                'vms_mb': mem_info.vms / (1024 ** 2)
            }

            # System memory
            sys_mem = psutil.virtual_memory()
            snapshot['system_memory'] = {
                'total': sys_mem.total,
                'available': sys_mem.available,
                'percent': sys_mem.percent,
                'total_gb': sys_mem.total / (1024 ** 3),
                'available_gb': sys_mem.available / (1024 ** 3)
            }

        # GPU memory
        if TORCH_AVAILABLE and device is not None and str(device).startswith('cuda'):
            if isinstance(device, str):
                device_id = int(device.split(':')[1]) if ':' in device else 0
            else:
                device_id = device.index if hasattr(device, 'index') else 0

            snapshot['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated(device_id),
                'reserved': torch.cuda.memory_reserved(device_id),
                'max_allocated': torch.cuda.max_memory_allocated(device_id),
                'allocated_gb': torch.cuda.memory_allocated(device_id) / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved(device_id) / (1024 ** 3),
                'max_allocated_gb': torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
            }

        self.memory_snapshots.append(snapshot)

    def add_metadata(self, key: str, value: Any):
        """
        Add metadata to the log.

        Args:
            key: Metadata key
            value: Metadata value (must be JSON-serializable)
        """
        self.metadata[key] = value

    def get_timer_stats(self, label: str) -> Dict[str, float]:
        """
        Get statistics for a timer.

        Args:
            label: Timer label

        Returns:
            Dictionary with mean, median, min, max, total, count
        """
        if label not in self.timers or not self.timers[label]:
            return {}

        times = np.array(self.timers[label])
        return {
            'mean': float(np.mean(times)),
            'median': float(np.median(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'p95': float(np.percentile(times, 95)),
            'p99': float(np.percentile(times, 99)),
            'total': float(np.sum(times)),
            'count': len(times)
        }

    def get_batch_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics across all batches.

        Returns:
            Dictionary with aggregate metrics
        """
        if not self.batch_stats:
            return {}

        batch_times = [b['batch_time'] for b in self.batch_stats]
        batch_sizes = [b['batch_size'] for b in self.batch_stats]
        throughputs = [b['throughput'] for b in self.batch_stats]

        stats = {
            'total_batches': len(self.batch_stats),
            'total_items': sum(batch_sizes),
            'batch_time': {
                'mean': float(np.mean(batch_times)),
                'median': float(np.median(batch_times)),
                'std': float(np.std(batch_times)),
                'min': float(np.min(batch_times)),
                'max': float(np.max(batch_times)),
                'p95': float(np.percentile(batch_times, 95)),
                'p99': float(np.percentile(batch_times, 99)),
                'total': float(np.sum(batch_times))
            },
            'throughput': {
                'mean': float(np.mean(throughputs)),
                'median': float(np.median(throughputs)),
                'std': float(np.std(throughputs)),
                'min': float(np.min(throughputs)),
                'max': float(np.max(throughputs))
            }
        }

        # Add seq_length stats if available
        if 'max_seq_length' in self.batch_stats[0]:
            seq_lengths = [b['max_seq_length'] for b in self.batch_stats]
            stats['seq_length'] = {
                'mean': float(np.mean(seq_lengths)),
                'median': float(np.median(seq_lengths)),
                'min': float(np.min(seq_lengths)),
                'max': float(np.max(seq_lengths))
            }

        return stats

    def export(self, output_path: str):
        """
        Export all metrics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output = {
            'name': self.name,
            'total_time': time.time() - self.start_time,
            'metadata': self.metadata,
            'timers': {label: self.get_timer_stats(label) for label in self.timers},
            'batch_stats': self.get_batch_stats(),
            'batch_details': self.batch_stats,
            'memory_snapshots': self.memory_snapshots
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

    def print_summary(self):
        """
        Print human-readable summary of metrics.
        """
        print(f"\n{'=' * 70}")
        print(f"Performance Summary: {self.name}")
        print(f"{'=' * 70}")

        print(f"\nTotal Time: {format_time(time.time() - self.start_time)}")

        # Print timer stats
        if self.timers:
            print(f"\n{'Timer Statistics':^70}")
            print("-" * 70)
            for label, times in self.timers.items():
                stats = self.get_timer_stats(label)
                print(f"{label}:")
                print(f"  Mean: {stats['mean']:.4f}s, Median: {stats['median']:.4f}s")
                print(f"  Total: {stats['total']:.2f}s ({stats['count']} calls)")

        # Print batch stats
        batch_stats = self.get_batch_stats()
        if batch_stats:
            print(f"\n{'Batch Statistics':^70}")
            print("-" * 70)
            print(f"Total Batches: {batch_stats['total_batches']}")
            print(f"Total Items: {batch_stats['total_items']}")
            print(f"Mean Batch Time: {batch_stats['batch_time']['mean']:.4f}s")
            print(f"Mean Throughput: {batch_stats['throughput']['mean']:.2f} items/sec")

        # Print memory stats
        if self.memory_snapshots:
            latest = self.memory_snapshots[-1]
            print(f"\n{'Memory Usage (Latest)':^70}")
            print("-" * 70)
            if 'cpu_memory' in latest:
                print(f"CPU RSS: {latest['cpu_memory']['rss_mb']:.1f} MB")
            if 'gpu_memory' in latest:
                print(f"GPU Allocated: {latest['gpu_memory']['allocated_gb']:.2f} GB")
                print(f"GPU Reserved: {latest['gpu_memory']['reserved_gb']:.2f} GB")
                print(f"GPU Peak: {latest['gpu_memory']['max_allocated_gb']:.2f} GB")

        print(f"\n{'=' * 70}\n")


def get_gpu_memory_stats(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU memory statistics.

    Args:
        device_id: CUDA device ID

    Returns:
        Dictionary with memory stats in bytes and GB
    """
    if not TORCH_AVAILABLE:
        return {}

    return {
        'allocated': torch.cuda.memory_allocated(device_id),
        'reserved': torch.cuda.memory_reserved(device_id),
        'max_allocated': torch.cuda.max_memory_allocated(device_id),
        'allocated_gb': torch.cuda.memory_allocated(device_id) / (1024 ** 3),
        'reserved_gb': torch.cuda.memory_reserved(device_id) / (1024 ** 3),
        'max_allocated_gb': torch.cuda.max_memory_allocated(device_id) / (1024 ** 3)
    }


def get_cpu_memory_stats() -> Dict[str, float]:
    """
    Get CPU memory statistics.

    Returns:
        Dictionary with memory stats in bytes and MB
    """
    if not PSUTIL_AVAILABLE:
        return {}

    process = psutil.Process()
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()

    return {
        'process_rss': mem_info.rss,
        'process_vms': mem_info.vms,
        'process_rss_mb': mem_info.rss / (1024 ** 2),
        'process_vms_mb': mem_info.vms / (1024 ** 2),
        'system_total': sys_mem.total,
        'system_available': sys_mem.available,
        'system_percent': sys_mem.percent,
        'system_total_gb': sys_mem.total / (1024 ** 3),
        'system_available_gb': sys_mem.available / (1024 ** 3)
    }


def get_gpu_utilization(device_id: int = 0) -> Dict[str, float]:
    """
    Get GPU utilization statistics.

    Args:
        device_id: CUDA device ID

    Returns:
        Dictionary with utilization, temperature, power
    """
    if not PYNVML_AVAILABLE:
        return {}

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts

        return {
            'gpu_util_percent': utilization.gpu,
            'memory_util_percent': utilization.memory,
            'temperature_c': temperature,
            'power_watts': power
        }
    except pynvml.NVMLError:
        return {}


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_memory(bytes_val: float) -> str:
    """
    Format bytes into human-readable memory string.

    Args:
        bytes_val: Memory in bytes

    Returns:
        Formatted string (e.g., "11.2 GB")
    """
    if bytes_val < 1024:
        return f"{bytes_val:.0f} B"
    elif bytes_val < 1024 ** 2:
        return f"{bytes_val / 1024:.1f} KB"
    elif bytes_val < 1024 ** 3:
        return f"{bytes_val / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
