"""
CPU vs GPU Benchmark for Protein Embedding Generation

This script runs the same workload on both CPU and GPU to measure
actual speedup for the GPU programming project report.
"""

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path
import random

def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class ProteinDataset(Dataset):
    def __init__(self, sequences, ids):
        self.data = list(zip(sequences, ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, id_ = self.data[idx]
        return seq, id_

def collate_fn(batch):
    seqs, ids = zip(*batch)
    return list(seqs), list(ids)

def benchmark_model(model_name, sequences, ids, device, batch_size):
    """Benchmark a single model on a given device"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name} on {device.upper()}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print(f"Loading model...")
    if 'esm2' in model_name.lower():
        if '3b' in model_name.lower():
            hf_name = "facebook/esm2_t36_3B_UR50D"
        else:
            hf_name = "facebook/esm2_t33_650M_UR50D"
        model = EsmModel.from_pretrained(hf_name)
        tokenizer = EsmTokenizer.from_pretrained(hf_name)
    elif 'protbert' in model_name.lower():
        hf_name = "Rostlab/prot_bert_bfd"
        model = AutoModel.from_pretrained(hf_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    elif 'ankh' in model_name.lower():
        hf_name = "ElnaggarLab/ankh-large"
        model = AutoModel.from_pretrained(hf_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device).eval()

    # Apply half precision only for CUDA
    if device == 'cuda':
        model = model.half()

    # Create dataset and dataloader
    dataset = ProteinDataset(sequences, ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device == 'cuda')
    )

    # Warmup iterations
    print("Warming up (3 batches)...")
    warmup_iter = iter(dataloader)
    for _ in range(min(3, len(dataloader))):
        batch_seqs, batch_ids = next(warmup_iter)
        if 'protbert' in model_name.lower():
            batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True,
                          truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Actual benchmark
    print("Running benchmark...")
    batch_times = []
    total_proteins = 0

    start_time = time.time()

    for batch_seqs, batch_ids in tqdm(dataloader, desc=f"{model_name} [{device}]"):
        batch_start = time.time()

        # Tokenize
        if 'protbert' in model_name.lower():
            batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
        inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True,
                          truncation=True, max_length=1024)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            if device == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

        # Wait for GPU to finish
        if device == 'cuda':
            torch.cuda.synchronize()

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        total_proteins += len(batch_seqs)

    total_time = time.time() - start_time

    # Calculate statistics
    stats = {
        'model': model_name,
        'device': device,
        'batch_size': batch_size,
        'total_proteins': total_proteins,
        'total_time': total_time,
        'proteins_per_sec': total_proteins / total_time,
        'avg_batch_time': np.mean(batch_times),
        'std_batch_time': np.std(batch_times),
        'min_batch_time': np.min(batch_times),
        'max_batch_time': np.max(batch_times)
    }

    # Memory stats for GPU
    if device == 'cuda':
        stats['peak_memory_gb'] = torch.cuda.max_memory_allocated() / 1e9
        stats['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Print summary
    print(f"\nResults:")
    print(f"  Total time: {total_time:.2f}s ({total_time/60:.1f}m)")
    print(f"  Throughput: {stats['proteins_per_sec']:.1f} proteins/sec")
    print(f"  Avg batch time: {stats['avg_batch_time']:.3f}s")
    if device == 'cuda':
        print(f"  Peak memory: {stats['peak_memory_gb']:.2f}GB")

    # Cleanup
    del model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

    return stats

def create_subset(fasta_path, subset_size, seed=42):
    """Create a random subset of sequences from FASTA file (seed already set globally)"""
    print(f"Reading sequences from {fasta_path}...")
    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)

    print(f"Total sequences: {len(sequences):,}")

    # Random subset (uses global seed)
    indices = random.sample(range(len(sequences)), min(subset_size, len(sequences)))
    subset_sequences = [sequences[i] for i in indices]
    subset_ids = [ids[i] for i in indices]

    print(f"Subset size: {len(subset_sequences):,}")
    avg_len = np.mean([len(seq) for seq in subset_sequences])
    print(f"Average sequence length: {avg_len:.1f}")

    return subset_sequences, subset_ids

def plot_comparison(results, output_path):
    """Create comparison visualization"""
    models = list(set(r['model'] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput comparison
    for model in models:
        cpu_result = next((r for r in results if r['model'] == model and r['device'] == 'cpu'), None)
        gpu_result = next((r for r in results if r['model'] == model and r['device'] == 'cuda'), None)

        if cpu_result and gpu_result:
            x_pos = models.index(model)
            axes[0].bar(x_pos - 0.2, cpu_result['proteins_per_sec'], width=0.4, label='CPU' if x_pos == 0 else '', color='orange')
            axes[0].bar(x_pos + 0.2, gpu_result['proteins_per_sec'], width=0.4, label='GPU' if x_pos == 0 else '', color='green')

    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Throughput (proteins/sec)')
    axes[0].set_title('CPU vs GPU Throughput')
    axes[0].set_xticks(range(len(models)))
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Speedup comparison
    speedups = []
    model_names = []
    for model in models:
        cpu_result = next((r for r in results if r['model'] == model and r['device'] == 'cpu'), None)
        gpu_result = next((r for r in results if r['model'] == model and r['device'] == 'cuda'), None)

        if cpu_result and gpu_result:
            speedup = gpu_result['proteins_per_sec'] / cpu_result['proteins_per_sec']
            speedups.append(speedup)
            model_names.append(model)

    axes[1].bar(range(len(speedups)), speedups, color='blue')
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Speedup (GPU / CPU)')
    axes[1].set_title('GPU Acceleration Speedup')
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No speedup')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU for protein embedding generation')
    parser.add_argument('--fasta', type=str, default='../data/train_sequences.fasta',
                       help='Path to FASTA file')
    parser.add_argument('--subset', type=int, default=1000,
                       help='Number of sequences to benchmark (default: 1000)')
    parser.add_argument('--models', nargs='+', default=['esm2_650m'],
                       choices=['esm2_650m', 'esm2_3b', 'protbert', 'ankh'],
                       help='Models to benchmark (default: esm2_650m)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output', type=str, default='../outputs/cpu_gpu_benchmark_results.json',
                       help='Output path for results JSON')
    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot run GPU benchmark.")
        return

    print("="*60)
    print("CPU vs GPU BENCHMARK")
    print("="*60)

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Subset size: {args.subset:,} sequences")
    print(f"Batch size: {args.batch_size}")

    # Create subset
    sequences, ids = create_subset(args.fasta, args.subset, args.seed)

    # Run benchmarks
    results = []

    for model_name in args.models:
        # CPU benchmark
        cpu_stats = benchmark_model(model_name, sequences, ids, 'cpu', args.batch_size)
        results.append(cpu_stats)

        # GPU benchmark
        gpu_stats = benchmark_model(model_name, sequences, ids, 'cuda', args.batch_size)
        results.append(gpu_stats)

        # Calculate speedup
        speedup = gpu_stats['proteins_per_sec'] / cpu_stats['proteins_per_sec']
        time_saved = cpu_stats['total_time'] - gpu_stats['total_time']

        print(f"\n{'='*60}")
        print(f"SPEEDUP SUMMARY: {model_name}")
        print(f"{'='*60}")
        print(f"CPU time: {cpu_stats['total_time']:.2f}s")
        print(f"GPU time: {gpu_stats['total_time']:.2f}s")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {time_saved:.2f}s")
        print(f"{'='*60}\n")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'benchmark_config': {
                'subset_size': args.subset,
                'batch_size': args.batch_size,
                'models': args.models,
                'seed': args.seed
            },
            'results': results
        }, f, indent=2)

    print(f"Results saved: {output_path}")

    # Create visualization
    plot_path = output_path.parent / "cpu_gpu_comparison.png"
    plot_comparison(results, plot_path)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for model_name in args.models:
        cpu_result = next(r for r in results if r['model'] == model_name and r['device'] == 'cpu')
        gpu_result = next(r for r in results if r['model'] == model_name and r['device'] == 'cuda')
        speedup = gpu_result['proteins_per_sec'] / cpu_result['proteins_per_sec']

        print(f"\n{model_name}:")
        print(f"  CPU: {cpu_result['proteins_per_sec']:.1f} proteins/sec")
        print(f"  GPU: {gpu_result['proteins_per_sec']:.1f} proteins/sec")
        print(f"  Speedup: {speedup:.2f}x")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
