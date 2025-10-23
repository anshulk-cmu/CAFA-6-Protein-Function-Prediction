"""
Torch Profiler Integration for Protein Embedding Generation

This script profiles the embedding generation process to identify
bottlenecks and optimization opportunities for custom CUDA kernels.
"""

import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import random
from pathlib import Path

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

def profile_model(model_name, sequences, ids, batch_size, num_batches, output_path):
    """Profile embedding generation for a model"""
    print(f"\n{'='*60}")
    print(f"Profiling {model_name}")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
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

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    if device == 'cuda':
        model = model.half()

    # Create dataloader
    dataset = ProteinDataset(sequences, ids)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=(device == 'cuda')
    )

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Profiling {num_batches} batches...")

    # Warmup
    print("Warming up...")
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

    # Profile with torch.profiler
    print("Starting profiler...")

    activities = [ProfilerActivity.CPU]
    if device == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=lambda p: None  # We'll export manually
    ) as prof:
        batch_iter = iter(dataloader)
        for batch_idx in range(min(num_batches, len(dataloader))):
            batch_seqs, batch_ids = next(batch_iter)

            with record_function(f"batch_{batch_idx}"):
                # Tokenization
                with record_function("tokenization"):
                    if 'protbert' in model_name.lower():
                        batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
                    inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True,
                                      truncation=True, max_length=1024)
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                # Inference
                with record_function("inference"):
                    with torch.no_grad():
                        if device == 'cuda':
                            with torch.amp.autocast('cuda'):
                                outputs = model(**inputs)
                                embeddings = outputs.last_hidden_state
                        else:
                            outputs = model(**inputs)
                            embeddings = outputs.last_hidden_state

                # Pooling
                with record_function("pooling"):
                    embeddings = embeddings.mean(dim=1)

                # CPU transfer
                with record_function("cpu_transfer"):
                    embeddings_cpu = embeddings.cpu()

            if device == 'cuda':
                torch.cuda.synchronize()

            print(f"  Profiled batch {batch_idx + 1}/{num_batches}")

    # Export trace
    trace_path = output_path / f"profiler_trace_{model_name}.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace exported: {trace_path}")
    print(f"  View in chrome://tracing")

    # Print summary
    print(f"\n{'='*60}")
    print("PROFILER SUMMARY")
    print(f"{'='*60}")

    # CPU time summary
    print("\nTop 10 CPU operations:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    # CUDA time summary (if available)
    if device == 'cuda':
        print("\nTop 10 CUDA operations:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Memory summary
        print("\nTop 10 memory-intensive operations:")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Key insights
    print(f"\n{'='*60}")
    print("KEY INSIGHTS FOR OPTIMIZATION")
    print(f"{'='*60}")

    key_averages = prof.key_averages()

    # Find tokenization time
    tokenization_events = [e for e in key_averages if 'tokenization' in e.key.lower()]
    if tokenization_events:
        tok_time = sum(e.cpu_time_total for e in tokenization_events) / 1e6  # Convert to ms
        print(f"\nTokenization: {tok_time:.2f}ms")
        print("  → Potential optimization: Custom tokenizer or caching")

    # Find inference time
    inference_events = [e for e in key_averages if 'inference' in e.key.lower()]
    if inference_events:
        inf_time = sum(e.cpu_time_total for e in inference_events) / 1e6
        print(f"\nInference: {inf_time:.2f}ms")
        print("  → Dominant computation (expected)")

    # Find pooling time
    pooling_events = [e for e in key_averages if 'pooling' in e.key.lower()]
    if pooling_events:
        pool_time = sum(e.cpu_time_total for e in pooling_events) / 1e6
        print(f"\nPooling: {pool_time:.2f}ms")
        print("  → Potential optimization: Custom CUDA reduction kernel")

    # Find CPU transfer time
    transfer_events = [e for e in key_averages if 'cpu_transfer' in e.key.lower()]
    if transfer_events:
        transfer_time = sum(e.cpu_time_total for e in transfer_events) / 1e6
        print(f"\nCPU Transfer: {transfer_time:.2f}ms")
        print("  → Potential optimization: Pinned memory, async transfers")

    # MatMul operations (key for transformer models)
    matmul_events = [e for e in key_averages if 'matmul' in e.key.lower() or 'gemm' in e.key.lower()]
    if matmul_events and device == 'cuda':
        total_matmul_time = sum(e.cuda_time_total for e in matmul_events) / 1e6
        print(f"\nMatrix Multiplications: {total_matmul_time:.2f}ms")
        print("  → Using Tensor Cores (fp16/bf16)")
        print("  → Already optimized by cuBLAS")

    # Attention operations
    attention_events = [e for e in key_averages if 'attention' in e.key.lower()]
    if attention_events:
        if device == 'cuda':
            attn_time = sum(e.cuda_time_total for e in attention_events) / 1e6
        else:
            attn_time = sum(e.cpu_time_total for e in attention_events) / 1e6
        print(f"\nAttention: {attn_time:.2f}ms")
        print("  → Potential optimization: Flash Attention 2")

    print(f"\n{'='*60}\n")

    # Cleanup
    del model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()

def create_subset(fasta_path, subset_size, seed=42):
    """Create a random subset of sequences"""
    print(f"Reading sequences from {fasta_path}...")
    sequences = []
    ids = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(record.id)

    print(f"Total sequences: {len(sequences):,}")

    # Random subset
    random.seed(seed)
    indices = random.sample(range(len(sequences)), min(subset_size, len(sequences)))
    subset_sequences = [sequences[i] for i in indices]
    subset_ids = [ids[i] for i in indices]

    print(f"Subset size: {len(subset_sequences):,}")
    return subset_sequences, subset_ids

def main():
    parser = argparse.ArgumentParser(description='Profile protein embedding generation')
    parser.add_argument('--fasta', type=str, default='../data/train_sequences.fasta',
                       help='Path to FASTA file')
    parser.add_argument('--model', type=str, default='esm2_650m',
                       choices=['esm2_650m', 'esm2_3b', 'protbert', 'ankh'],
                       help='Model to profile')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--num-batches', type=int, default=10,
                       help='Number of batches to profile (default: 10)')
    parser.add_argument('--subset', type=int, default=200,
                       help='Number of sequences to use (default: 200)')
    parser.add_argument('--output', type=str, default='../outputs',
                       help='Output directory for profiler traces')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    args = parser.parse_args()

    print("="*60)
    print("TORCH PROFILER FOR PROTEIN EMBEDDINGS")
    print("="*60)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Device: CPU")

    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches to profile: {args.num_batches}")

    # Create subset
    sequences, ids = create_subset(args.fasta, args.subset, args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Profile
    profile_model(
        args.model,
        sequences,
        ids,
        args.batch_size,
        args.num_batches,
        output_path
    )

    print("\nProfiling complete!")
    print(f"\nNext steps:")
    print(f"1. Open chrome://tracing in Chrome/Edge browser")
    print(f"2. Load the trace file: {output_path}/profiler_trace_{args.model}.json")
    print(f"3. Analyze kernel execution and memory patterns")
    print(f"4. Identify bottlenecks for custom CUDA kernel implementation")

if __name__ == "__main__":
    main()
