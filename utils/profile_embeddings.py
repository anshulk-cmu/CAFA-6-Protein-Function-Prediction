#!/usr/bin/env python3
"""
Profiling script for Phase 1B using torch.profiler.

Generates detailed Chrome traces for kernel-level performance analysis.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import json

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.performance_logger import format_time


def setup_logging():
    """Setup logging to console."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler()]
    )


def read_fasta(fasta_path: str, max_sequences: int = None):
    """Read FASTA file and return list of (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                    if max_sequences and len(sequences) >= max_sequences:
                        break
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None and (not max_sequences or len(sequences) < max_sequences):
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def preprocess_sequence_t5(sequence: str) -> str:
    """Preprocess sequence for T5 models."""
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    return ' '.join(list(sequence))


def get_prefix_token(model_name: str) -> str:
    """Get prefix token for specific models."""
    if 'prost' in model_name.lower():
        return '<AA2fold>'
    return ''


def main():
    parser = argparse.ArgumentParser(description='Profile embedding generation with torch.profiler')
    parser.add_argument('--model', type=str, default='esm2_3B',
                       choices=['esm2_3B', 'esm_c_600m', 'esm1b', 'prot_t5_xl', 'prost_t5'],
                       help='Model to profile')
    parser.add_argument('--input', type=str,
                       default='data/train_sequences_benchmark_1k.fasta',
                       help='Input FASTA file')
    parser.add_argument('--batch-size', type=int, default=24,
                       help='Batch size for profiling')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batches to profile')
    parser.add_argument('--output-dir', type=str, default='traces',
                       help='Output directory for traces')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file for model specifications')
    parser.add_argument('--config-t5', type=str, default='config_t5.yaml',
                       help='Config file for T5 models')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID')

    args = parser.parse_args()

    setup_logging()

    logging.info("=" * 70)
    logging.info("Phase 1B: Profiling with torch.profiler")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Num batches: {args.num_batches}")
    logging.info(f"GPU ID: {args.gpu_id}")

    # Check CUDA
    if not torch.cuda.is_available():
        logging.error("CUDA not available!")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu_id}')
    logging.info(f"Device: {device}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    is_t5 = args.model in ['prot_t5_xl', 'prost_t5']
    config_file = args.config_t5 if is_t5 else args.config

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if args.model not in config['models']:
        logging.error(f"Model {args.model} not found in {config_file}")
        sys.exit(1)

    model_config = config['models'][args.model]
    model_name = model_config['name']

    logging.info(f"Model name: {model_name}")

    # Read sequences (limit to what we need for profiling)
    max_sequences_needed = args.batch_size * (args.num_batches + 2)  # +2 for warmup
    logging.info(f"\nReading up to {max_sequences_needed} sequences from {args.input}...")
    sequences = read_fasta(args.input, max_sequences=max_sequences_needed)
    logging.info(f"Loaded {len(sequences)} sequences")

    # Sort by length
    sequences.sort(key=lambda x: len(x[1]))

    # Load model and tokenizer
    logging.info(f"\nLoading model {model_name}...")

    try:
        # Load tokenizer
        if 'esmplusplus' in model_name.lower() or 'synthyra' in model_name.lower():
            logging.info("Loading ESM2 tokenizer for ESMplusplus model")
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
        elif is_t5:
            # T5 models require T5Tokenizer with legacy=True for compatibility
            logging.info("Loading T5Tokenizer with legacy=True")
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                legacy=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Load model
        if is_t5:
            model = T5EncoderModel.from_pretrained(model_name, torch_dtype=torch.float16)
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)

        model = model.to(device)
        model.eval()

        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    prefix_token = get_prefix_token(model_name)
    if prefix_token:
        logging.info(f"Using prefix token: {prefix_token}")

    # Prepare profiling schedule
    # wait=1, warmup=1, active=args.num_batches, repeat=1
    logging.info(f"\nProfiling schedule:")
    logging.info(f"  Wait: 1 batch (skip profiling)")
    logging.info(f"  Warmup: 1 batch (warmup GPU)")
    logging.info(f"  Active: {args.num_batches} batches (profile)")
    logging.info(f"  Repeat: 1 time")

    # Setup profiler
    trace_path = output_dir / f"{args.model}_profile"
    logging.info(f"\nChrome trace will be saved to: {trace_path}")

    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=args.num_batches,
        repeat=1
    )

    # Collect kernel statistics
    kernel_stats = []

    def trace_handler(prof):
        """Custom trace handler to collect kernel statistics."""
        logging.info(f"Step {prof.step_num}: Exporting trace...")

        # Export Chrome trace
        prof.export_chrome_trace(f"{trace_path}_step{prof.step_num}.json")

        # Export for TensorBoard
        prof.export_stacks(f"{trace_path}_stacks.txt", "self_cuda_time_total")

        # Get key averages table
        key_avg = prof.key_averages()

        # Collect statistics
        for item in key_avg:
            kernel_stats.append({
                'name': item.key,
                'cpu_time': item.cpu_time_total,
                'cuda_time': item.cuda_time_total,
                'cpu_time_avg': item.cpu_time,
                'cuda_time_avg': item.cuda_time,
                'calls': item.count,
                'input_shapes': str(item.input_shapes) if hasattr(item, 'input_shapes') else None
            })

    # Start profiling
    logging.info("\n" + "=" * 70)
    logging.info("Starting profiling...")
    logging.info("=" * 70)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        total_batches = 2 + args.num_batches  # wait + warmup + active
        batch_idx = 0

        with torch.no_grad():
            while batch_idx < total_batches:
                # Get batch
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(sequences))

                if start_idx >= len(sequences):
                    break

                batch_sequences = sequences[start_idx:end_idx]
                batch_seqs_only = [seq[1] for seq in batch_sequences]

                # Preprocess if T5
                if is_t5:
                    batch_seqs_only = [preprocess_sequence_t5(seq) for seq in batch_seqs_only]
                    if prefix_token:
                        batch_seqs_only = [f"{prefix_token} {seq}" for seq in batch_seqs_only]

                # Tokenize
                inputs = tokenizer(
                    batch_seqs_only,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Forward pass
                outputs = model(**inputs)

                # Mean pooling
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs['attention_mask']

                if is_t5 and prefix_token:
                    attention_mask = attention_mask.clone()
                    attention_mask[:, 0] = 0
                    attention_mask[:, -1] = 0

                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_hidden / sum_mask

                # Make sure GPU is done
                torch.cuda.synchronize(device)

                # Step profiler
                prof.step()

                batch_idx += 1
                phase = "wait" if batch_idx == 1 else "warmup" if batch_idx == 2 else "active"
                logging.info(f"Batch {batch_idx}/{total_batches} ({phase}) - completed")

    logging.info("\n" + "=" * 70)
    logging.info("Profiling complete!")
    logging.info("=" * 70)

    # Aggregate kernel statistics
    if kernel_stats:
        logging.info("\nAggregating kernel statistics...")

        # Group by kernel name
        kernel_totals = {}
        for stat in kernel_stats:
            name = stat['name']
            if name not in kernel_totals:
                kernel_totals[name] = {
                    'cuda_time_total': 0,
                    'cpu_time_total': 0,
                    'calls': 0
                }
            kernel_totals[name]['cuda_time_total'] += stat['cuda_time']
            kernel_totals[name]['cpu_time_total'] += stat['cpu_time']
            kernel_totals[name]['calls'] += stat['calls']

        # Sort by CUDA time
        sorted_kernels = sorted(kernel_totals.items(), key=lambda x: x[1]['cuda_time_total'], reverse=True)

        # Save summary
        summary = {
            'model': args.model,
            'batch_size': args.batch_size,
            'num_batches_profiled': args.num_batches,
            'top_kernels': []
        }

        logging.info(f"\nTop 10 Most Expensive CUDA Kernels:")
        logging.info("-" * 70)
        for i, (name, stats) in enumerate(sorted_kernels[:10]):
            cuda_time_ms = stats['cuda_time_total'] / 1000.0  # Convert to ms
            logging.info(f"{i+1}. {name}")
            logging.info(f"   CUDA time: {cuda_time_ms:.2f} ms ({stats['calls']} calls)")

            summary['top_kernels'].append({
                'rank': i + 1,
                'name': name,
                'cuda_time_ms': cuda_time_ms,
                'cuda_time_us': stats['cuda_time_total'],
                'cpu_time_us': stats['cpu_time_total'],
                'calls': stats['calls']
            })

        # Save summary JSON
        summary_path = output_dir / f"{args.model}_profile_summary.json"
        logging.info(f"\nSaving summary to: {summary_path}")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    logging.info("\n" + "=" * 70)
    logging.info("Next Steps:")
    logging.info("=" * 70)
    logging.info(f"1. Open chrome://tracing in Chrome browser")
    logging.info(f"2. Load trace file: {trace_path}_step*.json")
    logging.info(f"3. Analyze kernel timeline and identify bottlenecks")
    logging.info(f"4. Review summary: {output_dir / f'{args.model}_profile_summary.json'}")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
