#!/usr/bin/env python3
"""
GPU Benchmark for Phase 1B.

Generates embeddings using GPU processing with detailed performance metrics.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
import yaml

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent))
from utils.performance_logger import PerformanceLogger, format_time, format_memory, get_gpu_utilization


def setup_logging(log_path: str):
    """Setup logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def read_fasta(fasta_path: str):
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
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def preprocess_sequence_t5(sequence: str) -> str:
    """
    Preprocess sequence for T5 models.
    Replace rare amino acids and add spacing.
    """
    # Replace rare amino acids
    sequence = sequence.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    # Add spaces between amino acids
    return ' '.join(list(sequence))


def get_prefix_token(model_name: str) -> str:
    """Get prefix token for specific models."""
    if 'prost' in model_name.lower():
        return '<AA2fold>'
    return ''


def main():
    parser = argparse.ArgumentParser(description='GPU Benchmark for Phase 1B')
    parser.add_argument('--model', type=str, required=True,
                       choices=['esm2_3B', 'esm_c_600m', 'esm1b', 'prot_t5_xl', 'prost_t5'],
                       help='Model to benchmark')
    parser.add_argument('--input', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory')
    parser.add_argument('--gpu-id', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Config file for model specifications')
    parser.add_argument('--config-t5', type=str, default='config_t5.yaml',
                       help='Config file for T5 models')

    args = parser.parse_args()

    # Setup output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    embeddings_dir = Path('embeddings')
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_path = log_dir / f"benchmark_gpu_{args.model}.log"
    setup_logging(log_path)

    logging.info("=" * 70)
    logging.info("Phase 1B: GPU Benchmark")
    logging.info("=" * 70)
    logging.info(f"Model: {args.model}")
    logging.info(f"Input: {args.input}")
    logging.info(f"GPU ID: {args.gpu_id}")

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.error("CUDA not available!")
        sys.exit(1)

    device = torch.device(f'cuda:{args.gpu_id}')
    logging.info(f"Device: {device}")
    logging.info(f"GPU Name: {torch.cuda.get_device_name(args.gpu_id)}")

    # Initialize performance logger
    perf_logger = PerformanceLogger(f"gpu_benchmark_{args.model}")
    perf_logger.add_metadata('model', args.model)
    perf_logger.add_metadata('device', str(device))
    perf_logger.add_metadata('gpu_id', args.gpu_id)
    perf_logger.add_metadata('gpu_name', torch.cuda.get_device_name(args.gpu_id))
    perf_logger.add_metadata('input_file', args.input)

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
    batch_size = model_config.get('batch_size', 16)

    logging.info(f"Model name: {model_name}")
    logging.info(f"Batch size: {batch_size}")

    # Read sequences
    logging.info(f"\nReading sequences from {args.input}...")
    with perf_logger.timer("read_sequences"):
        sequences = read_fasta(args.input)
    logging.info(f"Loaded {len(sequences)} sequences")

    # Sort by length (same as production for fair comparison)
    sequences.sort(key=lambda x: len(x[1]))
    seq_lengths = [len(seq[1]) for seq in sequences]
    logging.info(f"Sequence length range: {min(seq_lengths)}-{max(seq_lengths)} aa")
    logging.info(f"Mean length: {np.mean(seq_lengths):.1f} aa")

    perf_logger.add_metadata('num_sequences', len(sequences))
    perf_logger.add_metadata('mean_seq_length', float(np.mean(seq_lengths)))
    perf_logger.add_metadata('max_seq_length', max(seq_lengths))

    # Load model and tokenizer
    logging.info(f"\nLoading model {model_name} on GPU...")

    with perf_logger.timer("load_model"):
        try:
            # Load tokenizer
            if 'esmplusplus' in model_name.lower() or 'synthyra' in model_name.lower():
                # ESMplusplus uses ESM2 tokenizer
                logging.info("Loading ESM2 tokenizer for ESMplusplus model")
                tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Load model (use FP16 for T5, FP32 for ESM)
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

    perf_logger.log_memory(device=device, label="after_model_load")

    # Log initial GPU utilization
    gpu_util = get_gpu_utilization(args.gpu_id)
    if gpu_util:
        logging.info(f"GPU Utilization: {gpu_util['gpu_util_percent']}%")
        logging.info(f"GPU Temperature: {gpu_util['temperature_c']}°C")
        logging.info(f"GPU Power: {gpu_util['power_watts']:.1f}W")

    # Get model prefix token if needed
    prefix_token = get_prefix_token(model_name)
    if prefix_token:
        logging.info(f"Using prefix token: {prefix_token}")

    # Process sequences in batches
    num_batches = (len(sequences) + batch_size - 1) // batch_size
    logging.info(f"\nProcessing {len(sequences)} sequences in {num_batches} batches")
    logging.info(f"Batch size: {batch_size}")

    all_embeddings = []
    total_inference_time = 0

    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc=f"{args.model} [GPU]"):
            batch_start_time = time.time()

            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sequences))
            batch_sequences = sequences[start_idx:end_idx]
            batch_seqs_only = [seq[1] for seq in batch_sequences]

            # Preprocess if T5
            if is_t5:
                batch_seqs_only = [preprocess_sequence_t5(seq) for seq in batch_seqs_only]
                if prefix_token:
                    batch_seqs_only = [f"{prefix_token} {seq}" for seq in batch_seqs_only]

            # Tokenize (CPU)
            tokenize_start = time.time()
            with perf_logger.timer("tokenize"):
                inputs = tokenizer(
                    batch_seqs_only,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
            tokenize_time = time.time() - tokenize_start

            max_seq_len = inputs['input_ids'].shape[1]

            # Transfer to GPU
            h2d_start = time.time()
            with perf_logger.timer("h2d_transfer"):
                inputs = {k: v.to(device) for k, v in inputs.items()}
            h2d_time = time.time() - h2d_start

            # Forward pass
            forward_start = time.time()
            with perf_logger.timer("forward"):
                outputs = model(**inputs)
            torch.cuda.synchronize(device)  # Wait for GPU to finish
            forward_time = time.time() - forward_start

            # Mean pooling (on GPU)
            with perf_logger.timer("mean_pool"):
                hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                attention_mask = inputs['attention_mask']  # [batch, seq_len]

                # Handle special tokens for T5
                if is_t5 and prefix_token:
                    # Exclude prefix token (first) and </s> token (last)
                    attention_mask = attention_mask.clone()
                    attention_mask[:, 0] = 0  # Mask prefix
                    attention_mask[:, -1] = 0  # Mask </s>

                # Mean pooling
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embeddings = sum_hidden / sum_mask

            # Transfer to CPU
            d2h_start = time.time()
            with perf_logger.timer("d2h_transfer"):
                embeddings_np = embeddings.cpu().numpy()
            d2h_time = time.time() - d2h_start

            all_embeddings.append(embeddings_np)

            # Record batch stats
            batch_time = time.time() - batch_start_time
            total_inference_time += batch_time

            perf_logger.add_batch_stat(
                batch_time=batch_time,
                batch_size=len(batch_sequences),
                max_seq_length=max_seq_len,
                forward_time=forward_time,
                tokenize_time=tokenize_time,
                h2d_time=h2d_time,
                d2h_time=d2h_time
            )

            # Log memory and GPU utilization every 10 batches
            if batch_idx % 10 == 0:
                perf_logger.log_memory(device=device, label=f"batch_{batch_idx}")
                gpu_util = get_gpu_utilization(args.gpu_id)
                if gpu_util:
                    perf_logger.add_metadata(f'gpu_util_batch_{batch_idx}', gpu_util)

    # Concatenate all embeddings
    logging.info("\nConcatenating embeddings...")
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    logging.info(f"Final embedding shape: {all_embeddings.shape}")

    # Save embeddings
    embeddings_path = embeddings_dir / f"benchmark_gpu_{args.model}_1k.pt"
    logging.info(f"Saving embeddings to {embeddings_path}")
    torch.save(torch.from_numpy(all_embeddings), embeddings_path)

    # Final memory snapshot
    perf_logger.log_memory(device=device, label="final")

    # Final GPU utilization
    gpu_util = get_gpu_utilization(args.gpu_id)
    if gpu_util:
        perf_logger.add_metadata('final_gpu_util', gpu_util)

    # Calculate summary statistics
    batch_stats = perf_logger.get_batch_stats()
    total_time = time.time() - perf_logger.start_time

    logging.info("\n" + "=" * 70)
    logging.info("Benchmark Complete")
    logging.info("=" * 70)
    logging.info(f"Total time: {format_time(total_time)}")
    logging.info(f"Inference time: {format_time(total_inference_time)}")
    logging.info(f"Mean batch time: {batch_stats['batch_time']['mean']:.4f}s")
    logging.info(f"Mean throughput: {batch_stats['throughput']['mean']:.2f} proteins/sec")
    logging.info(f"Total proteins: {batch_stats['total_items']}")
    logging.info(f"Peak GPU memory: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")

    # Save performance metrics
    metrics_path = output_dir / f"{args.model}_gpu_1k.json"
    logging.info(f"\nSaving metrics to {metrics_path}")
    perf_logger.export(metrics_path)

    # Print summary
    perf_logger.print_summary()

    logging.info("=" * 70)


if __name__ == '__main__':
    main()
