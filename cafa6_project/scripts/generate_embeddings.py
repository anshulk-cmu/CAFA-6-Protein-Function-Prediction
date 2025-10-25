import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import os
import platform
import argparse
import yaml
import json
import time
import logging
import random
from datetime import datetime
from pathlib import Path

def set_seed(seed, deterministic=False):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
        deterministic: If True, sets PyTorch to fully deterministic mode (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Full determinism (slower but 100% reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        # Balanced approach: reproducible but allows some non-deterministic optimizations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

class ProteinDataset(Dataset):
    def __init__(self, sequences, ids):
        self.data = sorted(zip(sequences, ids), key=lambda x: len(x[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, id_ = self.data[idx]
        return seq, id_, len(seq)

def collate_fn(batch):
    seqs, ids, lens = zip(*batch)
    return list(seqs), list(ids), list(lens)

class ProteinEmbedder:
    def __init__(self, model_name, device='cuda', batch_size=8, config=None, logger=None):
        self.device = device
        self.base_batch_size = batch_size  # Store original batch size
        self.batch_size = batch_size
        self.model_name = model_name
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        self.stats = {
            'model_name': model_name,
            'device': device,
            'batch_size': batch_size,
            'total_time': 0,
            'total_proteins': 0,
            'total_batches': 0,
            'batch_times': [],
            'warmup_batches': 0,
            'peak_memory_gb': 0,
            'avg_memory_gb': 0
        }

        # Dynamic batching settings
        self.use_dynamic_batching = config.get('optimization', {}).get('dynamic_batching', True)
        self.length_thresholds = {
            256: 1.0,   # 100% of batch_size for sequences < 256
            512: 0.5,   # 50% of batch_size for sequences < 512
            768: 0.33,  # 33% of batch_size for sequences < 768
            1024: 0.25  # 25% of batch_size for sequences < 1024
        }

        self.logger.info(f"Loading {model_name}...")

        try:
            if 'esm2' in model_name.lower():
                if '3b' in model_name.lower():
                    hf_name = "facebook/esm2_t36_3B_UR50D"
                elif '150m' in model_name.lower():
                    hf_name = "facebook/esm2_t30_150M_UR50D"
                else:
                    hf_name = "facebook/esm2_t33_650M_UR50D"
                self.model = EsmModel.from_pretrained(hf_name)
                self.tokenizer = EsmTokenizer.from_pretrained(hf_name)
            elif 'protbert' in model_name.lower():
                hf_name = "Rostlab/prot_bert_bfd"
                self.model = AutoModel.from_pretrained(hf_name)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Move to device and set to eval mode
        self.model = self.model.to(device).eval()

        # Apply half precision if using CUDA and enabled in config
        if device == 'cuda' and self.config.get('optimization', {}).get('use_half', True):
            self.model = self.model.half()
            self.logger.info("Half precision (fp16) enabled")

        # Compile model if enabled (non-Windows)
        if device == 'cuda' and self.config.get('optimization', {}).get('use_compile', True):
            if platform.system() != 'Windows':
                try:
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    self.logger.info("torch.compile() enabled")
                except Exception as e:
                    self.logger.warning(f"torch.compile() unavailable: {e}")
            else:
                self.logger.info("torch.compile() disabled (Windows)")

    def read_fasta(self, fasta_path):
        """Read FASTA file and return sequences and IDs"""
        sequences = []
        ids = []

        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequences.append(str(record.seq))
                ids.append(record.id)
        except Exception as e:
            self.logger.error(f"Failed to read FASTA file {fasta_path}: {e}")
            raise

        self.logger.info(f"Read {len(sequences):,} sequences from {fasta_path}")
        return ids, sequences

    def embed_batch(self, batch_seqs):
        """Generate embeddings for a batch of sequences"""
        # ProtBERT requires space-separated amino acids
        if 'protbert' in self.model_name.lower():
            batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]

        # Tokenize
        max_length = self.config.get('tokenizer', {}).get('max_length', 1024)
        inputs = self.tokenizer(batch_seqs, return_tensors='pt', padding=True,
                               truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference with autocast if CUDA
        with torch.no_grad():
            if self.device == 'cuda' and self.config.get('optimization', {}).get('use_amp', True):
                with torch.amp.autocast('cuda'):
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1)
            else:
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings.cpu().float().numpy()

    def get_dynamic_batch_size(self, max_seq_length):
        """
        Calculate optimal batch size based on maximum sequence length in batch.

        Memory usage for transformers scales as O(n²) due to self-attention.
        Longer sequences require exponentially more memory.
        """
        if not self.use_dynamic_batching:
            return self.base_batch_size

        # Find appropriate multiplier based on sequence length
        multiplier = 0.25  # Default for very long sequences (>1024)
        for threshold, mult in sorted(self.length_thresholds.items()):
            if max_seq_length < threshold:
                multiplier = mult
                break

        dynamic_size = max(1, int(self.base_batch_size * multiplier))
        return dynamic_size

    def save_checkpoint(self, checkpoint_path, batch_idx, processed_embeddings, processed_ids):
        """Save checkpoint for crash recovery"""
        checkpoint = {
            'batch_idx': batch_idx,
            'processed_count': len(processed_ids),
            'embeddings': processed_embeddings,
            'ids': processed_ids,
            'model_name': self.model_name,
            'timestamp': time.time()
        }

        temp_path = f"{checkpoint_path}.tmp"
        np.save(temp_path, checkpoint, allow_pickle=True)
        os.rename(temp_path, checkpoint_path)  # Atomic operation

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint if exists"""
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = np.load(checkpoint_path, allow_pickle=True).item()
                self.logger.info(f"Loaded checkpoint: batch {checkpoint['batch_idx']}, "
                               f"{checkpoint['processed_count']} proteins processed")
                return checkpoint
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
                return None
        return None

    def check_existing(self, output_path):
        """Check if embeddings already exist"""
        emb_file = f"{output_path}_embeddings.npy"
        id_file = f"{output_path}_ids.npy"
        if os.path.exists(emb_file) and os.path.exists(id_file):
            self.logger.info(f"Found existing embeddings: {output_path}")
            return True
        return False

    def generate_embeddings(self, fasta_path, output_path):
        """
        Generate embeddings for all sequences in FASTA file with optimizations:
        - Dynamic batch sizing based on sequence length
        - Checkpoint/resume support for crash recovery
        - Streaming embeddings to disk (memory efficient)
        - Aggressive memory management for long sequences
        """
        if self.check_existing(output_path):
            return self.stats

        self.logger.info(f"\nProcessing: {fasta_path}")

        # Read FASTA
        ids, sequences = self.read_fasta(fasta_path)
        self.stats['total_proteins'] = len(sequences)

        # Setup checkpoint
        checkpoint_path = f"{output_path}_checkpoint.npy"
        checkpoint = self.load_checkpoint(checkpoint_path)

        # Resume from checkpoint if exists
        if checkpoint:
            all_embeddings = checkpoint['embeddings']
            all_ids = checkpoint['ids']
            start_batch_idx = checkpoint['batch_idx'] + 1
            self.logger.info(f"Resuming from batch {start_batch_idx}")
        else:
            all_embeddings = []
            all_ids = []
            start_batch_idx = 0

        # Create sorted dataset (by length for efficient batching)
        dataset = ProteinDataset(sequences, ids)
        warmup_batches = self.config.get('optimization', {}).get('warmup_batches', 3)
        checkpoint_interval = self.config.get('optimization', {}).get('checkpoint_interval', 500)

        # Manual batching with dynamic sizing
        data_list = list(dataset)
        total_samples = len(data_list)

        # Calculate total batches (approximate)
        approx_batches = (total_samples + self.base_batch_size - 1) // self.base_batch_size

        # Progress bar
        pbar = tqdm(total=total_samples, desc=f"{self.model_name}")
        start_time = time.time()

        memory_samples = []
        batch_idx = 0
        current_idx = 0

        # Skip to checkpoint position if resuming
        if start_batch_idx > 0:
            # Estimate how many samples to skip
            samples_processed = len(all_ids)
            current_idx = samples_processed
            pbar.update(samples_processed)

        while current_idx < total_samples:
            batch_start = time.time()

            try:
                # Get next batch with dynamic sizing
                # Peek at sequence lengths to determine batch size
                peek_end = min(current_idx + self.base_batch_size, total_samples)
                peek_batch = data_list[current_idx:peek_end]
                max_len = max(len(seq) for seq, _, _ in peek_batch)

                # Calculate dynamic batch size
                dynamic_batch_size = self.get_dynamic_batch_size(max_len)

                # Get actual batch
                batch_end = min(current_idx + dynamic_batch_size, total_samples)
                batch_data = data_list[current_idx:batch_end]
                batch_seqs, batch_ids, batch_lens = zip(*batch_data)

                # Generate embeddings
                embs = self.embed_batch(batch_seqs)
                all_embeddings.append(embs)
                all_ids.extend(batch_ids)

                # Track memory if CUDA
                if self.device == 'cuda':
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    memory_samples.append(mem_gb)

                # Timing (exclude warmup batches)
                batch_time = time.time() - batch_start
                if batch_idx >= warmup_batches:
                    self.stats['batch_times'].append(batch_time)
                else:
                    self.stats['warmup_batches'] += 1

                # Update progress bar with stats
                proteins_processed = len(all_ids)
                elapsed_time = time.time() - start_time
                proteins_per_sec = proteins_processed / elapsed_time if elapsed_time > 0 else 0

                pbar.update(len(batch_seqs))
                pbar.set_postfix({
                    'batch': f'{batch_idx}/{approx_batches}',
                    'bs': dynamic_batch_size,
                    'max_len': max_len,
                    'prot/s': f'{proteins_per_sec:.1f}',
                    'mem': f'{mem_gb:.1f}GB' if self.device == 'cuda' else 'N/A'
                })

                # Aggressive memory management for long sequences
                if max_len > 512 or batch_idx % 100 == 0:
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()

                # Save checkpoint periodically
                if (batch_idx > 0 and batch_idx % checkpoint_interval == 0):
                    self.save_checkpoint(checkpoint_path, batch_idx, all_embeddings, all_ids)
                    self.logger.info(f"Checkpoint saved at batch {batch_idx}")

                # Update counters
                current_idx = batch_end
                batch_idx += 1

            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_idx}: {e}")
                # Save emergency checkpoint
                self.save_checkpoint(checkpoint_path, batch_idx, all_embeddings, all_ids)
                raise

        pbar.close()

        # Calculate final statistics
        self.stats['total_time'] = time.time() - start_time
        self.stats['total_batches'] = batch_idx
        
        if len(self.stats['batch_times']) > 0:
            self.stats['avg_batch_time'] = float(np.mean(self.stats['batch_times']))
            self.stats['std_batch_time'] = float(np.std(self.stats['batch_times']))
            self.stats['min_batch_time'] = float(np.min(self.stats['batch_times']))
            self.stats['max_batch_time'] = float(np.max(self.stats['batch_times']))
            
            # Throughput metrics
            self.stats['proteins_per_sec'] = self.stats['total_proteins'] / self.stats['total_time']
            self.stats['batches_per_sec'] = self.stats['total_batches'] / self.stats['total_time']
            
            # Estimate tokens processed
            avg_seq_len = np.mean([len(seq) for seq in sequences])
            total_tokens = self.stats['total_proteins'] * avg_seq_len
            self.stats['tokens_per_sec'] = total_tokens / self.stats['total_time']

        # Memory statistics
        if memory_samples:
            self.stats['peak_memory_gb'] = float(np.max(memory_samples))
            self.stats['avg_memory_gb'] = float(np.mean(memory_samples))
            if self.device == 'cuda':
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.stats['memory_utilization_pct'] = (self.stats['peak_memory_gb'] / total_memory) * 100

        # Concatenate all embeddings
        embeddings_array = np.vstack(all_embeddings)
        self.logger.info(f"Reordering embeddings to original order...")
        
        # Create mapping from sorted IDs back to original order
        id_to_embedding = {id_: emb for id_, emb in zip(all_ids, embeddings_array)}
        reordered_embeddings = np.array([id_to_embedding[id_] for id_ in ids])
        
        # Save embeddings and IDs
        np.save(f"{output_path}_embeddings.npy", reordered_embeddings)
        np.save(f"{output_path}_ids.npy", np.array(ids))
        
        self.logger.info(f"Saved embeddings: {reordered_embeddings.shape}")

        # Print summary
        self._print_summary()

        # Cleanup
        del all_embeddings, embeddings_array, reordered_embeddings
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        # Remove checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            self.logger.info("Checkpoint file removed after successful completion")

        return self.stats

    def _print_summary(self):
        """Print performance summary"""
        self.logger.info("\n" + "="*70)
        self.logger.info(f"PERFORMANCE SUMMARY: {self.model_name}")
        self.logger.info("="*70)
        self.logger.info(f"Total proteins: {self.stats['total_proteins']:,}")
        self.logger.info(f"Total batches: {self.stats['total_batches']:,} (warmup: {self.stats['warmup_batches']})")
        self.logger.info(f"Total time: {self.stats['total_time']:.2f}s ({self.stats['total_time']/60:.1f}m)")

        if 'proteins_per_sec' in self.stats:
            self.logger.info(f"\nThroughput:")
            self.logger.info(f"  Proteins/sec: {self.stats['proteins_per_sec']:.1f}")
            self.logger.info(f"  Batches/sec: {self.stats['batches_per_sec']:.2f}")
            self.logger.info(f"  Tokens/sec: {self.stats['tokens_per_sec']:.1f}")

        if 'avg_batch_time' in self.stats:
            self.logger.info(f"\nBatch timing:")
            self.logger.info(f"  Average: {self.stats['avg_batch_time']:.3f}s")
            self.logger.info(f"  Std dev: {self.stats['std_batch_time']:.3f}s")
            self.logger.info(f"  Min: {self.stats['min_batch_time']:.3f}s")
            self.logger.info(f"  Max: {self.stats['max_batch_time']:.3f}s")

        if self.stats['peak_memory_gb'] > 0:
            self.logger.info(f"\nMemory usage:")
            self.logger.info(f"  Peak: {self.stats['peak_memory_gb']:.2f}GB")
            self.logger.info(f"  Average: {self.stats['avg_memory_gb']:.2f}GB")
            if 'memory_utilization_pct' in self.stats:
                self.logger.info(f"  Utilization: {self.stats['memory_utilization_pct']:.1f}%")

        self.logger.info("="*70 + "\n")

def setup_logging(config, args):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    # Create logs directory
    if log_config.get('save_to_file', True):
        log_dir = Path(config['paths']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"embedding_generation_{args.device}_{timestamp}.log"

        # Configure logging
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler() if log_config.get('console_output', True) else logging.NullHandler()
            ]
        )

        logging.info(f"Log file: {log_file}")
    else:
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate protein embeddings from multiple models')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use for inference (default: cuda)')
    parser.add_argument('--config', type=str, default='../config.yaml',
                       help='Path to config file (default: ../config.yaml)')
    parser.add_argument('--models', nargs='+',
                       choices=['esm2_650m', 'esm2_3b', 'protbert', 'esm2_150m'],
                       help='Specific models to run (default: all)')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration...")
        config = {'paths': {}, 'models': {}, 'optimization': {}, 'logging': {}}
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config, args)

    # Set random seeds for reproducibility
    seed = config.get('reproducibility', {}).get('seed', 42)
    deterministic = config.get('reproducibility', {}).get('cudnn_deterministic', False)
    set_seed(seed, deterministic)
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    logger.info(f"Device: {args.device}")

    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Apply CUDA optimizations (respecting reproducibility settings)
        cudnn_benchmark = config.get('reproducibility', {}).get('cudnn_benchmark', True)
        if cudnn_benchmark and not deterministic:
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled")
        elif deterministic:
            logger.info("cuDNN benchmark disabled (deterministic mode)")

        if config.get('optimization', {}).get('tf32_matmul', True):
            torch.set_float32_matmul_precision('high')
            logger.info("TensorFloat32 matmul enabled")

    # Setup paths
    data_dir = Path(config.get('paths', {}).get('data_dir', '../data'))
    emb_dir = Path(config.get('paths', {}).get('embeddings_dir', '../embeddings'))
    outputs_dir = Path(config.get('paths', {}).get('outputs_dir', '../outputs'))

    emb_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_fasta = data_dir / config.get('datasets', {}).get('train', {}).get('filename', 'train_sequences.fasta')
    test_fasta = data_dir / config.get('datasets', {}).get('test', {}).get('filename', 'testsuperset.fasta')

    # Determine which models to run
    if args.models:
        model_names = args.models
    else:
        model_names = ['esm2_650m', 'esm2_3b', 'protbert', 'esm2_150m']

    # Load model configs
    model_configs = config.get('models', {})
    configs = []
    for model_name in model_names:
        if model_name in model_configs:
            batch_size = model_configs[model_name].get('batch_size', 8)
        else:
            # Default batch sizes
            batch_sizes = {'esm2_650m': 16, 'esm2_3b': 4, 'protbert': 12, 'esm2_150m': 24}
            batch_size = batch_sizes.get(model_name, 8)
        configs.append((model_name, batch_size))

    # Track all performance stats
    all_stats = {}

    # Process each model
    for model_name, batch_size in configs:
        logger.info(f"\n{'='*70}")
        logger.info(f"MODEL: {model_name.upper()}")
        logger.info(f"{'='*70}")

        try:
            embedder = ProteinEmbedder(
                model_name,
                device=args.device,
                batch_size=batch_size,
                config=config,
                logger=logger
            )

            # Generate embeddings for train and test
            train_stats = embedder.generate_embeddings(
                str(train_fasta),
                str(emb_dir / f"train_{model_name}")
            )

            test_stats = embedder.generate_embeddings(
                str(test_fasta),
                str(emb_dir / f"test_{model_name}")
            )

            all_stats[model_name] = {
                'train': train_stats,
                'test': test_stats
            }

            del embedder
            if args.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to process model {model_name}: {e}")
            continue

    # Save performance report
    if config.get('logging', {}).get('json_report', True):
        report_path = outputs_dir / f"embedding_performance_report_{args.device}.json"
        with open(report_path, 'w') as f:
            json.dump(all_stats, f, indent=2, default=str)
        logger.info(f"\nPerformance report saved: {report_path}")

    logger.info("\n" + "="*70)
    logger.info("ALL EMBEDDINGS COMPLETE")
    logger.info("="*70)

if __name__ == "__main__":
    main()