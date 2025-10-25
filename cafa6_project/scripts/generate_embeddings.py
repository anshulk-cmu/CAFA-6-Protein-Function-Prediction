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
from datetime import datetime
from pathlib import Path

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

        self.logger.info(f"Loading {model_name}...")

        try:
            if 'esm2' in model_name.lower():
                if '3b' in model_name.lower():
                    hf_name = "facebook/esm2_t36_3B_UR50D"
                else:
                    hf_name = "facebook/esm2_t33_650M_UR50D"
                self.model = EsmModel.from_pretrained(hf_name)
                self.tokenizer = EsmTokenizer.from_pretrained(hf_name)
            elif 'protbert' in model_name.lower():
                hf_name = "Rostlab/prot_bert_bfd"
                self.model = AutoModel.from_pretrained(hf_name)
                self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            elif 'prott5' in model_name.lower():
                hf_name = "Rostlab/prot_t5_xl_uniref50"
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

    def check_existing(self, output_path):
        """Check if embeddings already exist"""
        emb_file = f"{output_path}_embeddings.npy"
        id_file = f"{output_path}_ids.npy"
        if os.path.exists(emb_file) and os.path.exists(id_file):
            self.logger.info(f"Found existing embeddings: {output_path}")
            return True
        return False

    def generate_embeddings(self, fasta_path, output_path):
        """Generate embeddings for all sequences in FASTA file"""
        if self.check_existing(output_path):
            return self.stats

        self.logger.info(f"\nProcessing: {fasta_path}")

        # Read FASTA
        ids, sequences = self.read_fasta(fasta_path)
        self.stats['total_proteins'] = len(sequences)

        # Create dataset and dataloader
        dataset = ProteinDataset(sequences, ids)
        pin_memory = self.config.get('optimization', {}).get('pin_memory', True)
        num_workers = self.config.get('optimization', {}).get('num_workers', 0)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=pin_memory and self.device == 'cuda',
            num_workers=num_workers
        )

        all_embeddings = []
        all_ids = []
        warmup_batches = self.config.get('optimization', {}).get('warmup_batches', 3)
        cache_clear_interval = self.config.get('optimization', {}).get('cache_clear_interval', 50)

        # Progress bar
        pbar = tqdm(dataloader, desc=f"{self.model_name}")
        start_time = time.time()

        memory_samples = []

        for batch_idx, (batch_seqs, batch_ids, batch_lens) in enumerate(pbar):
            batch_start = time.time()

            try:
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
                if batch_idx >= warmup_batches and len(self.stats['batch_times']) > 0:
                    avg_batch_time = np.mean(self.stats['batch_times'])
                    batches_per_sec = 1.0 / avg_batch_time if avg_batch_time > 0 else 0
                    proteins_per_sec = batches_per_sec * self.batch_size

                    pbar.set_postfix({
                        'batch/s': f'{batches_per_sec:.2f}',
                        'prot/s': f'{proteins_per_sec:.1f}',
                        'mem_gb': f'{memory_samples[-1]:.1f}' if memory_samples else 'N/A'
                    })

                # Clear CUDA cache periodically
                if self.device == 'cuda' and batch_idx > 0 and batch_idx % cache_clear_interval == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                self.logger.error(f"Failed to process batch {batch_idx}: {e}")
                raise

        pbar.close()

        # Calculate total time
        self.stats['total_time'] = time.time() - start_time
        self.stats['total_batches'] = len(dataloader)

        # Calculate memory stats
        if memory_samples:
            self.stats['peak_memory_gb'] = max(memory_samples)
            self.stats['avg_memory_gb'] = np.mean(memory_samples)
            if self.device == 'cuda':
                self.stats['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.stats['memory_utilization_pct'] = (self.stats['peak_memory_gb'] / self.stats['total_memory_gb']) * 100

        # Reorder embeddings to original sequence order
        self.logger.info("Reordering embeddings to original order...")
        original_order = {id_: i for i, id_ in enumerate(ids)}
        sorted_indices = [original_order[id_] for id_ in all_ids]

        embeddings_array = np.vstack(all_embeddings)
        reordered_embeddings = np.zeros_like(embeddings_array)
        for new_idx, orig_idx in enumerate(sorted_indices):
            reordered_embeddings[orig_idx] = embeddings_array[new_idx]

        # Save embeddings
        np.save(f"{output_path}_embeddings.npy", reordered_embeddings)
        np.save(f"{output_path}_ids.npy", np.array(ids))

        self.logger.info(f"Saved embeddings: {reordered_embeddings.shape}")

        # Calculate throughput metrics
        if self.stats['total_time'] > 0:
            self.stats['proteins_per_sec'] = self.stats['total_proteins'] / self.stats['total_time']
            self.stats['batches_per_sec'] = self.stats['total_batches'] / self.stats['total_time']

            # Estimate tokens processed
            avg_seq_len = np.mean([len(seq) for seq in sequences])
            total_tokens = self.stats['total_proteins'] * avg_seq_len
            self.stats['tokens_per_sec'] = total_tokens / self.stats['total_time']
            self.stats['avg_sequence_length'] = avg_seq_len

        # Batch timing statistics
        if self.stats['batch_times']:
            self.stats['avg_batch_time'] = np.mean(self.stats['batch_times'])
            self.stats['std_batch_time'] = np.std(self.stats['batch_times'])
            self.stats['min_batch_time'] = np.min(self.stats['batch_times'])
            self.stats['max_batch_time'] = np.max(self.stats['batch_times'])

        # Print summary
        self._print_summary()

        # Cleanup
        del all_embeddings, embeddings_array, reordered_embeddings
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

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
                       choices=['esm2_650m', 'esm2_3b', 'protbert', 'prott5'],
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

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.error("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'

    logger.info(f"Device: {args.device}")

    if args.device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

        # Apply CUDA optimizations
        if config.get('optimization', {}).get('cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled")

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
        model_names = ['esm2_650m', 'esm2_3b', 'protbert', 'prott5']

    # Load model configs
    model_configs = config.get('models', {})
    configs = []
    for model_name in model_names:
        if model_name in model_configs:
            batch_size = model_configs[model_name].get('batch_size', 8)
        else:
            # Default batch sizes
            batch_sizes = {'esm2_650m': 16, 'esm2_3b': 4, 'protbert': 12, 'prott5': 8}
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
