import os
import yaml
import logging
import argparse
import random
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import gc
import torch.multiprocessing as mp
from transformers import T5Tokenizer, T5EncoderModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(log_dir, worker_name=None):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{worker_name}" if worker_name else ""
    log_file = os.path.join(log_dir, f"embedding_generation{suffix}_{timestamp}.log")
    
    logger = logging.getLogger(worker_name or "Main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def get_model_and_tokenizer(model_name, device):
    """Load T5 encoder-only models (ProtT5-XL, ProstT5)"""
    tokenizer = T5Tokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        legacy=True  # Handle transformers 4.30+ compatibility
    )
    model = T5EncoderModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )
    model = model.to(device).eval()
    return model, tokenizer

def preprocess_sequences(sequences):
    """
    Preprocess protein sequences for T5 models:
    - Replace rare amino acids (U, Z, O, B) with X
    - Add spaces between amino acids
    """
    processed = []
    for seq in sequences:
        # Replace rare amino acids
        seq = re.sub(r"[UZOB]", "X", seq)
        # Add spaces between amino acids
        seq = " ".join(list(seq))
        processed.append(seq)
    return processed

def add_prostt5_prefix(sequences, model_key):
    """
    Add task-specific prefix for ProstT5 bilingual model.

    ProstT5 requires <AA2fold> prefix to activate structure-aware embeddings.
    ProtT5 doesn't need prefix tokens.

    Args:
        sequences: List of space-separated amino acid sequences
        model_key: Model identifier from config (e.g., 'prost_t5', 'prot_t5_xl')

    Returns:
        List of sequences with prefix added if ProstT5
    """
    if 'prost' in model_key.lower():
        prefix = "<AA2fold>"
        return [f"{prefix} {seq}" for seq in sequences]
    else:
        # ProtT5 doesn't need prefix, return as-is
        return sequences

def get_mean_embedding(last_hidden_state, attention_mask, model_key):
    """
    Calculate mean pooling over sequence length, excluding special tokens.

    T5 models add special tokens:
    - ProstT5: <AA2fold> prefix at position 0, </s> at end
    - ProtT5: </s> at end only

    Args:
        last_hidden_state: [batch_size, seq_len, hidden_dim]
        attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        model_key: Model identifier to determine token structure

    Returns:
        Mean-pooled embeddings [batch_size, hidden_dim]
    """
    if 'prost' in model_key.lower():
        # ProstT5: Remove prefix token at position 0
        # The prefix <AA2fold> is at position 0 after tokenization
        hidden_state = last_hidden_state[:, 1:, :]  # Remove position 0
        mask = attention_mask[:, 1:]  # Adjust mask accordingly
    else:
        # ProtT5: No prefix, keep all positions
        hidden_state = last_hidden_state
        mask = attention_mask

    # The </s> token is included in attention_mask but should be excluded
    # Get sequence lengths (includes </s>)
    seq_lengths = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]

    # Mask expansion for broadcasting
    mask_expanded = mask.unsqueeze(-1).expand(hidden_state.size()).float()

    # Sum embeddings, excluding padding (mask=0 positions)
    sum_embeddings = torch.sum(hidden_state * mask_expanded, dim=1)

    # Divide by length - 1 to exclude </s> token
    # For ProstT5, we already removed prefix, so just subtract 1 for </s>
    # For ProtT5, subtract 1 for </s>
    mean_embeddings = sum_embeddings / torch.clamp(seq_lengths - 1, min=1e-9)

    return mean_embeddings

def load_and_sort_fasta(fasta_path, logger):
    logger.info(f"Loading FASTA file: {fasta_path}")
    sequences = []
    current_id = None
    current_seq = []
    
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seq_str = "".join(current_seq)
                    sequences.append((current_id, seq_str, len(seq_str)))
                current_id = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)
        
        if current_id:
            seq_str = "".join(current_seq)
            sequences.append((current_id, seq_str, len(seq_str)))
    
    logger.info(f"Loaded {len(sequences)} sequences, sorting by length")
    sequences.sort(key=lambda x: x[2])
    
    sorted_data = [{'protein_id': pid, 'sequence': seq} for pid, seq, _ in sequences]
    logger.info("Sorting complete")
    return sorted_data

def worker_fn(model_key, config, data_list, data_key, seed):
    set_seed(seed)
    
    model_config = config['models'][model_key]
    model_name = model_config['name']
    device = torch.device(f"cuda:{model_config['device']}")
    batch_size = config['run_params']['batch_size']
    save_every = config['run_params']['save_every_n_batches']
    
    worker_name = f"{model_key}_{data_key}"
    logger = setup_logging(config['paths']['log_dir'], worker_name)
    logger.info(f"Worker started on {device} for model {model_name}")
    logger.info(f"Model type: T5 Encoder-Only ({model_key})")
    logger.info(f"Batch size: {batch_size} (optimized for A6000 48GB)")
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{data_key}_embeddings_{model_config['output_key']}.pt"
    output_path = output_dir / output_filename
    chk_path = output_dir / f"{output_filename}.chk"
    
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, device)
        logger.info("Model loaded successfully in FP16 precision")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    all_embeddings = []
    processed_count = 0
    
    if os.path.exists(chk_path):
        try:
            checkpoint = torch.load(chk_path, map_location='cpu', weights_only=False)
            all_embeddings = checkpoint['embeddings']
            processed_count = checkpoint['processed_count']
            logger.info(f"Resuming from checkpoint: {processed_count} proteins already processed")
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}, starting from scratch")
            all_embeddings = []
            processed_count = 0
    
    data_list = data_list[processed_count:]
    if not data_list:
        logger.info("All data already processed")
        return
    
    total_batches = (len(data_list) + batch_size - 1) // batch_size
    logger.info(f"Processing {len(data_list)} sequences in {total_batches} batches")
    
    pbar = tqdm(range(total_batches), desc=f"{model_key} [{data_key}]", position=model_config['device'])
    
    for i in pbar:
        batch_data = data_list[i * batch_size: (i + 1) * batch_size]
        if not batch_data:
            continue
        
        batch_seqs = [item['sequence'] for item in batch_data]
        batch_ids = [item['protein_id'] for item in batch_data]

        # Step 1: Preprocess sequences - replace rare AAs (U,Z,O,B) and add spaces
        batch_seqs = preprocess_sequences(batch_seqs)

        # Step 2: Add ProstT5 prefix token if needed (<AA2fold> for structure-aware embeddings)
        batch_seqs = add_prostt5_prefix(batch_seqs, model_key)

        try:
            with torch.no_grad():
                inputs = tokenizer(
                    batch_seqs,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=1024
                ).to(device)

                outputs = model(**inputs)
                hidden_state = outputs.last_hidden_state

                # Step 3: Mean pooling excluding special tokens (</s> and prefix if ProstT5)
                mean_embeddings = get_mean_embedding(
                    hidden_state,
                    inputs['attention_mask'],
                    model_key
                )
                all_embeddings.append(mean_embeddings.detach().cpu().half())
                processed_count += len(batch_data)
            
            if (i + 1) % save_every == 0:
                checkpoint = {
                    'embeddings': all_embeddings,
                    'processed_count': processed_count
                }
                torch.save(checkpoint, chk_path)
                logger.info(f"Checkpoint saved at batch {i+1}/{total_batches}")
            
            del inputs, outputs, hidden_state, mean_embeddings
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.error(f"OOM at batch {i}, clearing cache and skipping")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Error at batch {i}: {e}")
                continue
        except Exception as e:
            logger.error(f"Error at batch {i}: {e}, skipping batch")
            continue
    
    try:
        if all_embeddings:
            final_embeddings = torch.cat(all_embeddings, dim=0)
            torch.save(final_embeddings, output_path)
            logger.info(f"Saved embeddings to {output_path}, shape: {final_embeddings.shape}")
            
            if os.path.exists(chk_path):
                os.remove(chk_path)
                logger.info("Checkpoint file removed")
        else:
            logger.warning("No embeddings generated")
    except Exception as e:
        logger.error(f"Failed to save final embeddings: {e}")
    
    logger.info(f"Worker {worker_name} finished")
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

def main(config_path):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    config = load_config(config_path)
    logger = setup_logging(config['paths']['log_dir'])
    logger.info("Embedding generation started - T5 Encoder-Only models (ProtT5-XL, ProstT5)")
    logger.info(f"Config loaded from {config_path}")
    logger.info(f"Using encoder-only checkpoints for 2x speed and 50% memory reduction")
    
    seed = config['run_params'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Random seed set to {seed}")
    
    data_root = Path(config['paths']['data_dir'])
    train_fasta_path = data_root / config['paths']['raw_train_sequences']
    test_fasta_path = data_root / config['paths']['raw_test_sequences']
    
    train_data = load_and_sort_fasta(train_fasta_path, logger)
    test_data = load_and_sort_fasta(test_fasta_path, logger)
    
    if not train_data or not test_data:
        logger.error("Failed to load FASTA files")
        return
    
    logger.info(f"Train: {len(train_data)} sequences, Test: {len(test_data)} sequences")
    
    # Group T5 models by GPU
    model_groups = [
        ['prot_t5_xl', 'prost_t5']  # Both on GPU 0 (can run sequentially or parallel based on memory)
    ]
    
    for group_idx, group in enumerate(model_groups, 1):
        logger.info(f"Starting model group {group_idx}/{len(model_groups)}: {group}")
        
        for data_key, data_list in [('train', train_data), ('test', test_data)]:
            logger.info(f"Processing {data_key} dataset for group {group_idx}")
            processes = []
            
            for model_key in group:
                if model_key not in config['models']:
                    logger.warning(f"Model {model_key} not found in config, skipping")
                    continue
                
                p = mp.Process(
                    target=worker_fn,
                    args=(model_key, config, data_list, data_key, seed)
                )
                processes.append(p)
                p.start()
            
            for p in processes:
                p.join()
            
            logger.info(f"Completed {data_key} dataset for group {group_idx}")
            torch.cuda.empty_cache()
        
        logger.info(f"Completed model group {group_idx}/{len(model_groups)}")

    logger.info("T5 encoder-only embedding generation finished successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAFA-6 T5 Encoder-Only Embedding Generation (ProtT5-XL, ProstT5)")
    parser.add_argument("--config", type=str, default="config_t5.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
