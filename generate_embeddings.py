import os
import yaml
import logging
import argparse
import random
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel, T5EncoderModel

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

def get_model_and_tokenizer(model_name, repo_id, device):
    if repo_id == 't5':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)
    elif repo_id == 'ankh':
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    
    model = model.to(device).eval().half()
    return model, tokenizer

def get_mean_embedding(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

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
    
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{data_key}_embeddings_{model_config['output_key']}.pt"
    output_path = output_dir / output_filename
    chk_path = output_dir / f"{output_filename}.chk"
    
    try:
        model, tokenizer = get_model_and_tokenizer(model_name, model_config['repo_id'], device)
        logger.info("Model loaded successfully")
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
                
                if model_config['repo_id'] == 't5':
                    hidden_state = outputs.last_hidden_state
                else:
                    hidden_state = outputs.last_hidden_state
                
                mean_embeddings = get_mean_embedding(hidden_state, inputs['attention_mask'])
                all_embeddings.append(mean_embeddings.detach().cpu().half())
                processed_count += len(batch_data)
            
            if (i + 1) % save_every == 0:
                checkpoint = {
                    'embeddings': all_embeddings,
                    'processed_count': processed_count
                }
                torch.save(checkpoint, chk_path)
            
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

def main(config_path):
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    config = load_config(config_path)
    logger = setup_logging(config['paths']['log_dir'])
    logger.info("Embedding generation started")
    logger.info(f"Config loaded from {config_path}")
    
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
    
    model_groups = [
        ['esm2_3B', 'esm_c_600m'],
        ['prot_t5_xl', 'prot_bert_bfd']
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
    
    logger.info("Embedding generation finished")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAFA-6 Embedding Generation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
