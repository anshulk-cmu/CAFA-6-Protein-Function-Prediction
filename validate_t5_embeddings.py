#!/usr/bin/env python3
"""
Validation script for T5 embedding generation.
Tests ProtT5-XL and ProstT5 on a small sample to verify:
1. Models load correctly with encoder-only checkpoints
2. Embedding shapes are correct [N, 1024]
3. No NaN/Inf values in outputs
4. ProstT5 prefix tokens work properly
5. Memory usage is within A6000 48GB limits
"""

import os
import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel

# Import functions from generate_embeddings_t5
from generate_embeddings_t5 import (
    preprocess_sequences,
    add_prostt5_prefix,
    get_mean_embedding
)

def load_sample_sequences(fasta_path, n_samples=100):
    """Load first n_samples sequences from FASTA file"""
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    seq_str = "".join(current_seq)
                    sequences.append({
                        'protein_id': current_id,
                        'sequence': seq_str,
                        'length': len(seq_str)
                    })
                    if len(sequences) >= n_samples:
                        break
                current_id = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line)

        # Add last sequence if needed
        if current_id and len(sequences) < n_samples:
            seq_str = "".join(current_seq)
            sequences.append({
                'protein_id': current_id,
                'sequence': seq_str,
                'length': len(seq_str)
            })

    return sequences

def validate_model(model_name, model_key, device, test_sequences, batch_size=8):
    """Validate a single T5 model"""
    print(f"\n{'='*80}")
    print(f"Validating: {model_key}")
    print(f"Checkpoint: {model_name}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    # Step 1: Load model and tokenizer
    print("Step 1: Loading model and tokenizer...")
    try:
        tokenizer = T5Tokenizer.from_pretrained(
            model_name,
            do_lower_case=False,
            legacy=True
        )
        model = T5EncoderModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        model = model.to(device).eval()
        print(f"✓ Model loaded successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

    # Step 2: Test preprocessing
    print("\nStep 2: Testing sequence preprocessing...")
    try:
        batch_seqs = [item['sequence'] for item in test_sequences[:batch_size]]

        # Preprocess
        processed_seqs = preprocess_sequences(batch_seqs)
        print(f"✓ Preprocessing successful")
        print(f"  Original sample: {batch_seqs[0][:50]}...")
        print(f"  Processed sample: {processed_seqs[0][:100]}...")

        # Add prefix if ProstT5
        prefixed_seqs = add_prostt5_prefix(processed_seqs, model_key)
        if 'prost' in model_key.lower():
            print(f"✓ ProstT5 prefix added")
            print(f"  With prefix: {prefixed_seqs[0][:60]}...")
        else:
            print(f"✓ No prefix needed (ProtT5)")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False

    # Step 3: Test tokenization
    print("\nStep 3: Testing tokenization...")
    try:
        inputs = tokenizer(
            prefixed_seqs,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        ).to(device)
        print(f"✓ Tokenization successful")
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        return False

    # Step 4: Test embedding generation
    print("\nStep 4: Testing embedding generation...")
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_state = outputs.last_hidden_state

        print(f"✓ Forward pass successful")
        print(f"  Hidden state shape: {hidden_state.shape}")

        # Check for NaN/Inf
        if torch.isnan(hidden_state).any():
            print(f"✗ NaN values detected in hidden states!")
            return False
        if torch.isinf(hidden_state).any():
            print(f"✗ Inf values detected in hidden states!")
            return False
        print(f"✓ No NaN/Inf values detected")

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

    # Step 5: Test mean pooling
    print("\nStep 5: Testing mean pooling...")
    try:
        mean_embeddings = get_mean_embedding(
            hidden_state,
            inputs['attention_mask'],
            model_key
        )

        print(f"✓ Mean pooling successful")
        print(f"  Embedding shape: {mean_embeddings.shape}")
        print(f"  Expected shape: [{batch_size}, 1024]")

        # Validate shape
        if mean_embeddings.shape != (batch_size, 1024):
            print(f"✗ Incorrect embedding shape!")
            return False
        print(f"✓ Shape validation passed")

        # Check for NaN/Inf
        if torch.isnan(mean_embeddings).any():
            print(f"✗ NaN values detected in embeddings!")
            return False
        if torch.isinf(mean_embeddings).any():
            print(f"✗ Inf values detected in embeddings!")
            return False
        print(f"✓ No NaN/Inf values in embeddings")

        # Check embedding statistics
        emb_mean = mean_embeddings.mean().item()
        emb_std = mean_embeddings.std().item()
        emb_min = mean_embeddings.min().item()
        emb_max = mean_embeddings.max().item()

        print(f"  Embedding statistics:")
        print(f"    Mean: {emb_mean:.4f}")
        print(f"    Std:  {emb_std:.4f}")
        print(f"    Min:  {emb_min:.4f}")
        print(f"    Max:  {emb_max:.4f}")

    except Exception as e:
        print(f"✗ Mean pooling failed: {e}")
        return False

    # Step 6: Memory check
    print("\nStep 6: Checking GPU memory usage...")
    try:
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"✓ Memory usage:")
        print(f"  Allocated: {memory_allocated:.2f} GB")
        print(f"  Reserved:  {memory_reserved:.2f} GB")

        if memory_allocated > 45:
            print(f"⚠ Warning: High memory usage detected")
        else:
            print(f"✓ Memory usage within limits (< 48GB)")
    except Exception as e:
        print(f"⚠ Could not check memory: {e}")

    # Cleanup
    del model, tokenizer, inputs, outputs, hidden_state, mean_embeddings
    torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"✓ {model_key} validation PASSED")
    print(f"{'='*80}\n")

    return True

def main():
    print("\n" + "="*80)
    print("T5 EMBEDDING GENERATION VALIDATION")
    print("="*80 + "\n")

    # Load config
    config_path = "config_t5.yaml"
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load sample sequences
    data_root = Path(config['paths']['data_dir'])
    train_fasta = data_root / config['paths']['raw_train_sequences']

    print(f"Loading 100 sample sequences from: {train_fasta}")
    if not train_fasta.exists():
        print(f"✗ Error: FASTA file not found at {train_fasta}")
        print("Please update the path in config_t5.yaml or create a test FASTA file")
        sys.exit(1)

    test_sequences = load_sample_sequences(train_fasta, n_samples=100)
    print(f"✓ Loaded {len(test_sequences)} sequences")
    print(f"  Length range: {min(s['length'] for s in test_sequences)} - {max(s['length'] for s in test_sequences)} amino acids")

    # Validate each model
    results = {}
    for model_key, model_config in config['models'].items():
        model_name = model_config['name']
        device_id = model_config['device']
        device = torch.device(f"cuda:{device_id}")

        success = validate_model(
            model_name,
            model_key,
            device,
            test_sequences,
            batch_size=8
        )
        results[model_key] = success

    # Final summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")

    all_passed = all(results.values())

    for model_key, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {model_key}: {status}")

    print("\n" + "="*80)

    if all_passed:
        print("✓ ALL VALIDATIONS PASSED")
        print("You can now run the full embedding generation pipeline!")
        print("\nCommand to run:")
        print("  python generate_embeddings_t5.py --config config_t5.yaml")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("Please fix the issues above before running the full pipeline")
        sys.exit(1)

    print("="*80 + "\n")

if __name__ == "__main__":
    main()
