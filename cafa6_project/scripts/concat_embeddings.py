import numpy as np
import os
import time
from pathlib import Path

def concatenate_embeddings(emb_dir, prefix):
    """Concatenate embeddings from multiple models with validation"""
    models = ['esm2_650m', 'esm2_3b', 'protbert', 'esm2_150m']

    print(f"\nConcatenating {prefix.upper()} embeddings...")
    start_time = time.time()

    embeddings_list = []
    ids = None

    for model in models:
        emb_path = emb_dir / f"{prefix}_{model}_embeddings.npy"
        id_path = emb_dir / f"{prefix}_{model}_ids.npy"

        # Check if file exists
        if not emb_path.exists():
            print(f"✗ Missing: {model} embeddings")
            return False
        if not id_path.exists():
            print(f"✗ Missing: {model} IDs")
            return False

        print(f"  Loading {model}...", end='')

        try:
            emb = np.load(emb_path)
            model_ids = np.load(id_path)

            # Validate IDs match across models
            if ids is None:
                ids = model_ids
            else:
                if not np.array_equal(ids, model_ids):
                    print(f"\n✗ ERROR: ID mismatch for {model}!")
                    print(f"  Expected {len(ids)} IDs, got {len(model_ids)}")
                    # Check if just ordering is different
                    if set(ids) == set(model_ids):
                        print(f"  WARNING: IDs match but order is different!")
                    return False

            # Check for NaN or Inf values
            if np.isnan(emb).any():
                print(f"\n✗ ERROR: {model} contains NaN values!")
                return False
            if np.isinf(emb).any():
                print(f"\n✗ ERROR: {model} contains Inf values!")
                return False

            embeddings_list.append(emb)
            size_mb = emb_path.stat().st_size / 1e6
            print(f" {emb.shape} ({size_mb:.1f}MB) ✓")

        except Exception as e:
            print(f"\n✗ ERROR loading {model}: {e}")
            return False

    # Concatenate along feature dimension
    print(f"\n  Concatenating {len(embeddings_list)} embeddings...")
    try:
        concatenated = np.concatenate(embeddings_list, axis=1)
    except Exception as e:
        print(f"✗ ERROR during concatenation: {e}")
        return False

    # Save concatenated embeddings
    output_path = emb_dir / f"{prefix}_concatenated_embeddings.npy"
    output_ids_path = emb_dir / f"{prefix}_concatenated_ids.npy"

    try:
        np.save(output_path, concatenated)
        np.save(output_ids_path, ids)
    except Exception as e:
        print(f"✗ ERROR saving concatenated embeddings: {e}")
        return False

    # Calculate statistics
    elapsed_time = time.time() - start_time
    size_mb = output_path.stat().st_size / 1e6

    print(f"\n  ✓ Saved: {concatenated.shape}")
    print(f"    File size: {size_mb:.1f}MB")
    print(f"    Time: {elapsed_time:.2f}s")
    print(f"    Throughput: {size_mb/elapsed_time:.1f}MB/s")

    return True

def main():
    emb_dir = Path("../embeddings")
    
    print("="*60)
    print("EMBEDDING CONCATENATION")
    print("="*60)
    
    train_success = concatenate_embeddings(emb_dir, 'train')
    test_success = concatenate_embeddings(emb_dir, 'test')
    
    print("\n" + "="*60)
    if train_success and test_success:
        print("✓ ALL CONCATENATIONS COMPLETE")
    else:
        print("⚠ INCOMPLETE - check missing files")
    print("="*60)

if __name__ == "__main__":
    main()