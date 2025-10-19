import numpy as np
import os
from pathlib import Path

def concatenate_embeddings(emb_dir, prefix):
    models = ['esm2_650m', 'esm2_3b', 'protbert', 'ankh']
    
    print(f"\nConcatenating {prefix.upper()} embeddings...")
    
    embeddings_list = []
    ids = None
    
    for model in models:
        emb_path = emb_dir / f"{prefix}_{model}_embeddings.npy"
        id_path = emb_dir / f"{prefix}_{model}_ids.npy"
        
        if not emb_path.exists():
            print(f"✗ Missing: {model}")
            return False
        
        print(f"  Loading {model}...", end='')
        emb = np.load(emb_path)
        embeddings_list.append(emb)
        
        if ids is None:
            ids = np.load(id_path)
        
        print(f" {emb.shape}")
    
    concatenated = np.concatenate(embeddings_list, axis=1)
    
    output_path = emb_dir / f"{prefix}_concatenated_embeddings.npy"
    np.save(output_path, concatenated)
    np.save(emb_dir / f"{prefix}_concatenated_ids.npy", ids)
    
    size_mb = output_path.stat().st_size / 1e6
    print(f"  ✓ Saved: {concatenated.shape} ({size_mb:.1f}MB)")
    
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