import numpy as np
import os
from pathlib import Path

def analyze_embeddings():
    emb_dir = Path("../embeddings")
    
    models = ['esm2_650m', 'esm2_3b', 'protbert', 'prott5']
    datasets = ['train', 'test']
    
    print("="*70)
    print("EMBEDDING ANALYSIS")
    print("="*70)
    
    for dataset in datasets:
        print(f"\n{dataset.upper()} SET:")
        print("-"*70)
        
        total_size = 0
        for model in models:
            emb_file = emb_dir / f"{dataset}_{model}_embeddings.npy"
            id_file = emb_dir / f"{dataset}_{model}_ids.npy"
            
            if emb_file.exists():
                emb = np.load(emb_file)
                ids = np.load(id_file)
                size_mb = emb_file.stat().st_size / 1e6
                total_size += size_mb
                
                print(f"  {model:15s}: {emb.shape[0]:6d} × {emb.shape[1]:4d} = "
                      f"{size_mb:6.1f}MB ✓")
            else:
                print(f"  {model:15s}: MISSING ✗")
        
        print(f"  {'TOTAL':15s}: {total_size:6.1f}MB")
    
    print("\n" + "="*70)
    
    ready_for_concat = all([
        (emb_dir / f"{ds}_{m}_embeddings.npy").exists() 
        for ds in datasets for m in models
    ])
    
    if ready_for_concat:
        print("✓ Ready for concatenation!")
        print("\nRun: python concat_embeddings_v2.py")
    else:
        print("⚠ Missing embeddings - continue generation")
    
    print("="*70)

if __name__ == "__main__":
    analyze_embeddings()