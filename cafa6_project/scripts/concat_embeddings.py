import numpy as np
import os

def concatenate_embeddings(emb_dir, prefix='train'):
    models = ['esm2_650m', 'esm2_3b', 'protbert', 'ankh']
    
    embeddings_list = []
    ids = None
    
    for model in models:
        emb_path = f"{emb_dir}/{prefix}_{model}_embeddings.npy"
        id_path = f"{emb_dir}/{prefix}_{model}_ids.npy"
        
        print(f"Loading {model}...")
        emb = np.load(emb_path)
        embeddings_list.append(emb)
        
        if ids is None:
            ids = np.load(id_path)
    
    concatenated = np.concatenate(embeddings_list, axis=1)
    
    output_path = f"{emb_dir}/{prefix}_concatenated_embeddings.npy"
    np.save(output_path, concatenated)
    np.save(f"{emb_dir}/{prefix}_concatenated_ids.npy", ids)
    
    print(f"Concatenated shape: {concatenated.shape}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    emb_dir = "../embeddings"
    
    print("Concatenating train embeddings...")
    concatenate_embeddings(emb_dir, 'train')
    
    print("\nConcatenating test embeddings...")
    concatenate_embeddings(emb_dir, 'test')
    
    print("\nDone!")