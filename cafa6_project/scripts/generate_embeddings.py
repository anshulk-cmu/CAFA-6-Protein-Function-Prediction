import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import os
import platform

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
    def __init__(self, model_name, device='cuda', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        
        print(f"Loading {model_name}...")
        if 'esm2' in model_name.lower():
            if '3b' in model_name.lower():
                self.model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D")
                self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            else:
                self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
                self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        elif 'protbert' in model_name.lower():
            self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
            self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        elif 'ankh' in model_name.lower():
            self.model = AutoModel.from_pretrained("ElnaggarLab/ankh-large")
            self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")
        
        self.model = self.model.to(device).eval().half()
        
        if platform.system() != 'Windows':
            try:
                self.model = torch.compile(self.model, mode='reduce-overhead')
                print("✓ Torch compile enabled")
            except:
                print("⚠ Torch compile unavailable")
        else:
            print("⚠ Torch compile disabled (Windows)")
    
    def read_fasta(self, fasta_path):
        sequences = []
        ids = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            ids.append(record.id)
        return ids, sequences
    
    def embed_batch(self, batch_seqs):
        if 'protbert' in self.model_name.lower():
            batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
        
        inputs = self.tokenizer(batch_seqs, return_tensors='pt', padding=True, 
                               truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().float().numpy()
    
    def check_existing(self, output_path):
        emb_file = f"{output_path}_embeddings.npy"
        id_file = f"{output_path}_ids.npy"
        if os.path.exists(emb_file) and os.path.exists(id_file):
            print(f"✓ Found existing: {output_path}")
            return True
        return False
    
    def generate_embeddings(self, fasta_path, output_path):
        if self.check_existing(output_path):
            return
        
        print(f"\nProcessing: {fasta_path}")
        ids, sequences = self.read_fasta(fasta_path)
        
        dataset = ProteinDataset(sequences, ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                              shuffle=False, collate_fn=collate_fn,
                              pin_memory=True, num_workers=0)
        
        all_embeddings = []
        all_ids = []
        
        for batch_seqs, batch_ids, _ in tqdm(dataloader, desc=f"{self.model_name}"):
            embs = self.embed_batch(batch_seqs)
            all_embeddings.append(embs)
            all_ids.extend(batch_ids)
            
            if len(all_embeddings) % 50 == 0:
                torch.cuda.empty_cache()
        
        original_order = {id_: i for i, id_ in enumerate(ids)}
        sorted_indices = [original_order[id_] for id_ in all_ids]
        
        embeddings_array = np.vstack(all_embeddings)
        reordered_embeddings = np.zeros_like(embeddings_array)
        for new_idx, orig_idx in enumerate(sorted_indices):
            reordered_embeddings[orig_idx] = embeddings_array[new_idx]
        
        np.save(f"{output_path}_embeddings.npy", reordered_embeddings)
        np.save(f"{output_path}_ids.npy", np.array(ids))
        
        print(f"✓ Saved {reordered_embeddings.shape}")
        
        del all_embeddings, embeddings_array, reordered_embeddings
        torch.cuda.empty_cache()
        gc.collect()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    
    data_dir = "../data"
    emb_dir = "../embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    
    train_fasta = f"{data_dir}/train_sequences.fasta"
    test_fasta = f"{data_dir}/testsuperset.fasta"
    
    configs = [
        ('esm2_650m', 32),
        ('esm2_3b', 8),
        ('protbert', 24),
        ('ankh', 24)
    ]
    
    for model_name, batch_size in configs:
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name.upper()}")
        print(f"{'='*60}")
        
        embedder = ProteinEmbedder(model_name, device=device, batch_size=batch_size)
        
        embedder.generate_embeddings(train_fasta, f"{emb_dir}/train_{model_name}")
        embedder.generate_embeddings(test_fasta, f"{emb_dir}/test_{model_name}")
        
        del embedder
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\n" + "="*60)
    print("✓ ALL EMBEDDINGS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()