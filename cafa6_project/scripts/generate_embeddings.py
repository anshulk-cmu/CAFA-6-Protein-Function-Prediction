import torch
import numpy as np
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModel, EsmModel, EsmTokenizer
from tqdm import tqdm
import gc
import os

class ProteinEmbedder:
    def __init__(self, model_name, device='cuda', batch_size=8):
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        
        if 'esm2' in model_name.lower():
            if '3b' in model_name.lower():
                self.model = EsmModel.from_pretrained("facebook/esm2_t36_3B_UR50D").to(device).eval()
                self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
            else:
                self.model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device).eval()
                self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        elif 'protbert' in model_name.lower():
            self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd").to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd")
        elif 'ankh' in model_name.lower():
            self.model = AutoModel.from_pretrained("ElnaggarLab/ankh-large").to(device).eval()
            self.tokenizer = AutoTokenizer.from_pretrained("ElnaggarLab/ankh-large")
        
        self.model.half()
    
    def read_fasta(self, fasta_path):
        sequences = []
        ids = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
            ids.append(record.id)
        return ids, sequences
    
    def batch_sequences(self, sequences):
        batches = []
        for i in range(0, len(sequences), self.batch_size):
            batches.append(sequences[i:i+self.batch_size])
        return batches
    
    def embed_batch(self, batch_seqs):
        if 'protbert' in self.model_name.lower():
            batch_seqs = [' '.join(list(seq)) for seq in batch_seqs]
        
        inputs = self.tokenizer(batch_seqs, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().float().numpy()
    
    def generate_embeddings(self, fasta_path, output_path):
        print(f"Processing {self.model_name}...")
        ids, sequences = self.read_fasta(fasta_path)
        batches = self.batch_sequences(sequences)
        
        all_embeddings = []
        for batch in tqdm(batches, desc=f"Embedding {self.model_name}"):
            embs = self.embed_batch(batch)
            all_embeddings.append(embs)
            
            if len(all_embeddings) % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        embeddings_array = np.vstack(all_embeddings)
        
        np.save(f"{output_path}_embeddings.npy", embeddings_array)
        np.save(f"{output_path}_ids.npy", np.array(ids))
        
        print(f"Saved {embeddings_array.shape} embeddings to {output_path}")
        del all_embeddings, embeddings_array
        torch.cuda.empty_cache()
        gc.collect()

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    data_dir = "../data"
    emb_dir = "../embeddings"
    os.makedirs(emb_dir, exist_ok=True)
    
    train_fasta = f"{data_dir}/train_sequences.fasta"
    test_fasta = f"{data_dir}/testsuperset.fasta"
    
    models = [
        ('esm2_650m', 16),
        ('esm2_3b', 4),
        ('protbert', 12),
        ('ankh', 12)
    ]
    
    for model_name, batch_size in models:
        print(f"\n{'='*50}")
        print(f"Processing {model_name}")
        print(f"{'='*50}")
        
        embedder = ProteinEmbedder(model_name, device=device, batch_size=batch_size)
        
        embedder.generate_embeddings(train_fasta, f"{emb_dir}/train_{model_name}")
        embedder.generate_embeddings(test_fasta, f"{emb_dir}/test_{model_name}")
        
        del embedder
        torch.cuda.empty_cache()
        gc.collect()
    
    print("\nAll embeddings generated successfully!")

if __name__ == "__main__":
    main()