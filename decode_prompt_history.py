import torch
import clip
import numpy as np
import os
import argparse
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Decode Soft Prompts to Words')
    parser.add_argument('--prompt_dir', type=str, required=True, help='Cartella contenente i file .npy')
    parser.add_argument('--arch', default='ViT-B/16', type=str, help='Architettura CLIP usata')
    parser.add_argument('--topk', default=1, type=int, help='Quante parole mostrare per ogni token')
    return parser.parse_args()

def main():
    args = get_arguments()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {args.arch} on {device}...")
    model, _ = clip.load(args.arch, device=device)
    
    # 1. Estraiamo la matrice di embedding di tutte le parole conosciute da CLIP
    # token_embedding weight shape: [vocab_size, transformer_width]
    token_embeddings = model.token_embedding.weight.detach()
    
    # Normalizziamo per usare la Cosine Similarity
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)
    
    # Vocabolario inverso (ID -> Parola)
    # CLIP usa un BPE tokenizer. Dobbiamo accedere al decoder.
    import gzip
    import html
    import ftfy
    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    id_to_word = {v: k for k, v in tokenizer.encoder.items()}

    # Troviamo i file .npy
    files = sorted([f for f in os.listdir(args.prompt_dir) if f.endswith('.npy')])
    
    print(f"Trovati {len(files)} snapshot di prompt. Inizio decodifica...\n")
    print("-" * 60)

    # 2. Iteriamo su ogni step salvato
    # Ne leggiamo magari uno ogni 10 o 50 per non intasare l'output, oppure tutti se sono pochi
    step_skip = 1 
    
    for i, fname in enumerate(files):
        if i % step_skip != 0: continue
        
        path = os.path.join(args.prompt_dir, fname)
        
        # Carica il soft prompt: Shape [n_ctx, dim] (es. [4, 512])
        ctx_vector = np.load(path) 
        ctx_tensor = torch.from_numpy(ctx_vector).to(device)
        
        # Normalizziamo il vettore del prompt
        ctx_tensor = ctx_tensor / ctx_tensor.norm(dim=-1, keepdim=True)
        
        decoded_sentence = []
        
        # 3. Per ogni token del prompt (es. 4 token), troviamo la parola più vicina
        for token_idx in range(ctx_tensor.shape[0]):
            single_token_vec = ctx_tensor[token_idx].unsqueeze(0) # [1, dim]
            
            # Calcolo similarità con TUTTE le parole del vocabolario
            # (Matrice moltiplicazione è equivalente a cosine similarity su vettori normalizzati)
            similarity = (single_token_vec @ token_embeddings.T).squeeze(0) # [vocab_size]
            
            # Prendi i top-K indici
            values, indices = similarity.topk(args.topk)
            
            top_words = []
            for idx in indices:
                token_id = idx.item()
                # Decodifica il token ID in stringa
                word = tokenizer.decode([token_id]).strip()
                top_words.append(word)
            
            # Prendi la migliore per la frase principale
            decoded_sentence.append(top_words[0])
        
        # Stampa risultato
        full_sentence = " ".join(decoded_sentence)
        print(f"[{fname}] -> '{full_sentence}'")

    print("-" * 60)
    print("Nota: I caratteri strani o parole non correlate sono normali.")
    print("TPT sposta i vettori in spazi semantici che non corrispondono perfettamente a parole umane.")

if __name__ == "__main__":
    main()