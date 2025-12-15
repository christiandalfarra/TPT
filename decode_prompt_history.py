import torch
import clip
import numpy as np
import os
import argparse
from tqdm import tqdm

def get_arguments():
    parser = argparse.ArgumentParser(description='Decode Soft Prompts to Words and Save to File')
    parser.add_argument('--prompt_dir', type=str, required=True, help='Cartella contenente i file .npy')
    parser.add_argument('--arch', default='RN50', type=str, help='Architettura CLIP usata')
    parser.add_argument('--topk', default=1, type=int, help='Quante parole mostrare per ogni token')
    parser.add_argument('--output', type=str, default='decoded_prompts.txt', help='Nome del file di output')
    return parser.parse_args()

def main():
    args = get_arguments()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP {args.arch} on {device}...")
    model, _ , _ = clip.load(args.arch, device=device)
    
    # 1. Estraiamo la matrice di embedding di tutte le parole conosciute da CLIP
    token_embeddings = model.token_embedding.weight.detach()
    
    # Normalizziamo per usare la Cosine Similarity
    token_embeddings = token_embeddings / token_embeddings.norm(dim=-1, keepdim=True)
    
    # Vocabolario inverso (ID -> Parola)
    from clip.simple_tokenizer import SimpleTokenizer
    tokenizer = SimpleTokenizer()
    
    # Troviamo i file .npy
    if not os.path.exists(args.prompt_dir):
        print(f"Errore: La directory {args.prompt_dir} non esiste.")
        return

    files = sorted([f for f in os.listdir(args.prompt_dir) if f.endswith('.npy')])
    
    print(f"Trovati {len(files)} snapshot. Decodifica in corso...")
    print(f"Output verrà salvato in: {args.output}")

    # Apriamo il file in scrittura
    with open(args.output, 'w', encoding='utf-8') as f_out:
        # Usiamo tqdm per mostrare una barra di progresso invece di stampare le parole
        for fname in tqdm(files, desc="Processing"):
            
            path = os.path.join(args.prompt_dir, fname)
            
            try:
                # Carica il soft prompt
                ctx_vector = np.load(path) 
                ctx_tensor = torch.from_numpy(ctx_vector).to(device)
                
                # Normalizziamo il vettore del prompt
                ctx_tensor = ctx_tensor / ctx_tensor.norm(dim=-1, keepdim=True)
                
                decoded_sentence = []
                
                # Per ogni token del prompt, troviamo la parola più vicina
                for token_idx in range(ctx_tensor.shape[0]):
                    single_token_vec = ctx_tensor[token_idx].unsqueeze(0) # [1, dim]
                    
                    # Calcolo similarità
                    similarity = (single_token_vec @ token_embeddings.T).squeeze(0)
                    
                    # Prendi i top-K indici
                    values, indices = similarity.topk(args.topk)
                    
                    top_words = []
                    for idx in indices:
                        token_id = idx.item()
                        word = tokenizer.decode([token_id]).strip()
                        top_words.append(word)
                    
                    # Prendi la migliore
                    decoded_sentence.append(top_words[0])
                
                # Componi la frase
                full_sentence = " ".join(decoded_sentence)
                
                # SCRIVIAMO SU FILE INVECE CHE A TERMINALE
                f_out.write(f"[{fname}] -> '{full_sentence}'\n")
            
            except Exception as e:
                print(f"Errore processando {fname}: {e}")

    print("-" * 60)
    print("Decodifica completata.")
    print(f"Puoi leggere i risultati nel file: {args.output}")

if __name__ == "__main__":
    main()
