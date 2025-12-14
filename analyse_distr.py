import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAZIONE ---
# Inserisci il percorso corretto alla cartella ImageNet-A estratta
# La struttura attesa è: root/n012345/immagine.jpg
DATA_PATH = "../datasets/imagenet-rendition/imagenet-r" 

def analyze_distribution():
    if not os.path.exists(DATA_PATH):
        print(f"Errore: Percorso {DATA_PATH} non trovato.")
        return

    class_counts = []
    class_names = []

    print("Conteggio immagini in corso...")
    
    # Itera su ogni cartella di classe
    for folder_name in sorted(os.listdir(DATA_PATH)):
        folder_full_path = os.path.join(DATA_PATH, folder_name)
        
        # Controlla se è una directory
        if os.path.isdir(folder_full_path):
            # Conta solo i file immagine
            images = [f for f in os.listdir(folder_full_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            count = len(images)
            if count > 0:
                class_counts.append(count)
                class_names.append(folder_name)

    # --- STATISTICHE ---
    counts_arr = np.array(class_counts)
    
    print("\n=== Statistiche ImageNet-A ===")
    print(f"Numero totale di Classi trovate: {len(counts_arr)}")
    print(f"Totale Immagini: {counts_arr.sum()}")
    print(f"Media per classe: {counts_arr.mean():.2f}")
    print(f"Minimo per classe: {counts_arr.min()}")
    print(f"Massimo per classe: {counts_arr.max()}")
    print(f"Deviazione Standard: {counts_arr.std():.2f}")
    
    # Controllo soglia 10
    under_10 = np.sum(counts_arr < 10)
    if under_10 == 0:
        print("\n✅ Conferma: Tutte le classi hanno almeno 10 immagini.")
    else:
        print(f"\n⚠️ Attenzione: {under_10} classi hanno meno di 10 immagini.")

    # --- GRAFICO ---
    plt.figure(figsize=(10, 6))
    plt.hist(counts_arr, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(counts_arr.mean(), color='red', linestyle='dashed', linewidth=1, label=f'Media ({counts_arr.mean():.1f})')
    plt.title('Distribuzione Immagini per Classe (ImageNet-A)')
    plt.xlabel('Numero di Immagini')
    plt.ylabel('Numero di Classi')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    output_file = "distribuzione_imagenet_a.png"
    plt.savefig(output_file)
    print(f"\nGrafico salvato come '{output_file}'")

if __name__ == "__main__":
    analyze_distribution()
