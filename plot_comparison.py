import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_comparison():
    # Cerca tutti i file CSV generati (adatta il pattern al nome dei tuoi file output)
    # Esempio pattern: "continuous_analysis_*.csv"
    csv_files = glob.glob("continuous_analysis_*.csv")
    
    if not csv_files:
        print("Nessun CSV trovato per il plotting.")
        return

    plt.figure(figsize=(15, 10))
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Estrai info dal nome file per la legenda (molto grezzo, dipende dai tuoi nomi file)
            # Assumiamo nomi tipo: continuous_analysis_casual_lr0.005_reset10.csv
            label = file.replace("continuous_analysis_", "").replace(".csv", "")
            
            # Plot dell'accuratezza media (più stabile)
            if 'Avg_Acc' in df.columns:
                plt.plot(df['Step'], df['Avg_Acc'], label=label, linewidth=2)
        except Exception as e:
            print(f"Errore leggendo {file}: {e}")

    plt.title("Confronto Analisi Continua: Ordinato vs Casuale (Vari LR e Reset)")
    plt.xlabel("Step (Immagini processate)")
    plt.ylabel("Accuratezza Media Cumulativa (%)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("confronto_totale.png")
    print("Grafico salvato come 'confronto_totale.png'")

if __name__ == "__main__":
    plot_comparison()
