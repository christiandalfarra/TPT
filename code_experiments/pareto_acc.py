import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Opzionale, per stile migliore

def calculate_pareto_frontier(data_x, data_y):
    """
    Identifica i punti sulla frontiera di Pareto (Minimizzare X, Massimizzare Y).
    Assumiamo:
    X = Tempo (da minimizzare)
    Y = Accuratezza (da massimizzare)
    """
    # Ordina per tempo (X) crescente
    sorted_indices = np.argsort(data_x)
    sorted_x = data_x[sorted_indices]
    sorted_y = data_y[sorted_indices]
    
    frontier_x = []
    frontier_y = []
    frontier_indices = []
    
    current_max_y = -float('inf')
    
    for i, (cx, cy) in enumerate(zip(sorted_x, sorted_y)):
        # Se troviamo un punto che ha un'accuratezza maggiore del massimo visto finora
        # (tra i metodi più veloci), allora è un punto di Pareto.
        # Significa: "Per ottenere questa accuratezza, questo è il tempo minimo richiesto finora".
        if cy > current_max_y:
            frontier_x.append(cx)
            frontier_y.append(cy)
            frontier_indices.append(sorted_indices[i])
            current_max_y = cy
            
    return frontier_x, frontier_y, frontier_indices

def main():
    # 1. Caricamento Dati
    filename = 'final_summary_grid_search.csv'
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Errore: Il file {filename} non è stato trovato.")
        return

    # 2. Calcolo Frontiera
    # Assicuriamoci che i dati siano numpy array
    time_col = df['Time_Sec'].values
    acc_col = df['Accuracy'].values
    
    fx, fy, f_idx = calculate_pareto_frontier(time_col, acc_col)
    
    # 3. Plotting
    plt.figure(figsize=(12, 8))
    
    # Usiamo seaborn per uno scatter plot più bello con palette automatica
    # Coloriamo per 'Views' e usiamo lo stile del marker per 'Steps'
    sns.set_style("whitegrid")
    scatter = sns.scatterplot(
        data=df, 
        x='Time_Sec', 
        y='Accuracy', 
        hue='Views', 
        style='Steps', 
        palette='viridis', 
        s=100, 
        edgecolor='black',
        alpha=0.8
    )

    # Disegna la linea di Pareto
    plt.plot(fx, fy, color='red', linestyle='--', linewidth=2, label='Pareto Frontier', zorder=5)
    
    # Evidenzia i punti di Pareto con una croce rossa
    plt.scatter(fx, fy, color='red', marker='X', s=150, zorder=6)

    # 4. Annotazioni sui punti ottimali
    # Aggiungiamo un'etichetta per capire quale config è sulla frontiera
    print("--- Configurazioni sulla Frontiera di Pareto ---")
    for i, idx in enumerate(f_idx):
        row = df.iloc[idx]
        label = f"V:{int(row['Views'])} S:{int(row['Steps'])}"
        
        # Stampa a console per il report
        print(f"Config: {label} -> Acc: {row['Accuracy']:.2f}% | Time: {row['Time_Sec']:.1f}s")
        
        # Aggiungi testo al grafico
        plt.annotate(
            label, 
            (fx[i], fy[i]),
            xytext=(10, -10), 
            textcoords='offset points',
            fontsize=9,
            color='darkred',
            fontweight='bold',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='gray')
        )

    # 5. Formattazione Grafico
    plt.title('Analisi Efficienza: Frontiera di Pareto\n(Trade-off Tempo vs Accuratezza)', fontsize=16)
    plt.xlabel('Tempo Totale di Esecuzione (Secondi) [Minimizzare]', fontsize=12)
    plt.ylabel('Accuratezza (%) [Massimizzare]', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig('pareto_efficiency_analysis.png', dpi=300)
    print("\nGrafico salvato come 'pareto_efficiency_analysis.png'")
    plt.show()

if __name__ == "__main__":
    main()