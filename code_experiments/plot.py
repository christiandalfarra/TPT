import pandas as pd
import matplotlib.pyplot as plt

# 1. Carica i dati dal CSV
# Assicurati che il file sia nella stessa cartella dello script
file_path = 'inr_continuous_analysis_sorted_lr0.005_reset10.csv'
df = pd.read_csv(file_path)

# 2. Imposta la dimensione del grafico
plt.figure(figsize=(12, 6))

# 3. Disegna l'Accuratezza Istantanea
# Usiamo alpha=0.3 per renderla trasparente, così non copre la media
plt.plot(df['Step'], df['Instant_Acc'], 
         label='Instant Accuracy', 
         color='orange', 
         alpha=0.3, 
         linewidth=1)

# 4. Disegna l'Accuratezza Media
# Linea più spessa e scura per evidenziare il trend
plt.plot(df['Step'], df['Avg_Acc'], 
         label='Average Accuracy', 
         color='blue', 
         linewidth=2)

# 5. Aggiungi etichette, titolo e griglia
plt.title('Andamento dell\'Accuratezza (LR=0.00001, Reset=0)')
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Mostra o salva il grafico
plt.savefig('grafico_accuratezza.png') # Salva come immagine
plt.show()                             # Mostra a schermo