import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Caricamento e Pulizia Dati
# -----------------------------------
filename = 'tpt_continuous_results1.csv'
df = pd.read_csv(filename)

# Convertiamo colonne critiche in numerico, gestendo errori (es. headers ripetuti)
cols_to_numeric = ['LR', 'Batch_Acc', 'Running_Acc', 'Samples_Seen']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Rimuoviamo righe corrotte (dove la conversione ha fallito)
df.dropna(subset=cols_to_numeric, inplace=True)

# 2. Calcolo Statistiche
# -----------------------------------
# Calcolo media mobile per visualizzare la stabilità
df['Moving_Avg_Acc'] = df.groupby(['LR', 'Mode'])['Batch_Acc'].transform(
    lambda x: x.rolling(window=100, min_periods=10).mean()
)

# Impostazioni grafiche
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams['font.family'] = 'serif'

# Ordiniamo i Learning Rate dal più alto (instabile) al più basso (stabile)
lrs = sorted(df['LR'].unique(), reverse=True)
n_lrs = len(lrs)

# 3. Plot 1: Cumulative Running Accuracy
# -----------------------------------
fig1, axes1 = plt.subplots(1, n_lrs, figsize=(6 * n_lrs, 5), sharey=True)
if n_lrs == 1: axes1 = [axes1]

for i, lr in enumerate(lrs):
    ax = axes1[i]
    subset = df[df['LR'] == lr]
    
    sns.lineplot(data=subset, x='Samples_Seen', y='Running_Acc', hue='Mode', 
                 palette={'Casual': 'blue', 'Sorted': 'red'}, ax=ax, linewidth=2.5)
    
    # Formattazione LR in notazione scientifica se molto piccolo
    title_lr = f"{lr:.0e}" if lr < 0.001 else f"{lr}"
    ax.set_title(f"Learning Rate: {title_lr}")
    
    ax.set_xlabel("Samples Seen")
    if i == 0:
        ax.set_ylabel("Cumulative Accuracy (%)")
        ax.legend(title='Data Stream')
    else:
        ax.set_ylabel("")
        if ax.get_legend(): ax.get_legend().remove()
    
    ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('tpt_paper_cumulative_1.png', dpi=300, bbox_inches='tight')

# 4. Plot 2: Instantaneous Stability
# -----------------------------------
fig2, axes2 = plt.subplots(1, n_lrs, figsize=(6 * n_lrs, 5), sharey=True)
if n_lrs == 1: axes2 = [axes2]

for i, lr in enumerate(lrs):
    ax = axes2[i]
    subset = df[df['LR'] == lr]
    
    sns.lineplot(data=subset, x='Samples_Seen', y='Moving_Avg_Acc', hue='Mode', 
                 palette={'Casual': 'blue', 'Sorted': 'red'}, ax=ax, linewidth=1.5, alpha=0.9)
    
    title_lr = f"{lr:.0e}" if lr < 0.001 else f"{lr}"
    ax.set_title(f"Stability (LR: {title_lr})")
    
    ax.set_xlabel("Samples Seen")
    if i == 0:
        ax.set_ylabel("Instantaneous Accuracy\n(Rolling Avg 100 samples)")
    else:
        ax.set_ylabel("")
        
    ax.set_ylim(0, 100)
    
    # Linee verticali per i cambi di dominio (ogni 500 sample, su 2000 totali)
    for domain_boundary in [250, 500, 750, 1000]:
        ax.axvline(x=domain_boundary, color='gray', linestyle='--', alpha=0.3)
        
    if i > 0 and ax.get_legend(): ax.get_legend().remove()

plt.tight_layout()
plt.savefig('tpt_paper_stability_1.png', dpi=300, bbox_inches='tight')

# 5. Tabella Riassuntiva
# -----------------------------------
summary = df.groupby(['LR', 'Mode'])['Running_Acc'].last().unstack()
summary['Delta (Casual - Sorted)'] = summary['Casual'] - summary['Sorted']
print("\n=== FINAL RESULTS TABLE ===")
print(summary)