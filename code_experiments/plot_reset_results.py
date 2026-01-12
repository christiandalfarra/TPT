import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('tpt_continuous_reset_lr1e_3results.csv')

# --- CONFIGURATION ---
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
# Palette: Green (Freq Reset), Blue (Mid), Red (Slow Reset)
palette = {10: "#2ecc71", 50: "#3498db", 100: "#e74c3c"}

# ==========================================
# PLOT 1: Cumulative Average Accuracy
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Casual Stream
sns.lineplot(
    data=df[df['Mode'] == 'Casual'],
    x='Samples_Seen', y='Running_Acc',
    hue='Reset_Interval', palette=palette,
    linewidth=2.5, ax=axes[0]
)
axes[0].set_title("Casual Stream (I.I.D.)")
axes[0].set_ylabel("Average Accuracy (%)")
axes[0].set_xlabel("Samples Seen")
axes[0].legend(title="Reset Interval", loc='lower right')
axes[0].set_ylim(0, 100)

# Sorted Stream
sns.lineplot(
    data=df[df['Mode'] == 'Sorted'],
    x='Samples_Seen', y='Running_Acc',
    hue='Reset_Interval', palette=palette,
    linewidth=2.5, ax=axes[1]
)
axes[1].set_title("Sorted Stream (Class-Incremental)")
axes[1].set_xlabel("Samples Seen")
axes[1].set_ylabel("")
axes[1].legend().remove()

plt.tight_layout()
plt.savefig('tpt_reset_avg_acc.png', dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# PLOT 2: Stability Analysis (Rolling Window)
# ==========================================
# Calculate rolling mean (window=50 samples)
df['Rolling_Acc'] = df.groupby(['Mode', 'Reset_Interval'])['Batch_Acc'].transform(lambda x: x.rolling(window=50, min_periods=1).mean())

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# Casual Stability
sns.lineplot(
    data=df[df['Mode'] == 'Casual'],
    x='Samples_Seen', y='Rolling_Acc',
    hue='Reset_Interval', palette=palette,
    linewidth=2, alpha=0.8, ax=axes2[0]
)
axes2[0].set_title("Stability: Casual Stream")
axes2[0].set_ylabel("Instant Accuracy (Rolling Avg 50)")
axes2[0].set_xlabel("Samples Seen")
axes2[0].legend(title="Reset Interval", loc='lower right')
axes2[0].set_ylim(0, 105)

# Sorted Stability
sns.lineplot(
    data=df[df['Mode'] == 'Sorted'],
    x='Samples_Seen', y='Rolling_Acc',
    hue='Reset_Interval', palette=palette,
    linewidth=2, alpha=0.8, ax=axes2[1]
)
axes2[1].set_title("Stability: Sorted Stream")
axes2[1].set_xlabel("Samples Seen")
axes2[1].set_ylabel("")
axes2[1].legend().remove()

plt.tight_layout()
plt.savefig('tpt_reset_stability.png', dpi=300, bbox_inches='tight')
plt.show()