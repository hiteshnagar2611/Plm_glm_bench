#!/usr/bin/env python3
"""Heatmap of wild-type vs mutant amino acid pathogenicity rates."""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(BASE, "benchmark_v3")
OUT = os.path.join(V3, "figures", "fig_aa_heatmap.png")

df = pd.read_csv(os.path.join(V3, "data", "benchmark_v3.csv"))
df['label'] = df['ClinVar_label'].map({1.0:1, 0.0:0})

aa_list = ['Cys','Ser','Thr','Pro','Ala','Gly','Asn','Asp','Glu','Gln',
           'His','Arg','Lys','Ile','Leu','Met','Phe','Tyr','Trp','Val']
aa_1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Gln':'Q','Glu':'E',
        'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
        'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V'}

# Build pathogenicity rate matrix
matrix = pd.DataFrame(np.nan, index=aa_list, columns=aa_list)
counts = pd.DataFrame(0, index=aa_list, columns=aa_list)

for r in aa_list:
    for a in aa_list:
        if r != a:
            mask = (df['ref_aa'] == r) & (df['alt_aa'] == a)
            sub = df[mask]
            if len(sub) >= 3:
                matrix.loc[r, a] = sub['label'].mean()
                counts.loc[r, a] = len(sub)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.3})

# ── Left: Pathogenicity rate heatmap ──
ax = axes[0]
cmap = LinearSegmentedColormap.from_list('risk', ['#22c55e','#fbbf24','#ef4444'])
im = ax.imshow(matrix.values, cmap=cmap, vmin=0, vmax=1, aspect='equal')

# Labels
labels_1 = [aa_1[a] for a in aa_list]
ax.set_xticks(range(len(aa_list)))
ax.set_xticklabels(labels_1, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(aa_list)))
ax.set_yticklabels(labels_1, fontsize=11, fontweight='bold')
ax.set_xlabel("Mutant amino acid", fontsize=13, fontweight='bold')
ax.set_ylabel("Wild-type amino acid", fontsize=13, fontweight='bold')
ax.set_title("Pathogenicity rate", fontsize=15, fontweight='bold', pad=12)

# Annotate cells
for i in range(len(aa_list)):
    for j in range(len(aa_list)):
        val = matrix.iloc[i, j]
        cnt = counts.iloc[i, j]
        if cnt >= 3 and not np.isnan(val):
            text_color = 'white' if val > 0.7 or val < 0.3 else 'black'
            ax.text(j, i, f"{val:.0%}\n(n={cnt})", ha='center', va='center',
                    fontsize=7, color=text_color)
        elif i == j:
            ax.text(j, i, "—", ha='center', va='center', fontsize=9, color='gray')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pathogenicity rate", fontsize=11)
cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels(['0%\n(Benign)', '25%', '50%', '75%', '100%\n(Pathogenic)'])

# ── Right: Variant count heatmap ──
ax2 = axes[1]
cmap2 = LinearSegmentedColormap.from_list('count', ['#f0f9ff','#0ea5e9','#1e3a5f'])
im2 = ax2.imshow(counts.values, cmap=cmap2, aspect='equal')

ax2.set_xticks(range(len(aa_list)))
ax2.set_xticklabels(labels_1, fontsize=11, fontweight='bold')
ax2.set_yticks(range(len(aa_list)))
ax2.set_yticklabels(labels_1, fontsize=11, fontweight='bold')
ax2.set_xlabel("Mutant amino acid", fontsize=13, fontweight='bold')
ax2.set_ylabel("Wild-type amino acid", fontsize=13, fontweight='bold')
ax2.set_title("Variant count", fontsize=15, fontweight='bold', pad=12)

for i in range(len(aa_list)):
    for j in range(len(aa_list)):
        cnt = counts.iloc[i, j]
        if cnt >= 3:
            text_color = 'white' if cnt > 100 else 'black'
            ax2.text(j, i, str(cnt), ha='center', va='center', fontsize=7, color=text_color)
        elif i == j:
            ax2.text(j, i, "—", ha='center', va='center', fontsize=9, color='gray')

cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label("Variant count", fontsize=11)

fig.suptitle("Amino Acid Substitution Pathogenicity Heatmap  (ClinVar V3, n=5,932)",
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(OUT, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}")
