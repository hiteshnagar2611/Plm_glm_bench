#!/usr/bin/env python3
"""Figure 0: Benchmark V2 dataset overview."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

df = pd.read_csv('benchmark_v2/data/benchmark_v2.csv')
prots = pd.read_csv('benchmark_v2/data/protein_sequences.csv')

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

# ── Panel A: Class balance (donut) ──
ax1 = fig.add_subplot(gs[0, 0])
counts = df['ClinVar_label'].value_counts()
colors_pie = ['#E53935', '#1E88E5']
wedges, texts, autotexts = ax1.pie(
    counts.values, labels=counts.index, autopct='%1.1f%%',
    colors=colors_pie, startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
for t in autotexts:
    t.set_fontsize(11)
    t.set_fontweight('bold')
ax1.set_title('A. Class Distribution', fontsize=12, fontweight='bold', pad=10)
ax1.text(0, 0, f'{len(df)}\nvariants', ha='center', va='center', fontsize=13, fontweight='bold')

# ── Panel B: Variants per gene ──
ax2 = fig.add_subplot(gs[0, 1])
gene_counts = df.groupby('GeneSymbol').size().sort_values(ascending=False)
ax2.hist(gene_counts.values, bins=30, color='#7B1FA2', edgecolor='white', linewidth=0.5, alpha=0.85)
ax2.axvline(gene_counts.median(), color='#E53935', linestyle='--', linewidth=1.5, label=f'Median = {gene_counts.median():.0f}')
ax2.axvline(gene_counts.mean(), color='#1E88E5', linestyle='--', linewidth=1.5, label=f'Mean = {gene_counts.mean():.1f}')
ax2.set_xlabel('Variants per Gene', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
ax2.set_title('B. Variant Density per Gene', fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel C: Chromosome distribution ──
ax3 = fig.add_subplot(gs[0, 2])
chrom_counts = df.groupby('Chromosome').size().sort_index()
chrom_labels = [str(c) for c in chrom_counts.index]
bar_colors = ['#E53935' if df[df['Chromosome']==c]['ClinVar_label'].eq('Pathogenic').sum() >
              df[df['Chromosome']==c]['ClinVar_label'].eq('Benign').sum() else '#1E88E5'
              for c in chrom_counts.index]
ax3.bar(range(len(chrom_counts)), chrom_counts.values, color=bar_colors, edgecolor='white', linewidth=0.5)
ax3.set_xticks(range(len(chrom_counts)))
ax3.set_xticklabels(chrom_labels, fontsize=7, rotation=0)
ax3.set_xlabel('Chromosome', fontsize=11, fontweight='bold')
ax3.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
ax3.set_title('C. Genomic Distribution', fontsize=12, fontweight='bold', pad=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# ── Panel D: Amino acid position distribution ──
ax4 = fig.add_subplot(gs[1, 0])
path_pos = df[df['ClinVar_label']=='Pathogenic']['aa_position'].dropna()
ben_pos = df[df['ClinVar_label']=='Benign']['aa_position'].dropna()
ax4.hist(path_pos, bins=40, alpha=0.6, color='#E53935', label='Pathogenic', density=True, edgecolor='white', linewidth=0.3)
ax4.hist(ben_pos, bins=40, alpha=0.6, color='#1E88E5', label='Benign', density=True, edgecolor='white', linewidth=0.3)
ax4.set_xlabel('Amino Acid Position', fontsize=11, fontweight='bold')
ax4.set_ylabel('Density', fontsize=11, fontweight='bold')
ax4.set_title('D. Position Along Protein', fontsize=12, fontweight='bold', pad=10)
ax4.legend(fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel E: ClinVar star rating ──
ax5 = fig.add_subplot(gs[1, 1])
star_data = df.groupby(['gold_stars', 'ClinVar_label']).size().unstack(fill_value=0)
star_labels = [f'{int(s)} star' + ('s' if s > 1 else '') for s in star_data.index]
x_s = np.arange(len(star_data))
width_s = 0.35
path_vals = star_data.get('Pathogenic', pd.Series(0, index=star_data.index)).values
ben_vals = star_data.get('Benign', pd.Series(0, index=star_data.index)).values
ax5.bar(x_s - width_s/2, path_vals, width_s, color='#E53935', label='Pathogenic', edgecolor='white')
ax5.bar(x_s + width_s/2, ben_vals, width_s, color='#1E88E5', label='Benign', edgecolor='white')
for p, b, x in zip(path_vals, ben_vals, x_s):
    ax5.text(x - width_s/2, p + 3, str(p), ha='center', fontsize=8, fontweight='bold', color='#E53935')
    ax5.text(x + width_s/2, b + 3, str(b), ha='center', fontsize=8, fontweight='bold', color='#1E88E5')
ax5.set_xticks(x_s)
ax5.set_xticklabels(star_labels, fontsize=9)
ax5.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
ax5.set_title('E. ClinVar Review Stars', fontsize=12, fontweight='bold', pad=10)
ax5.legend(fontsize=9)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel F: Top 15 genes ──
ax6 = fig.add_subplot(gs[1, 2])
top_n = 15
top_genes = gene_counts.head(top_n)
top_path = df[df['ClinVar_label']=='Pathogenic'].groupby('GeneSymbol').size().reindex(top_genes.index, fill_value=0)
top_ben = df[df['ClinVar_label']=='Benign'].groupby('GeneSymbol').size().reindex(top_genes.index, fill_value=0)
y_pos = range(len(top_genes))
ax6.barh(y_pos, top_path.values, color='#E53935', label='Pathogenic', edgecolor='white', linewidth=0.5)
ax6.barh(y_pos, top_ben.values, left=top_path.values, color='#1E88E5', label='Benign', edgecolor='white', linewidth=0.5)
ax6.set_yticks(y_pos)
ax6.set_yticklabels(top_genes.index, fontsize=8)
ax6.set_xlabel('Number of Variants', fontsize=11, fontweight='bold')
ax6.set_title(f'F. Top {top_n} Genes by Variant Count', fontsize=12, fontweight='bold', pad=10)
ax6.legend(fontsize=9, loc='lower right')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.invert_yaxis()

fig.suptitle('Benchmark V2: Dataset Overview (207 Genes, 793 ClinVar Missense Variants)',
             fontsize=15, fontweight='bold', y=0.98)

fig.savefig('benchmark_v2/figures/fig0_dataset_overview.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 0 saved: benchmark_v2/figures/fig0_dataset_overview.png")
