#!/usr/bin/env python3
"""Figure: Biological diversity of the benchmark dataset."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'benchmark_v2' / 'data'
FIG_DIR = BASE_DIR / 'benchmark_v2' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / 'benchmark_v2.csv')
prots = pd.read_csv(DATA_DIR / 'protein_sequences.csv')
gene_len = dict(zip(prots['gene'], prots['length']))

# ── Figure: 6-panel diversity overview ──
fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.30)

# ── Panel A: Protein length distribution ──
ax1 = fig.add_subplot(gs[0, 0])
protein_lengths = [gene_len[g] for g in df['GeneSymbol'].unique() if g in gene_len]
ax1.hist(protein_lengths, bins=25, color='#1565C0', edgecolor='white', alpha=0.85, zorder=3)
ax1.axvline(np.median(protein_lengths), color='#E53935', linestyle='--', linewidth=1.5,
            label=f'Median: {np.median(protein_lengths):.0f} aa')
ax1.set_xlabel('Protein Length (aa)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
ax1.set_title('A. Protein Length Distribution', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='upper right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ── Panel B: Variant position distribution (normalized) ──
ax2 = fig.add_subplot(gs[0, 1])
pos_frac = []
for _, row in df.iterrows():
    gene = row['GeneSymbol']
    if gene in gene_len:
        pos_frac.append(row['aa_position'] / gene_len[gene])
ax2.hist(pos_frac, bins=30, color='#E65100', edgecolor='white', alpha=0.85, zorder=3)
ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Relative Position (0=N-term, 1=C-term)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
ax2.set_title('B. Variant Position Along Protein', fontsize=12, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel C: Amino acid substitution frequency ──
ax3 = fig.add_subplot(gs[0, 2])
aa_order = list('ACDEFGHIKLMNPQRSTVWY')
aa_labels = list('ACDEFGHIKLMNPQRSTVWY')

# Build substitution matrix
sub_matrix = np.zeros((20, 20))
aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}
for _, row in df.iterrows():
    ref = row['ref_aa']
    alt = row['alt_aa']
    if ref in aa_to_idx and alt in aa_to_idx:
        sub_matrix[aa_to_idx[ref], aa_to_idx[alt]] += 1

# Normalize rows
row_sums = sub_matrix.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
sub_norm = sub_matrix / row_sums

im = ax3.imshow(sub_norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.3)
ax3.set_xticks(range(20))
ax3.set_xticklabels(aa_labels, fontsize=8, rotation=45, ha='right')
ax3.set_yticks(range(20))
ax3.set_yticklabels(aa_labels, fontsize=8)
ax3.set_xlabel('Alternate Amino Acid', fontsize=11, fontweight='bold')
ax3.set_ylabel('Reference Amino Acid', fontsize=11, fontweight='bold')
ax3.set_title('C. Amino Acid Substitution Pattern', fontsize=12, fontweight='bold')
cb = plt.colorbar(im, ax=ax3, shrink=0.7, pad=0.02)
cb.set_label('Frequency', fontsize=9)

# ── Panel D: Variants per gene ──
ax4 = fig.add_subplot(gs[1, 0])
variants_per_gene = df.groupby('GeneSymbol').size().sort_values(ascending=False)
ax4.hist(variants_per_gene.values, bins=range(1, variants_per_gene.max() + 2),
         color='#2E7D32', edgecolor='white', alpha=0.85, zorder=3, align='left')
ax4.set_xlabel('Number of Variants per Gene', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
ax4.set_title('D. Variant Density per Gene', fontsize=12, fontweight='bold')
ax4.axvline(variants_per_gene.median(), color='#E53935', linestyle='--', linewidth=1.5,
            label=f'Median: {variants_per_gene.median():.0f}')
ax4.legend(fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# ── Panel E: Chromosome distribution ──
ax5 = fig.add_subplot(gs[1, 1])
chr_counts = df['Chromosome'].value_counts().sort_index()
# Sort numerically
def chr_sort_key(c):
    try:
        return (0, int(c))
    except ValueError:
        return (1, c)

chr_labels = sorted([str(c) for c in chr_counts.index], key=chr_sort_key)
chr_values = [chr_counts[c] for c in chr_labels]
colors = ['#1565C0' if i % 2 == 0 else '#42A5F5' for i in range(len(chr_labels))]
ax5.bar(range(len(chr_labels)), chr_values, color=colors, edgecolor='white', alpha=0.85, zorder=3)
ax5.set_xticks(range(len(chr_labels)))
ax5.set_xticklabels(chr_labels, fontsize=7, rotation=45, ha='right')
ax5.set_xlabel('Chromosome', fontsize=11, fontweight='bold')
ax5.set_ylabel('Number of Variants', fontsize=11, fontweight='bold')
ax5.set_title('E. Chromosomal Distribution', fontsize=12, fontweight='bold')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# ── Panel F: Pathogenicity ratio per gene ──
ax6 = fig.add_subplot(gs[1, 2])
gene_labels = df.groupby('GeneSymbol')['ClinVar_label'].apply(
    lambda x: (x == 'Pathogenic').sum() / len(x) if len(x) > 0 else 0.5
).sort_values()
# Show distribution of pathogenic fraction
ax6.hist(gene_labels.values, bins=20, color='#7B1FA2', edgecolor='white', alpha=0.85, zorder=3)
ax6.axvline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax6.set_xlabel('Fraction Pathogenic Variants per Gene', fontsize=11, fontweight='bold')
ax6.set_ylabel('Number of Genes', fontsize=11, fontweight='bold')
ax6.set_title('F. Class Balance per Gene', fontsize=12, fontweight='bold')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.text(0.52, ax6.get_ylim()[1] * 0.9, '50/50 line', fontsize=8, color='gray')

# ── Summary stats ──
n_genes = df['GeneSymbol'].nunique()
n_variants = len(df)
n_path = (df['ClinVar_label'] == 'Pathogenic').sum()
n_ben = (df['ClinVar_label'] == 'Benign').sum()
median_pos_frac = np.median(pos_frac)

fig.suptitle(f'Benchmark V2: Biological Diversity Overview '
             f'({n_variants} variants, {n_genes} genes, '
             f'{n_path} pathogenic / {n_ben} benign)',
             fontsize=14, fontweight='bold', y=1.01)

fig.savefig(FIG_DIR / 'fig9_data_diversity.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig9_data_diversity.png'}")
print(f"\nSummary:")
print(f"  Genes: {n_genes}")
print(f"  Variants: {n_variants} ({n_path} pathogenic, {n_ben} benign)")
print(f"  Protein lengths: {min(protein_lengths)}-{max(protein_lengths)} aa (median {np.median(protein_lengths):.0f})")
print(f"  Variants per gene: {variants_per_gene.min()}-{variants_per_gene.max()} (median {variants_per_gene.median():.0f})")
print(f"  Variant positions: median {median_pos_frac:.2f} along protein")
print(f"  Unique substitutions: {len(set(zip(df['ref_aa'], df['alt_aa'])))}")
