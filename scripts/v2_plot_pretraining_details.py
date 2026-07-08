#!/usr/bin/env python3
"""Figure 12: Detailed pretraining data provenance and types."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

FIG_DIR = Path(__file__).parent.parent / 'benchmark_v2' / 'figures'

fig = plt.figure(figsize=(28, 22))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.4, 0.8, 1.0], hspace=0.30, wspace=0.25,
                       left=0.04, right=0.96, top=0.94, bottom=0.03)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A: Data source hierarchy (wide, top)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.set_xlim(0, 28)
ax1.set_ylim(0, 8)

# ── Title ──
ax1.text(14, 7.7, 'Pretraining Data Provenance: Source Databases → Derived Datasets → Models',
         fontsize=15, fontweight='bold', ha='center', va='center')

# ── Layer 1: Primary Databases (top) ──
primary_dbs = [
    (3.5, 6.2, 'UniProt\n(EBI/NCBI)', '#1565C0', '250M+ sequences\nAll organisms'),
    (8.5, 6.2, 'BFD\n(Max Planck)', '#42A5F5', '2.5B sequences\nMetagenomic'),
    (14.0, 6.2, 'AlphaFold DB\n(EMBL-EBI)', '#7B1FA2', '200M+ structures\nAF2 predictions'),
    (19.5, 6.2, '1000 Genomes\n(NCBI)', '#E65100', '3,202 human genomes\nPopulation diversity'),
    (24.5, 6.2, 'ENCODE\n(NHGRI)', '#F57C00', 'Regulatory elements\nEpigenomic tracks'),
]
for x, y, name, color, desc in primary_dbs:
    rect = mpatches.FancyBboxPatch((x-1.6, y-0.7), 3.2, 1.4, boxstyle='round,pad=0.15',
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.9)
    ax1.add_patch(rect)
    ax1.text(x, y+0.2, name, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax1.text(x, y-0.35, desc, ha='center', va='center', fontsize=8, color='white', style='italic')

# ── Layer 2: Derived Datasets (middle) ──
derived = [
    (4.5, 3.8, 'UniRef50', '#90CAF9', '30-65M clusters\nSequence similarity'),
    (10.5, 3.8, 'UniRef50 + BFD', '#90CAF9', '45M seqs +\n393B amino acids'),
    (16.0, 3.8, 'AF2 Structures\n(pLDDT > 70)', '#CE93D8', '40M structures\nFoldseek 3Di'),
    (21.5, 3.8, 'Multi-species\nGenomes', '#FFCC80', '850 species\n174B nucleotides'),
    (26.0, 3.8, 'GRCh38\n+ ENCODE', '#FFE082', 'Human genome\nRegulatory elements'),
]
for x, y, name, color, desc in derived:
    rect = mpatches.FancyBboxPatch((x-1.8, y-0.6), 3.6, 1.2, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#666', linewidth=1.2)
    ax1.add_patch(rect)
    ax1.text(x, y+0.15, name, ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(x, y-0.30, desc, ha='center', va='center', fontsize=8, color='#333')

# ── Arrows: primary → derived ──
arrows_pd = [
    (3.5, 5.5, 4.5, 4.4),   # UniProt → UniRef50
    (8.5, 5.5, 10.5, 4.4),  # BFD → UniRef50+BFD
    (3.5, 5.5, 10.5, 4.4),  # UniProt also → UniRef50+BFD
    (14.0, 5.5, 16.0, 4.4), # AFDB → AF2 Structures
    (19.5, 5.5, 21.5, 4.4), # 1000G → Multi-species
    (24.5, 5.5, 21.5, 4.4), # ENCODE → Multi-species (regulatory)
    (24.5, 5.5, 26.0, 4.4), # ENCODE → GRCh38+ENCODE
]
for x1, y1, x2, y2 in arrows_pd:
    ax1.annotate('', xy=(x2, y2+0.1), xytext=(x1, y1-0.1),
                 arrowprops=dict(arrowstyle='->', color='#888', lw=1.8,
                                 connectionstyle='arc3,rad=0.12'))

# ── Layer 3: Models (bottom) ──
model_boxes = [
    (2.0, 1.4, 'ESM1b\n30M seqs', '#1E88E5'),
    (5.5, 1.4, 'ESM2\n65M seqs', '#1565C0'),
    (9.0, 1.4, 'ProtT5-XL\n393B AA', '#42A5F5'),
    (12.5, 1.4, 'SaProt\n40M structs', '#7B1FA2'),
    (16.0, 1.4, 'ESM3-sm\n1.4B params', '#0D47A1'),
    (19.5, 1.4, 'AlphaGenome\nENCODE', '#FF8F00'),
    (23.0, 1.4, 'NT-v2\n300B tokens', '#F57C00'),
    (26.5, 1.4, 'HyenaDNA\n3.2B nt', '#FFB300'),
]
for x, y, name, color in model_boxes:
    rect = mpatches.FancyBboxPatch((x-1.4, y-0.5), 2.8, 1.0, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x, y, name, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Arrows: derived → models
arrows_dm = [
    (4.5, 3.2, 2.0, 1.9),   # UniRef50 → ESM1b
    (4.5, 3.2, 5.5, 1.9),   # UniRef50 → ESM2
    (10.5, 3.2, 9.0, 1.9),  # UniRef50+BFD → ProtT5
    (16.0, 3.2, 12.5, 1.9), # AF2 → SaProt
    (16.0, 3.2, 16.0, 1.9), # AF2 → ESM3
    (17.5, 3.2, 19.5, 1.9), # Multi-species → NT-v2
    (17.5, 3.2, 23.0, 1.9), # Multi-species → HyenaDNA
    (20.5, 3.2, 19.5, 1.9), # GRCh38+ENCODE → AlphaGenome
]
for x1, y1, x2, y2 in arrows_dm:
    ax1.annotate('', xy=(x2, y2+0.1), xytext=(x1, y1-0.1),
                 arrowprops=dict(arrowstyle='->', color='#888', lw=1.8,
                                 connectionstyle='arc3,rad=0.08'))

# ══════════════════════════════════════════════════════════════════════════════
# Panel B: Data type breakdown
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0])

data_types = {
    'Protein sequences\n(UniRef/BFD)': 65,
    '3D structures\n(AlphaFold2)': 40,
    'Human genomes\n(1000 Genomes)': 3202,
    'Other species\ngenomes': 850,
    'Regulatory tracks\n(ENCODE)': 16000,
    'AlphaGenome\npretraining': 3000,
}
names = list(data_types.keys())
vals = list(data_types.values())
colors = ['#1565C0', '#7B1FA2', '#E65100', '#F57C00', '#FFB300', '#FF8F00']
units = ['M sequences', 'M structures', 'genomes', 'genomes', 'tracks', 'M variants']

ax2.barh(range(len(names)), vals, color=colors, edgecolor='white', linewidth=0.5, zorder=3, height=0.6)
ax2.set_yticks(range(len(names)))
ax2.set_yticklabels(names, fontsize=10)
ax2.set_xscale('log')
ax2.set_xlabel('Count (log scale)', fontsize=11, fontweight='bold')
ax2.set_title('B. Data Volume by Type', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, (v, u) in enumerate(zip(vals, units)):
    ax2.text(v * 1.3, i, f'{v:,} {u}', va='center', fontsize=9, fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# Panel C: Data processing pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.set_xlim(0, 14)
ax3.set_ylim(0, 6)

processing = [
    (7.0, 5.2, 'Raw Data', '#E3F2FD', 13),
    (7.0, 4.2, 'Clustering / Deduplication', '#BBDEFB', 11),
    (7.0, 3.2, 'Length Filtering (<1022 aa / <12kb)', '#90CAF9', 10),
    (7.0, 2.2, 'Quality Filtering (pLDDT>70 for AF2)', '#64B5F6', 10),
    (7.0, 1.2, 'Tokenization (AA / 6-mer / 3Di)', '#42A5F5', 10),
    (7.0, 0.2, 'Masked LM Pretraining', '#1565C0', 12),
]
for x, y, text, color, fs in processing:
    rect = mpatches.FancyBboxPatch((x-4.5, y-0.35), 9.0, 0.7, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1)
    ax3.add_patch(rect)
    ax3.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight='bold')

for i in range(len(processing)-1):
    ax3.annotate('', xy=(7.0, processing[i+1][1]+0.4), xytext=(7.0, processing[i][1]-0.4),
                 arrowprops=dict(arrowstyle='->', color='#666', lw=1.5))

ax3.set_title('C. Data Processing Pipeline', fontsize=13, fontweight='bold', pad=10)

# ══════════════════════════════════════════════════════════════════════════════
# Panel D: Detailed model-data mapping table
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

col_labels = ['Model', 'Primary Source', 'Database', 'License', 'Filtered To', 'Token Type', 'Pretraining Task']
table_data = [
    ['ESM1b-650M', 'UniProt', 'UniRef50 2018_03', 'CC BY 4.0', '30M sequences, <1023 aa', 'Amino acid (33)', 'Masked LM (predict AA)'],
    ['ESM2-650M', 'UniProt', 'UniRef50 (UR50D)', 'CC BY 4.0', '65M sequences, <1022 aa', 'Amino acid (33)', 'Masked LM (predict AA)'],
    ['ProtT5-XL', 'UniProt + BFD', 'UniRef50 + BFD', 'CC BY 4.0', '45M seqs, 393B amino acids', 'Amino acid (33)', 'Span corruption (T5)'],
    ['SaProt-650M', 'UniProt + EMBL-EBI', 'UniRef50 + AFDB', 'CC BY 4.0', '40M structs, pLDDT>70', 'AA + 3Di (446)', 'Masked LM (predict AA)'],
    ['ESM3-sm-1.4B', 'UniProt + EMBL-EBI', 'UniRef50 + AFDB + PDB', 'NC-BY 4.0', 'Multi-modal pretraining', 'AA + struct + SS + Func', 'Multi-modal masked LM'],
    ['AlphaGenome', 'NHGRI + ENCODE', 'GRCh38 + ENCODE', 'Restricted', 'Human genome + regulatory', 'Nucleotide', 'Variant effect scoring'],
    ['NT-v2-500M', 'NCBI + 1000G', '3,202 humans + 850 spp', 'CC BY 4.0', '174B nucleotides', '6-mer (4096)', 'Masked LM (predict 6-mer)'],
    ['HyenaDNA-150M', 'NCBI', 'Human hg38', 'Public', '3.2B nucleotides', 'Nucleotide (5)', 'Next-nucleotide prediction'],
]

table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 2.0)

col_widths = [0.11, 0.13, 0.15, 0.08, 0.18, 0.12, 0.18]
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_widths[col])

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold', fontsize=9)

type_colors = {
    'ESM1b-650M': '#E3F2FD', 'ESM2-650M': '#E3F2FD', 'ProtT5-XL': '#E3F2FD',
    'SaProt-650M': '#F3E5F5', 'ESM3-sm-1.4B': '#E8EAF6',
    'AlphaGenome': '#FFF3E0',
    'NT-v2-500M': '#FFF3E0', 'HyenaDNA-150M': '#FFF3E0',
}
model_names = ['ESM1b-650M', 'ESM2-650M', 'ProtT5-XL', 'SaProt-650M', 'ESM3-sm-1.4B', 'AlphaGenome', 'NT-v2-500M', 'HyenaDNA-150M']
for i, model in enumerate(model_names):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(type_colors.get(model, 'white'))

ax4.set_title('D. Complete Pretraining Data Specification', fontsize=13, fontweight='bold', pad=15)

fig.suptitle('Benchmark V2: Detailed Pretraining Data Types & Provenance',
             fontsize=16, fontweight='bold', y=0.97)

fig.savefig(FIG_DIR / 'fig12_pretraining_details.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig12_pretraining_details.png'}")
