#!/usr/bin/env python3
"""Figure 11: Pretraining data comparison across models."""

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

# ── Model pretraining data ──
models = [
    {
        'name': 'ESM1b-650M',
        'data': 'UniRef50',
        'version': '2018_03',
        'size_m': 30,  # millions of sequences
        'type': 'protein',
        'tokens_b': 2.2,  # billions of tokens (amino acids)
        'organisms': 'UniProt (all species)',
        'year': 2020,
        'color': '#1E88E5',
    },
    {
        'name': 'ESM2-650M',
        'data': 'UniRef50',
        'version': 'UR50D',
        'size_m': 65,
        'type': 'protein',
        'tokens_b': 6.5,
        'organisms': 'UniProt (all species)',
        'year': 2022,
        'color': '#1565C0',
    },
    {
        'name': 'ProtT5-XL',
        'data': 'UniRef50 + BFD',
        'version': '',
        'size_m': 45,
        'type': 'protein',
        'tokens_b': 393,  # 393B amino acids
        'organisms': 'UniProt + BFD metagenomic',
        'year': 2021,
        'color': '#42A5F5',
    },
    {
        'name': 'SaProt-650M',
        'data': 'UniRef50 + AF2',
        'version': '',
        'size_m': 40,
        'type': 'protein+structure',
        'tokens_b': 4.0,
        'organisms': 'UniProt + AlphaFold2 structures',
        'year': 2024,
        'color': '#7B1FA2',
    },
    {
        'name': 'ESM3-sm-1.4B',
        'data': 'UniRef50 + AFDB + PDB',
        'version': 'esm3_sm_open_v1',
        'size_m': 1400,  # 1.4B params, trained on large data
        'type': 'protein+structure',
        'tokens_b': 50,  # estimated
        'organisms': 'UniProt + AlphaFold DB + PDB structures',
        'year': 2025,
        'color': '#0D47A1',
    },
    {
        'name': 'AlphaGenome',
        'data': 'Human genome + regulatory',
        'version': '',
        'size_m': 0,  # not disclosed
        'type': 'dna',
        'tokens_b': 0,
        'organisms': 'Human (GRCh38) + ENCODE',
        'year': 2025,
        'color': '#E65100',
    },
    {
        'name': 'NT-v2-500M',
        'data': '3,202 human genomes + 850 species',
        'version': '1000 Genomes',
        'size_m': 0,
        'type': 'dna',
        'tokens_b': 300,  # ~300B tokens
        'organisms': '3,202 humans + 850 species',
        'year': 2024,
        'color': '#F57C00',
    },
    {
        'name': 'HyenaDNA-150M',
        'data': 'Human reference genome',
        'version': 'hg38',
        'size_m': 0,
        'type': 'dna',
        'tokens_b': 3.2,  # 3.2B nucleotides
        'organisms': 'Human (1 genome)',
        'year': 2023,
        'color': '#FFB300',
    },
]

# ── Figure: 4-panel pretraining comparison ──
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.30,
                       left=0.08, right=0.95, top=0.92, bottom=0.06)

# ── Panel A: Training data size (log scale) ──
ax1 = fig.add_subplot(gs[0, 0])
model_names = [m['name'] for m in models]
# Use tokens as measure of data scale
token_vals = [max(m['tokens_b'], 0.1) for m in models]  # min 0.1 for log scale
colors = [m['color'] for m in models]

bars = ax1.barh(range(len(model_names)), token_vals, color=colors, edgecolor='white', linewidth=0.5, zorder=3, height=0.65)
ax1.set_xscale('log')
ax1.set_yticks(range(len(model_names)))
ax1.set_yticklabels(model_names, fontsize=10)
ax1.set_xlabel('Training Tokens (billions, log scale)', fontsize=11, fontweight='bold')
ax1.set_title('A. Pretraining Data Scale', fontsize=13, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add value labels
for i, (bar, val, m) in enumerate(zip(bars, token_vals, models)):
    if m['tokens_b'] > 0:
        label = f'{val:.1f}B' if val < 10 else f'{val:.0f}B'
    else:
        label = 'Not disclosed'
    ax1.text(bar.get_width() * 1.15, bar.get_y() + bar.get_height()/2,
             label, va='center', fontsize=9, fontweight='bold')

# ── Panel B: Sequence count ──
ax2 = fig.add_subplot(gs[0, 1])
seq_counts = []
for m in models:
    if m['size_m'] > 0:
        seq_counts.append(m['size_m'])
    else:
        seq_counts.append(0)

# Only show protein models with sequence counts
protein_models = [(m['name'], m['size_m'], m['color']) for m in models if m['size_m'] > 0]
names_p = [p[0] for p in protein_models]
counts_p = [p[1] for p in protein_models]
colors_p = [p[2] for p in protein_models]

bars2 = ax2.bar(range(len(names_p)), counts_p, color=colors_p, edgecolor='white', linewidth=0.5, zorder=3, width=0.6)
ax2.set_xticks(range(len(names_p)))
ax2.set_xticklabels(names_p, fontsize=9, rotation=20, ha='right')
ax2.set_ylabel('Sequences (millions)', fontsize=11, fontweight='bold')
ax2.set_title('B. Protein Sequence Count', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for bar, val in zip(bars2, counts_p):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val}M', ha='center', fontsize=10, fontweight='bold')

# ── Panel C: Data type breakdown ──
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 7)

# Protein models
ax3.text(1.5, 6.3, 'Protein Language Models', fontsize=12, fontweight='bold', color='#1565C0')
protein_data = [
    ('ESM1b', 'UniRef50 (30M seqs)', '#1E88E5'),
    ('ESM2', 'UniRef50 (65M seqs)', '#1565C0'),
    ('ProtT5', 'UniRef50 + BFD (393B AA)', '#42A5F5'),
    ('SaProt', 'UniRef50 + AF2 (40M structs)', '#7B1FA2'),
]
for i, (name, data, color) in enumerate(protein_data):
    y = 5.8 - i * 0.45
    rect = mpatches.FancyBboxPatch((0.5, y-0.12), 0.2, 0.24, boxstyle='round,pad=0.02',
                                     facecolor=color, edgecolor='white')
    ax3.add_patch(rect)
    ax3.text(0.9, y, f'{name}:', fontsize=10, fontweight='bold', va='center')
    ax3.text(3.0, y, data, fontsize=10, va='center', color='#333')

# DNA models
ax3.text(1.5, 3.8, 'DNA Language Models', fontsize=12, fontweight='bold', color='#E65100')
dna_data = [
    ('AlphaGenome', 'Human genome + ENCODE regulatory', '#E65100'),
    ('NT-v2', '3,202 human + 850 species (300B tokens)', '#F57C00'),
    ('HyenaDNA', 'Human reference genome hg38 (3.2B nt)', '#FFB300'),
]
for i, (name, data, color) in enumerate(dna_data):
    y = 3.3 - i * 0.45
    rect = mpatches.FancyBboxPatch((0.5, y-0.12), 0.2, 0.24, boxstyle='round,pad=0.02',
                                     facecolor=color, edgecolor='white')
    ax3.add_patch(rect)
    ax3.text(0.9, y, f'{name}:', fontsize=10, fontweight='bold', va='center')
    ax3.text(3.0, y, data, fontsize=10, va='center', color='#333')

ax3.set_title('C. Pretraining Data Sources', fontsize=13, fontweight='bold', pad=10)

# ── Panel D: Key differences table ──
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

col_labels = ['Model', 'Training Data', 'Sequences', 'Key Feature']
table_data = [
    ['ESM1b-650M', 'UniRef50', '30M', 'First large protein LM'],
    ['ESM2-650M', 'UniRef50 (dense)', '65M', 'Scaled architecture'],
    ['ProtT5-XL', 'UniRef50 + BFD', '45M', 'Encoder-decoder, 393B AA'],
    ['SaProt-650M', 'UniRef50 + AF2', '40M', 'Structure-aware tokens'],
    ['ESM3-sm-1.4B', 'UniRef50 + AFDB + PDB', '100M+', 'Multi-modal (seq+struct+func)'],
    ['AlphaGenome', 'Human genome', 'N/A', 'Google DeepMind, API'],
    ['NT-v2-500M', '3,202 humans + 850 spp', 'N/A', '300B tokens, multi-species'],
    ['HyenaDNA-150M', 'Human hg38', 'N/A', 'Single genome, long-range'],
]

table = ax4.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

col_widths = [0.18, 0.28, 0.12, 0.35]
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
for i, model in enumerate([m['name'] for m in models]):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(type_colors.get(model, 'white'))

ax4.set_title('D. Pretraining Data Comparison', fontsize=13, fontweight='bold', pad=15)

fig.suptitle('Benchmark V2: Pretraining Data for Pathogenic Variant Prediction Models',
             fontsize=15, fontweight='bold', y=0.98)

fig.savefig(FIG_DIR / 'fig11_pretraining_data.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig11_pretraining_data.png'}")
