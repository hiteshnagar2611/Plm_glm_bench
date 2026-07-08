#!/usr/bin/env python3
"""Figure 10: Scoring methodology comparison across models."""

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

# ── Model data ──
models = {
    'ESM2-650M': {
        'type': 'Protein LM', 'input': 'Protein sequence',
        'method': 'Masked log-likelihood\nratio (LLR)',
        'scoring': 'P(wild-type) vs P(mutant)\nat variant position',
        'color': '#1565C0', 'auroc': 0.848,
    },
    'ESM1b-650M': {
        'type': 'Protein LM', 'input': 'Protein sequence',
        'method': 'Masked log-likelihood\nratio (LLR)',
        'scoring': 'P(wild-type) vs P(mutant)\nat variant position',
        'color': '#1E88E5', 'auroc': 0.846,
    },
    'ProtT5-XL': {
        'type': 'Protein LM', 'input': 'Protein sequence',
        'method': 'Encoder-decoder\nlog-likelihood',
        'scoring': 'LLR of wild-type vs mutant\nprotein sequence',
        'color': '#42A5F5', 'auroc': 0.603,
    },
    'SaProt-650M': {
        'type': 'Protein+Structure LM', 'input': 'Protein + 3Di\nstructure tokens',
        'method': 'Masked log-likelihood\nratio (LLR)',
        'scoring': 'P(wild-type) vs P(mutant)\nat AA+3Di token position',
        'color': '#7B1FA2', 'auroc': 0.817,
    },
    'ESM3-sm-struct': {
        'type': 'Multi-modal Protein LM', 'input': 'Protein seq +\n3D coordinates',
        'method': 'Multi-modal log-likelihood\nratio (LLR)',
        'scoring': 'P(wild-type) vs P(mutant)\nseq + structure conditioned',
        'color': '#0D47A1', 'auroc': 0.776,
    },
    'AlphaGenome': {
        'type': 'Genome Model', 'input': 'DNA sequence\n(6001 bp)',
        'method': 'Delta-score\n(api_score)',
        'scoring': 'Score(reference) - Score(alternate)\nacross all positions',
        'color': '#E65100', 'auroc': 0.704,
    },
    'NT-v2-500M': {
        'type': 'Nucleotide LM', 'input': 'DNA sequence\n(6001 bp)',
        'method': 'Hidden state\nmean difference',
        'scoring': '|H(reference) - H(alternate)|\nmean over layers',
        'color': '#F57C00', 'auroc': 0.568,
    },
    'HyenaDNA-150M': {
        'type': 'Nucleotide LM', 'input': 'DNA sequence\n(6001 bp)',
        'method': 'Log-likelihood\nratio (LLR)',
        'scoring': 'P(reference) vs P(alternate)\nper-nucleotide',
        'color': '#FFB300', 'auroc': 0.471,
    },
}

# ── Figure: methodology overview ──
fig = plt.figure(figsize=(18, 14))
gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 0.8, 1.5], hspace=0.40, wspace=0.30,
                       left=0.08, right=0.95, top=0.92, bottom=0.05)

# ── Top row: Methodology schematic (single wide panel) ──
ax_schema = fig.add_subplot(gs[0, :])
ax_schema.axis('off')
ax_schema.set_xlim(0, 10)
ax_schema.set_ylim(0, 4)

# Input types
input_boxes = [
    (1.0, 3.2, 'Protein\nSequence', '#BBDEFB'),
    (4.0, 3.2, 'Protein +\nStructure (3Di)', '#CE93D8'),
    (7.0, 3.2, 'DNA Sequence\n(6001 bp)', '#FFE0B2'),
]
for x, y, text, color in input_boxes:
    rect = mpatches.FancyBboxPatch((x-0.8, y-0.4), 1.6, 0.8, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax_schema.add_patch(rect)
    ax_schema.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows from inputs to methods
arrows = [
    (1.0, 2.8, 1.0, 2.2, '#1565C0'),   # Protein -> ESM2/ESM1b/ProtT5
    (1.0, 2.8, 2.5, 2.2, '#42A5F5'),   # Protein -> ProtT5
    (4.0, 2.8, 4.0, 2.2, '#7B1FA2'),   # Structure -> SaProt
    (7.0, 2.8, 5.5, 2.2, '#E65100'),   # DNA -> AlphaGenome
    (7.0, 2.8, 7.0, 2.2, '#F57C00'),   # DNA -> NT-v2
    (7.0, 2.8, 8.5, 2.2, '#FFB300'),   # DNA -> HyenaDNA
]
for x1, y1, x2, y2, color in arrows:
    ax_schema.annotate('', xy=(x2, y2+0.1), xytext=(x1, y1-0.1),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2, connectionstyle='arc3,rad=0'))

# Method boxes
method_boxes = [
    (1.0, 1.8, 'Masked LM\n(LLR)', '#1565C0'),
    (2.5, 1.8, 'Encoder-Decoder\n(LLR)', '#42A5F5'),
    (4.0, 1.8, 'Masked LM\n(LLR + 3Di)', '#7B1FA2'),
    (5.5, 1.8, 'Delta-Score\n(Variant Effect)', '#E65100'),
    (7.0, 1.8, 'Hidden State\nDifference', '#F57C00'),
    (8.5, 1.8, 'Nucleotide\nLLR', '#FFB300'),
]
for x, y, text, color in method_boxes:
    rect = mpatches.FancyBboxPatch((x-0.75, y-0.4), 1.5, 0.8, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
    ax_schema.add_patch(rect)
    ax_schema.text(x, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='white')

# Output
output_rect = mpatches.FancyBboxPatch((3.5, 0.3), 3.0, 0.8, boxstyle='round,pad=0.1',
                                        facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2)
ax_schema.add_patch(output_rect)
ax_schema.text(5.0, 0.7, 'Pathogenicity Score\n(LLR or Delta)', ha='center', va='center',
               fontsize=11, fontweight='bold', color='#2E7D32')

# Arrows from methods to output
for x in [1.0, 2.5, 4.0, 5.5, 7.0, 8.5]:
    ax_schema.annotate('', xy=(5.0, 1.1), xytext=(x, 1.4),
                       arrowprops=dict(arrowstyle='->', color='#666', lw=1.2,
                                       connectionstyle='arc3,rad=0'))

ax_schema.set_title('Scoring Methodology Overview', fontsize=14, fontweight='bold', pad=10)

# ── Middle row: AUROC comparison ──
ax_bar = fig.add_subplot(gs[1, 0])
model_names = list(models.keys())
aurocs = [models[m]['auroc'] for m in model_names]
colors = [models[m]['color'] for m in model_names]
bars = ax_bar.barh(range(len(model_names)), aurocs, color=colors, edgecolor='white', linewidth=0.5, zorder=3, height=0.7)
ax_bar.set_yticks(range(len(model_names)))
ax_bar.set_yticklabels(model_names, fontsize=10)
ax_bar.set_xlabel('AUROC', fontsize=11, fontweight='bold')
ax_bar.set_title('Zero-Shot Performance', fontsize=12, fontweight='bold')
ax_bar.set_xlim(0, 1.0)
ax_bar.axvline(0.5, color='gray', linestyle=':', alpha=0.4)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
for bar, val in zip(bars, aurocs):
    ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

# ── Model type legend ──
ax_legend = fig.add_subplot(gs[1, 1])
ax_legend.axis('off')
legend_items = [
    ('Protein Language Model', '#1565C0', 'ESM2, ESM1b, ProtT5'),
    ('Protein + Structure LM', '#7B1FA2', 'SaProt (3Di tokens)'),
    ('Genome Model (API)', '#E65100', 'AlphaGenome'),
    ('Nucleotide Language Model', '#F57C00', 'NT-v2, HyenaDNA'),
]
for i, (label, color, models_str) in enumerate(legend_items):
    y = 0.85 - i * 0.22
    rect = mpatches.FancyBboxPatch((0.05, y-0.06), 0.12, 0.12, boxstyle='round,pad=0.02',
                                     facecolor=color, edgecolor='white', linewidth=1)
    ax_legend.add_patch(rect)
    ax_legend.text(0.22, y+0.02, label, fontsize=11, fontweight='bold', va='center')
    ax_legend.text(0.22, y-0.05, models_str, fontsize=9, color='#666', va='center')
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_title('Model Categories', fontsize=12, fontweight='bold')

# ── Bottom row: Key technique differences table ──
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')

col_labels = ['Model', 'Input', 'Scoring Method', 'Direction', 'Key Advantage']
table_data = [
    ['ESM2-650M', 'Protein seq', 'Masked LM LLR', 'Higher = pathogenic', 'Large pre-training, captures evolutionary patterns'],
    ['ESM1b-650M', 'Protein seq', 'Masked LM LLR', 'Higher = pathogenic', 'Well-validated, strong zero-shot performance'],
    ['ProtT5-XL', 'Protein seq', 'Encoder-decoder LLR', 'Higher = pathogenic', '3B params, understands sequence context'],
    ['SaProt-650M', 'Protein + 3Di', 'Masked LM LLR', 'Higher = pathogenic', 'Structure-aware, captures 3D effects'],
    ['ESM3-sm', 'Protein + 3D coords', 'Multi-modal LLR', 'Higher = pathogenic', '1.4B params, structure-conditioned scoring'],
    ['AlphaGenome', 'DNA (6001 bp)', 'Delta-score', 'Higher = pathogenic', 'Google DeepMind, directly models DNA'],
    ['NT-v2-500M', 'DNA (6001 bp)', 'Hidden state diff', 'Higher = pathogenic', 'Pre-trained on 3.2B nucleotides'],
    ['HyenaDNA-150M', 'DNA (6001 bp)', 'Nucleotide LLR', 'Higher = pathogenic', 'Long-range dependencies via Hyena layers'],
]

table = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center',
                       cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.8)

col_widths = [0.12, 0.10, 0.15, 0.15, 0.40]
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_widths[col])

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold', fontsize=9)

# Color rows by model type
type_colors = {
    'ESM2-650M': '#E3F2FD', 'ESM1b-650M': '#E3F2FD', 'ProtT5-XL': '#E3F2FD',
    'SaProt-650M': '#F3E5F5', 'ESM3-sm-struct': '#E8EAF6',
    'AlphaGenome': '#FFF3E0',
    'NT-v2-500M': '#FFF3E0', 'HyenaDNA-150M': '#FFF3E0',
}
for i, model in enumerate(model_names):
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(type_colors.get(model, 'white'))

ax_table.set_title('Model Comparison Details', fontsize=12, fontweight='bold', pad=15)

fig.suptitle('Benchmark V2: Scoring Methodology for Pathogenic Variant Prediction',
             fontsize=15, fontweight='bold', y=0.98)

fig.savefig(FIG_DIR / 'fig10_scoring_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig10_scoring_methods.png'}")
