#!/usr/bin/env python3
"""Figure 13: End-to-end project architecture."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2

FIG_DIR = Path(__file__).parent.parent / 'benchmark_v2' / 'figures'

fig = plt.figure(figsize=(30, 24))
gs = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.2, 1.0, 0.8], hspace=0.25,
                       left=0.03, right=0.97, top=0.95, bottom=0.03)

# ══════════════════════════════════════════════════════════════════════════════
# Panel A: Data Acquisition & Filtering
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :])
ax1.axis('off')
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 8)

ax1.text(15, 7.7, 'A. Data Acquisition & Filtering Pipeline', fontsize=14, fontweight='bold', ha='center')

# Step boxes
steps_a = [
    (2.0, 5.5, 'ClinVar\nvariant_summary.txt.gz', '#E3F2FD', '2.1M variants\nGRCh38'),
    (7.0, 5.5, 'MANE Select\nv1.5 GFF', '#E8F5E9', '19,293 transcripts\nSingle isoform'),
    (12.0, 5.5, 'Filter:\nSNVs + Missense\n+ Pathogenic/Benign', '#FFF3E0', '793 variants\n207 genes'),
    (18.0, 5.5, 'Protein Sequences\n(NCBI + UniProt)', '#F3E5F5', '207 proteins\n119-979 aa'),
    (24.0, 5.5, 'DNA Sequences\n(6001bp window)', '#FFEBEE', '793 sequences\nCentered on variant'),
]

for x, y, text, color, desc in steps_a:
    rect = mpatches.FancyBboxPatch((x-2.2, y-0.8), 4.4, 1.6, boxstyle='round,pad=0.15',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax1.add_patch(rect)
    ax1.text(x, y+0.15, text, ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(x, y-0.45, desc, ha='center', va='center', fontsize=8, color='#555', style='italic')

# Arrows
arrows_a = [
    (4.2, 5.5, 4.8, 5.5),
    (9.2, 5.5, 9.8, 5.5),
    (14.2, 5.5, 15.8, 5.5),
    (20.2, 5.5, 21.8, 5.5),
]
for x1, y1, x2, y2 in arrows_a:
    ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                 arrowprops=dict(arrowstyle='->', color='#666', lw=2.0))

# Stats box
ax1.text(15, 2.5, 'Final Dataset: 793 missense variants across 207 genes (571 Pathogenic, 222 Benign)',
         fontsize=11, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD', edgecolor='#1565C0', linewidth=2))

# ══════════════════════════════════════════════════════════════════════════════
# Panel B: Model Inference (7 models)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1, :])
ax2.axis('off')
ax2.set_xlim(0, 30)
ax2.set_ylim(0, 10)

ax2.text(15, 9.7, 'B. Model Inference: Log-Likelihood Ratio (LLR) Scoring', fontsize=14, fontweight='bold', ha='center')

# Protein models
ax2.text(7.5, 8.8, 'Protein Language Models', fontsize=12, fontweight='bold', ha='center', color='#1565C0')
protein_models = [
    (2.0, 7.0, 'ESM1b-650M', '#1E88E5', 'AA tokens (33)\nMasked LM'),
    (5.5, 7.0, 'ESM2-650M', '#1565C0', 'AA tokens (33)\nMasked LM'),
    (9.0, 7.0, 'ESM3-sm-1.4B', '#0D47A1', 'AA+struct (64)\nMulti-modal'),
    (12.5, 7.0, 'ProtT5-XL', '#42A5F5', 'AA tokens (33)\nSpan corruption'),
    (16.0, 7.0, 'SaProt-650M', '#7B1FA2', 'AA+3Di (446)\nFoldseek 3Di'),
]

for x, y, name, color, desc in protein_models:
    rect = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x, y+0.15, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax2.text(x, y-0.3, desc, ha='center', va='center', fontsize=7, color='white')

# DNA models
ax2.text(23.0, 8.8, 'Genome Models', fontsize=12, fontweight='bold', ha='center', color='#E65100')
dna_models = [
    (19.5, 7.0, 'AlphaGenome', '#FF8F00', 'Nucleotide\nENCODE features'),
    (23.5, 7.0, 'NT-v2-500M', '#F57C00', '6-mer (4096)\nMasked LM'),
    (27.0, 7.0, 'HyenaDNA-150M', '#FFB300', 'Nucleotide (5)\nNext-nucleotide'),
]

for x, y, name, color, desc in dna_models:
    rect = mpatches.FancyBboxPatch((x-1.4, y-0.6), 2.8, 1.2, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax2.add_patch(rect)
    ax2.text(x, y+0.15, name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax2.text(x, y-0.3, desc, ha='center', va='center', fontsize=7, color='white')

# Scoring formula
ax2.text(15, 4.5, 'LLR = log P(mut | context) − log P(wt | context)',
         fontsize=12, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))

ax2.text(15, 3.5, 'Pathogenic variants → more negative LLR (deleterious mutations less likely)',
         fontsize=10, ha='center', va='center', color='#555')

# Input arrows
for x in [2.0, 5.5, 9.0, 12.5, 16.0, 19.5, 23.5, 27.0]:
    ax2.annotate('', xy=(x, 7.7), xytext=(x, 8.3),
                 arrowprops=dict(arrowstyle='->', color='#999', lw=1.0, linestyle='dashed'))

# SaProt special note
ax2.text(12.5, 5.5, 'SaProt requires PDB structure → Foldseek 3Di tokenization',
         fontsize=8, ha='center', va='center', color='#7B1FA2', style='italic')

# ══════════════════════════════════════════════════════════════════════════════
# Panel C: Evaluation Pipeline
# ══════════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2, :])
ax3.axis('off')
ax3.set_xlim(0, 30)
ax3.set_ylim(0, 8)

ax3.text(15, 7.7, 'C. Evaluation & Metrics', fontsize=14, fontweight='bold', ha='center')

# Evaluation steps
eval_steps = [
    (3.0, 5.5, 'LLR Scores\nper Variant', '#E3F2FD', '793 variants × 7 models'),
    (9.0, 5.5, 'Threshold\nOptimization', '#E8F5E9', 'Maximize AUROC\non validation set'),
    (15.0, 5.5, 'Binary\nClassification', '#FFF3E0', 'Pathogenic vs Benign\n(pred > threshold = Path)'),
    (21.0, 5.5, 'Metrics\nCalculation', '#F3E5F5', 'AUROC, AUPRC, MCC\nSpearman, Recall'),
    (27.0, 5.5, 'Literature\nComparison', '#FFEBEE', 'Compare with\npublished benchmarks'),
]

for x, y, text, color, desc in eval_steps:
    rect = mpatches.FancyBboxPatch((x-2.2, y-0.7), 4.4, 1.4, boxstyle='round,pad=0.15',
                                     facecolor=color, edgecolor='#333', linewidth=1.5)
    ax3.add_patch(rect)
    ax3.text(x, y+0.15, text, ha='center', va='center', fontsize=10, fontweight='bold')
    ax3.text(x, y-0.35, desc, ha='center', va='center', fontsize=8, color='#555')

# Arrows
for i in range(len(eval_steps)-1):
    x1 = eval_steps[i][0] + 2.2
    x2 = eval_steps[i+1][0] - 2.2
    ax3.annotate('', xy=(x2, 5.5), xytext=(x1, 5.5),
                 arrowprops=dict(arrowstyle='->', color='#666', lw=2.0))

# Metrics detail
metrics = [
    (5.0, 2.5, 'AUROC\n0.848 (ESM2)', '#1565C0'),
    (11.0, 2.5, 'AUPRC\n0.938 (ESM2)', '#42A5F5'),
    (17.0, 2.5, 'Spearman\n0.483 (ESM2)', '#7B1FA2'),
    (23.0, 2.5, 'MCC\n0.547 (ESM2)', '#FF8F00'),
]

for x, y, text, color in metrics:
    rect = mpatches.FancyBboxPatch((x-2.0, y-0.5), 4.0, 1.0, boxstyle='round,pad=0.1',
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.9)
    ax3.add_patch(rect)
    ax3.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax3.text(15, 0.5, 'Best Model: ESM2-650M (AUROC=0.848, AUPRC=0.938)',
         fontsize=12, fontweight='bold', ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#C8E6C9', edgecolor='#2E7D32', linewidth=2))

# ══════════════════════════════════════════════════════════════════════════════
# Panel D: Final Rankings & Outputs
# ══════════════════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[3, :])
ax4.axis('off')
ax4.set_xlim(0, 30)
ax4.set_ylim(0, 6)

ax4.text(15, 5.7, 'D. Results Summary', fontsize=14, fontweight='bold', ha='center')

# Model ranking
models_rank = [
    (1.5, 4.0, 'ESM2-650M', 0.848, '#1565C0'),
    (5.0, 4.0, 'ESM1b-650M', 0.846, '#1E88E5'),
    (8.5, 4.0, 'SaProt-650M', 0.817, '#7B1FA2'),
    (12.0, 4.0, 'ESM3-sm-1.4B', 0.781, '#0D47A1'),
    (15.5, 4.0, 'AlphaGenome', 0.704, '#FF8F00'),
    (19.0, 4.0, 'ProtT5-XL', 0.603, '#42A5F5'),
    (22.5, 4.0, 'NT-v2-500M', 0.568, '#F57C00'),
    (26.0, 4.0, 'HyenaDNA-150M', 0.471, '#FFB300'),
]

for x, y, name, auroc, color in models_rank:
    bar_width = auroc * 3.5
    rect = mpatches.FancyBboxPatch((x-1.2, y-0.3), 2.4, 0.6, boxstyle='round,pad=0.05',
                                     facecolor=color, edgecolor='#333', linewidth=1)
    ax4.add_patch(rect)
    ax4.text(x, y, f'{name}', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    ax4.text(x, y-0.6, f'AUROC={auroc:.3f}', ha='center', va='center', fontsize=9, fontweight='bold')

# Key findings
findings = [
    (5.0, 1.5, 'Protein LMs >> DNA Models', '#1565C0'),
    (15.0, 1.5, 'ESM2 ≈ ESM1b > SaProt > ESM3', '#42A5F5'),
    (25.0, 1.5, 'SaProt + AlphaFold 3Di = 0.817', '#7B1FA2'),
]

for x, y, text, color in findings:
    ax4.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=color, linewidth=1.5))

# Outputs
ax4.text(15, 0.3, 'Outputs: benchmark_v2/results/ | benchmark_v2/figures/ | portable_package/',
         fontsize=9, ha='center', va='center', color='#666', style='italic')

fig.suptitle('Benchmark V2: End-to-End Project Architecture',
             fontsize=18, fontweight='bold', y=0.98)

fig.savefig(FIG_DIR / 'fig13_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig13_architecture.png'}")
