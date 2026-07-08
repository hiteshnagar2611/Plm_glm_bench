#!/usr/bin/env python3
"""Figure 7: Our results vs. published literature cross-check with full annotation."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Data: model, our_auroc, n_our, lit_auroc, n_lit, source, year, method, note
data = [
    ('ESM2-650M',     0.848, 793,  0.862, 36537, 'Su et al. (SaProt)',     2024, 'Zero-shot LLR',    'PDB subset bias'),
    ('ESM1b-650M',    0.846, 793,  0.905, 36537, 'Brandes et al.',         2023, 'Zero-shot LLR',    'Smaller PDB subset'),
    ('ProtT5-XL',     0.603, 793,  0.720, 36537, 'Thanabalasingam et al.', 2024, 'LLR only (est.)',  'Cosine sim vs LLR'),
    ('SaProt-650M',   0.817, 715,  0.909, 36537, 'Su et al. (SaProt)',     2024, 'Zero-shot LLR',    'Smaller PDB subset'),
    ('ESM3-sm',       0.776, 789,  0.830, 36537, 'Hayes et al.',           2025, 'Multi-modal LLR',   'Seq + structure'),
    ('AlphaGenome',   0.704, 782,  0.708, 250000,'Lu et al. (VEP-eval)',   2025, 'Delta-score',       'Missense only'),
    ('NT-v2-500M',    0.568, 793,  0.730, 250000,'Dalla-Torre et al.',     2024, 'Fine-tuned',        'Zero-shot vs ft'),
    ('HyenaDNA-150M', 0.471, 793,  0.520, 50000, 'Alfisi et al.',          2025, 'Zero-shot',         'Human-only train'),
]

models     = [d[0] for d in data]
our_auroc  = [d[1] for d in data]
n_our      = [d[2] for d in data]
lit_auroc  = [d[3] for d in data]
n_lit      = [d[4] for d in data]
sources    = [d[5] for d in data]
years      = [d[6] for d in data]
methods    = [d[7] for d in data]
notes      = [d[8] for d in data]
types      = ['protein' if m in ['ESM2-650M','ESM1b-650M','ProtT5-XL','SaProt-650M','ESM3-sm'] else 'dna' for m in models]

# ── Figure layout: 2 panels on top, full-width table on bottom ──
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7], hspace=0.35, wspace=0.35)

# ── Panel A: Grouped bar chart ──
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(models))
width = 0.35
colors_our = ['#1565C0' if t == 'protein' else '#E65100' for t in types]
colors_lit = ['#BBDEFB' if t == 'protein' else '#FFE0B2' for t in types]

bars1 = ax1.bar(x - width/2, our_auroc, width, label='Our Benchmark',
                color=colors_our, edgecolor='white', linewidth=0.5, zorder=3)
bars2 = ax1.bar(x + width/2, lit_auroc, width, label='Literature',
                color=colors_lit, edgecolor='gray', linewidth=0.5, hatch='///', zorder=3)

for bar, val in zip(bars1, our_auroc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1565C0')
for bar, val in zip(bars2, lit_auroc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
             f'{val:.3f}', ha='center', va='bottom', fontsize=7.5, color='#666666')

ax1.set_xticks(x)
ax1.set_xticklabels([m.replace('-', '-\n', 1) if len(m) > 10 else m for m in models],
                     rotation=0, ha='center', fontsize=8)
ax1.set_ylabel('AUROC', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 1.10)
ax1.set_title('A. Zero-Shot Variant Pathogenicity Prediction', fontsize=12, fontweight='bold', pad=8)
ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)

for i, (o, l) in enumerate(zip(our_auroc, lit_auroc)):
    diff = abs(o - l)
    y_pos = max(o, l) + 0.04
    if diff < 0.03:
        ax1.text(i, y_pos, 'MATCH', fontsize=6, ha='center', color='#2E7D32', fontweight='bold')
    elif diff < 0.10:
        ax1.text(i, y_pos, 'CLOSE', fontsize=6, ha='center', color='#E65100', fontweight='bold')
    else:
        ax1.text(i, y_pos, 'DEVIANT', fontsize=6, ha='center', color='#C62828', fontweight='bold')

# ── Panel B: Scatter with annotations ──
ax2 = fig.add_subplot(gs[0, 1])
for i in range(len(models)):
    marker = 'o' if types[i] == 'protein' else 's'
    color = '#1565C0' if types[i] == 'protein' else '#E65100'
    ax2.scatter(lit_auroc[i], our_auroc[i], c=color, s=120, marker=marker,
                edgecolors='white', linewidth=0.8, zorder=3)

ax2.plot([0.4, 0.95], [0.4, 0.95], 'k--', alpha=0.3, linewidth=1)
ax2.set_xlabel('Published Literature AUROC', fontsize=10, fontweight='bold')
ax2.set_ylabel('Our Benchmark AUROC', fontsize=10, fontweight='bold')
ax2.set_xlim(0.45, 0.98)
ax2.set_ylim(0.40, 0.98)
ax2.set_title('B. Agreement with Literature', fontsize=12, fontweight='bold', pad=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

offsets = {
    'ESM2-650M': (8, 8), 'ESM1b-650M': (8, -12), 'ProtT5-XL': (8, -10),
    'SaProt-650M': (8, -14), 'AlphaGenome': (8, 8), 'NT-v2-500M': (8, -10),
    'HyenaDNA-150M': (8, -10)
}
for i, m in enumerate(models):
    ox, oy = offsets.get(m, (8, 5))
    ax2.annotate(m, (lit_auroc[i], our_auroc[i]), fontsize=7, ha='left', va='bottom',
                 xytext=(ox, oy), textcoords='offset points')

blue_patch = mpatches.Patch(color='#1565C0', label='Protein Model')
orange_patch = mpatches.Patch(color='#E65100', label='DNA Model')
ax2.legend(handles=[blue_patch, orange_patch], loc='lower right', fontsize=7.5)

# ── Panel C: Full-width table at bottom ──
ax3 = fig.add_subplot(gs[1, :])
ax3.axis('off')

col_labels = ['Model', 'Literature Source', 'Year', 'Scoring Method', 'Our N', 'Lit N', 'Gap (Our-Lit)', 'Reason for Gap']
table_data = []
for i in range(len(models)):
    gap = our_auroc[i] - lit_auroc[i]
    gap_str = f'{gap:+.3f}'
    table_data.append([
        models[i],
        sources[i],
        str(years[i]),
        methods[i],
        f'{n_our[i]:,}',
        f'{n_lit[i]:,}',
        gap_str,
        notes[i],
    ])

table = ax3.table(cellText=table_data, colLabels=col_labels, loc='center',
                  cellLoc='center', colLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.8)

# Set column widths manually
col_widths = [0.12, 0.16, 0.05, 0.13, 0.07, 0.08, 0.09, 0.15]
for (row, col), cell in table.get_celld().items():
    cell.set_width(col_widths[col])

for j in range(len(col_labels)):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold', fontsize=8)
for i in range(len(models)):
    color = '#E3F2FD' if types[i] == 'protein' else '#FFF3E0'
    for j in range(len(col_labels)):
        table[i+1, j].set_facecolor(color)
    gap_val = our_auroc[i] - lit_auroc[i]
    if abs(gap_val) < 0.03:
        table[i+1, 6].set_facecolor('#C8E6C9')
    elif abs(gap_val) < 0.10:
        table[i+1, 6].set_facecolor('#FFE0B2')
    else:
        table[i+1, 6].set_facecolor('#FFCDD2')

ax3.set_title('C. Literature Sources & Dataset Comparison', fontsize=12, fontweight='bold', pad=15)

fig.suptitle('Benchmark V2: Cross-Check Against Published Literature (ClinVar Missense Variants)',
             fontsize=14, fontweight='bold', y=0.98)

fig.savefig('benchmark_v2/figures/fig7_literature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved: benchmark_v2/figures/fig7_literature_comparison.png")
