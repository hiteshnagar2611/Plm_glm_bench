#!/usr/bin/env python3
"""Figure 8: Data filtering pipeline — from raw ClinVar to benchmark_v2."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# ── Pipeline data ──
# (label, n_variants, n_genes, color, detail)
steps = [
    ('Raw ClinVar\nSNVs (GRCh38)', 8900000, '~20000', '#90A4AE', 'Single nucleotide variants\nfrom ClinVar database'),
    ('Non-somatic,\nstandard chr', 7500000, '~18000', '#78909C', 'Remove somatic mutations\nChr 1-22, X, Y'),
    ('Pathogenic +\nBenign only', 580000, '~12000', '#607D8B', 'Keep definitive labels\n(Likely P/B excluded)'),
    ('Review stars\n>= 1', 180000, '~8000', '#546E7A', 'Evidence quality filter\n>=1 submitter reviewed'),
    ('Missense +\nsingle-transcript', 25162, 1377, '#1565C0', 'Single AA change\nMANE Select transcripts\n>= 1 star'),
    ('PDB position\nmapped', 2765, 414, '#0D47A1', 'Variant AA position matches\nPDB resolved residue'),
    ('Protein length\n< 1001 aa', 1667, 323, '#4527A0', 'Exclude oversized proteins\nfor model compatibility'),
    ('NCBI full-length\nre-filter', 1457, 271, '#6A1B9A', 'Remove >= 1001 aa genes\n+ missing (IRF3)'),
    ('Remove\nstop-gain', 793, 207, '#E53935', 'Exclude Ter/* variants\n(true missense only)'),
]

n_steps = len(steps)

fig, (ax_bar, ax_flow) = plt.subplots(2, 1, figsize=(16, 10),
                                        gridspec_kw={'height_ratios': [1, 1.2]})

# ══════════════════════════════════════════════════════════════════════════════
# TOP PANEL: Waterfall bar chart showing variant counts (log scale)
# ══════════════════════════════════════════════════════════════════════════════
counts = [s[1] for s in steps]
labels = [s[0] for s in steps]
colors = [s[3] for s in steps]

bars = ax_bar.bar(range(n_steps), counts, color=colors, edgecolor='white', linewidth=1, width=0.7, zorder=3)
ax_bar.set_yscale('log')
ax_bar.set_ylim(500, 2e7)
ax_bar.set_xticks(range(n_steps))
ax_bar.set_xticklabels(labels, fontsize=8, ha='center')
ax_bar.set_ylabel('Number of Variants (log scale)', fontsize=12, fontweight='bold')
ax_bar.set_title('A. Variant Count at Each Filtering Step', fontsize=13, fontweight='bold', pad=10)
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    if count >= 1000:
        label = f'{count:,}'
    else:
        label = str(count)
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
                label, ha='center', va='bottom', fontsize=8, fontweight='bold', color=colors[i])

# Add percentage retained
for i in range(1, n_steps):
    pct = counts[i] / counts[0] * 100
    if pct >= 0.01:
        ax_bar.text(i, counts[i] * 0.3, f'{pct:.2f}%', ha='center', va='bottom',
                    fontsize=6.5, color='white', fontweight='bold')

# ══════════════════════════════════════════════════════════════════════════════
# BOTTOM PANEL: Horizontal flow diagram with arrows
# ══════════════════════════════════════════════════════════════════════════════
ax_flow.axis('off')
ax_flow.set_xlim(-0.5, n_steps - 0.5)
ax_flow.set_ylim(-2.5, 3.5)

# Draw flow boxes
box_width = 0.75
for i, (label, n_var, n_gene, color, detail) in enumerate(steps):
    # Box
    rect = mpatches.FancyBboxPatch(
        (i - box_width/2, 0.3), box_width, 1.8,
        boxstyle="round,pad=0.05", facecolor=color, edgecolor='white',
        linewidth=1.5, alpha=0.9, zorder=3)
    ax_flow.add_patch(rect)

    # Variant count
    if n_var >= 1000:
        var_text = f'{n_var:,}'
    else:
        var_text = str(n_var)
    ax_flow.text(i, 1.6, var_text, ha='center', va='center',
                 fontsize=11, fontweight='bold', color='white', zorder=4)

    # Genes
    ax_flow.text(i, 1.15, f'{n_gene} genes', ha='center', va='center',
                 fontsize=8, color='white', alpha=0.9, zorder=4)

    # Step label above
    ax_flow.text(i, 2.5, label, ha='center', va='center',
                 fontsize=8, fontweight='bold', color='#333333')

    # Detail below
    ax_flow.text(i, -0.5, detail, ha='center', va='top',
                 fontsize=6.5, color='#555555', linespacing=1.4)

    # Arrow to next
    if i < n_steps - 1:
        ax_flow.annotate('', xy=(i + 0.5, 1.2), xytext=(i + 0.25, 1.2),
                         arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5),
                         zorder=5)

# Drop annotations between steps
for i in range(n_steps - 1):
    dropped = counts[i] - counts[i+1]
    pct_kept = counts[i+1] / counts[i] * 100
    if dropped > 0:
        ax_flow.text(i + 0.5, -1.8, f'-{dropped:,}\n({pct_kept:.0f}% kept)',
                     ha='center', va='top', fontsize=5.5, color='#B71C1C',
                     fontweight='bold')

# Legend
ax_flow.text(-0.3, 3.2, 'Filtering Pipeline: ClinVar to Benchmark V2',
             fontsize=13, fontweight='bold', color='#333333')
ax_flow.text(-0.3, 2.8, 'Final: 793 missense variants across 207 genes with PDB structures',
             fontsize=9, color='#666666')

fig.savefig('benchmark_v2/figures/fig8_filtering_pipeline.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 8 saved: benchmark_v2/figures/fig8_filtering_pipeline.png")
