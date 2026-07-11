#!/usr/bin/env python3
"""
V3 Benchmark Data Filtering Pipeline Visualization.
Creates a funnel diagram showing how data was filtered from ClinVar to V3.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'benchmark_v3', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── Style ──
plt.rcParams.update({
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#0f172a',
    'text.color': '#e2e8f0',
    'font.family': 'sans-serif',
})

fig = plt.figure(figsize=(16, 10))

# ════════════════════════════════════════════════════════════
# FUNNEL DIAGRAM (Left side)
# ════════════════════════════════════════════════════════════
ax_funnel = fig.add_axes([0.02, 0.05, 0.55, 0.88])
ax_funnel.set_xlim(0, 10)
ax_funnel.set_ylim(0, 12)
ax_funnel.axis('off')

# Funnel stages
stages = [
    {'label': 'ClinVar Raw', 'variants': '~2M+', 'genes': '—',
     'desc': 'All ClinVar variants\n(GRCh38, SNV, non-somatic,\nstd chr, Path/Benign,\nstars>=1, MANE annotated)',
     'color': '#64748b', 'width': 9.0, 'y': 11.0},
    {'label': 'Missense + Synonymous', 'variants': '25,162', 'genes': '1,377',
     'desc': 'Pathogenic/Benign only\nValid nucleotides (A/T/C/G)',
     'color': '#8b5cf6', 'width': 7.5, 'y': 9.2},
    {'label': 'Protein Sequences Downloaded', 'variants': '~24,500', 'genes': '1,366',
     'desc': '−11 genes: failed\nNCBI protein download',
     'color': '#3b82f6', 'width': 6.2, 'y': 7.4},
    {'label': 'Protein Length < 1,001 aa', 'variants': '~18,500', 'genes': '1,025',
     'desc': '−341 genes: too large\nfor ESM2/ESM1b (1024 tokens)',
     'color': '#06b6d4', 'width': 5.0, 'y': 5.6},
    {'label': 'Stop-Gain Variants Removed', 'variants': '5,932', 'genes': '973',
     'desc': '−52 genes: only had\nstop-gain (Ter/*) mutations',
     'color': '#22c55e', 'width': 4.0, 'y': 3.8},
    {'label': 'V3 Benchmark', 'variants': '5,932', 'genes': '973',
     'desc': '3,625 Pathogenic | 2,307 Benign\n24 chromosomes | 20 AA types',
     'color': '#eab308', 'width': 3.2, 'y': 1.8},
]

for i, stage in enumerate(stages):
    w = stage['width']
    y = stage['y']
    x_center = 5.0
    color = stage['color']

    # Draw trapezoid (funnel segment)
    if i < len(stages) - 1:
        next_w = stages[i+1]['width']
        pts = np.array([
            [x_center - w/2, y + 0.7],
            [x_center + w/2, y + 0.7],
            [x_center + next_w/2, y - 0.1],
            [x_center - next_w/2, y - 0.1],
        ])
    else:
        # Last stage is a rectangle
        pts = np.array([
            [x_center - w/2, y + 0.7],
            [x_center + w/2, y + 0.7],
            [x_center + w/2, y - 0.1],
            [x_center - w/2, y - 0.1],
        ])

    from matplotlib.patches import Polygon
    poly = Polygon(pts, closed=True, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.85)
    ax_funnel.add_patch(poly)

    # Stage label (inside funnel)
    ax_funnel.text(x_center, y + 0.45, stage['label'], ha='center', va='center',
                   fontsize=11, fontweight='bold', color='white', zorder=5)

    # Variant/gene count (right side)
    ax_funnel.text(x_center + w/2 + 0.3, y + 0.55, f"{stage['variants']} variants",
                   ha='left', va='center', fontsize=10, fontweight='bold', color=color, zorder=5)
    ax_funnel.text(x_center + w/2 + 0.3, y + 0.2, f"{stage['genes']} genes",
                   ha='left', va='center', fontsize=9, color='#94a3b8', zorder=5)

    # Description (left side, for non-first stages)
    if i > 0:
        ax_funnel.text(x_center - w/2 - 0.3, y + 0.45, stage['desc'],
                       ha='right', va='center', fontsize=8, color='#94a3b8', zorder=5,
                       linespacing=1.3)

    # Draw removal arrow
    if i > 0 and i < len(stages) - 1:
        removed_genes = int(stages[i-1]['genes'].replace(',','').replace('~','').replace('—','0')) - int(stage['genes'].replace(',',''))
        if removed_genes > 0:
            ax_funnel.annotate('', xy=(x_center + w/2 + 0.1, y + 0.8),
                             xytext=(x_center + stages[i-1]['width']/2 + 0.1, y + 1.5),
                             arrowprops=dict(arrowstyle='->', color='#ef4444', lw=1.5, ls='--'))

# Title
ax_funnel.text(5.0, 11.8, 'V3 Benchmark Data Filtering Pipeline', ha='center', va='center',
               fontsize=16, fontweight='bold', color='white')
ax_funnel.text(5.0, 11.4, 'ClinVar → 5,932 missense variants across 973 genes', ha='center', va='center',
               fontsize=10, color='#94a3b8')

# ════════════════════════════════════════════════════════════
# FILTER DETAILS TABLE (Right side)
# ════════════════════════════════════════════════════════════
ax_table = fig.add_axes([0.60, 0.05, 0.38, 0.88])
ax_table.set_xlim(0, 10)
ax_table.set_ylim(0, 12)
ax_table.axis('off')

# Title
ax_table.text(5.0, 11.5, 'Filtering Criteria', ha='center', va='center',
              fontsize=14, fontweight='bold', color='white')

filters = [
    ('1. Variant Type', 'Missense only (single AA substitution)', '#8b5cf6'),
    ('2. Clinical Significance', 'Pathogenic or Benign only\n(no VUS, Likely, Conflicting)', '#3b82f6'),
    ('3. Transcript', 'MANE Select (one canonical per gene)', '#06b6d4'),
    ('4. Assembly', 'GRCh38 only', '#22c55e'),
    ('5. Somatic', 'Exclude somatic variants', '#eab308'),
    ('6. Chromosomes', 'Standard (1-22, X, Y) only', '#f97316'),
    ('7. Review Status', 'Gold stars >= 1', '#ef4444'),
    ('8. Nucleotides', 'Valid REF/ALT (A, T, C, G)', '#ec4899'),
    ('9. Protein Length', '< 1,001 amino acids', '#8b5cf6'),
    ('10. Stop-Gain', 'Remove genes with only Ter/* variants', '#3b82f6'),
]

for i, (title, desc, color) in enumerate(filters):
    y = 10.5 - i * 0.95

    # Number circle
    circle = plt.Circle((0.6, y), 0.3, facecolor=color, edgecolor='white', linewidth=1, zorder=5)
    ax_table.add_patch(circle)
    ax_table.text(0.6, y, str(i+1), ha='center', va='center', fontsize=10, fontweight='bold', color='white', zorder=6)

    # Title + description
    ax_table.text(1.2, y + 0.15, title, ha='left', va='center', fontsize=10, fontweight='bold', color=color)
    ax_table.text(1.2, y - 0.2, desc, ha='left', va='center', fontsize=8, color='#94a3b8', linespacing=1.3)

    # Connecting line
    if i < len(filters) - 1:
        ax_table.plot([0.6, 0.6], [y - 0.3, y - 0.65], color='#334155', linewidth=1, zorder=3)

# ════════════════════════════════════════════════════════════
# BAR CHART: Variants removed at each step (Bottom right inset)
# ════════════════════════════════════════════════════════════
ax_bar = fig.add_axes([0.62, 0.02, 0.34, 0.35])
ax_bar.set_facecolor('#1e293b')

step_labels = ['ClinVar\nStart', 'P/B\nOnly', 'Protein\nDownload', 'Length\n<1001', 'Stop-Gain\nRemove', 'V3\nFinal']
gene_counts = [1377, 1377, 1366, 1025, 973, 973]
removed = [0, 0, 11, 341, 52, 0]
colors_bar = ['#64748b', '#8b5cf6', '#3b82f6', '#06b6d4', '#22c55e', '#eab308']

bars = ax_bar.bar(range(len(step_labels)), gene_counts, color=colors_bar, edgecolor='white', linewidth=0.8, alpha=0.85)

# Add removed count annotations
for i, (bar, rem) in enumerate(zip(bars, removed)):
    if rem > 0:
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f'−{rem}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#ef4444')
    ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
               f'{gene_counts[i]}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax_bar.set_xticks(range(len(step_labels)))
ax_bar.set_xticklabels(step_labels, fontsize=7, color='#94a3b8')
ax_bar.set_ylabel('Genes', fontsize=9, color='#94a3b8')
ax_bar.set_title('Genes Retained at Each Step', fontsize=10, fontweight='bold', color='white', pad=8)
ax_bar.set_ylim(0, 1600)
ax_bar.tick_params(colors='#94a3b8')
ax_bar.spines['bottom'].set_color('#334155')
ax_bar.spines['left'].set_color('#334155')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='y', alpha=0.2, color='#334155')

# ════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════
out_path = os.path.join(FIG_DIR, 'fig_v3_filtering_pipeline.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Saved: {out_path}")

# Also save individual bar chart
fig2, ax2 = plt.subplots(figsize=(10, 5))
fig2.patch.set_facecolor('#0f172a')
ax2.set_facecolor('#1e293b')

bars2 = ax2.bar(range(len(step_labels)), gene_counts, color=colors_bar, edgecolor='white', linewidth=1, alpha=0.85, width=0.6)
for i, (bar, rem) in enumerate(zip(bars2, removed)):
    if rem > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'−{rem} genes', ha='center', va='bottom', fontsize=11, fontweight='bold', color='#ef4444')
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{gene_counts[i]}', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

ax2.set_xticks(range(len(step_labels)))
ax2.set_xticklabels(step_labels, fontsize=10, fontweight='bold', color='#e2e8f0')
ax2.set_ylabel('Number of Genes', fontsize=12, fontweight='bold', color='#e2e8f0')
ax2.set_title('V3 Benchmark: Gene Count Through Filtering Pipeline', fontsize=14, fontweight='bold', color='white', pad=15)
ax2.set_ylim(0, 1600)
ax2.tick_params(colors='#94a3b8')
ax2.spines['bottom'].set_color('#334155')
ax2.spines['left'].set_color('#334155')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', alpha=0.2, color='#334155')

# Add summary text box
props = dict(boxstyle='round,pad=0.5', facecolor='#1e293b', edgecolor='#334155', alpha=0.9)
ax2.text(0.98, 0.95, 'Final: 973 genes, 5,932 variants\n3,625 Pathogenic | 2,307 Benign',
         transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
         bbox=props, color='#eab308', fontweight='bold')

plt.tight_layout()
out2 = os.path.join(FIG_DIR, 'fig_v3_gene_filtering.png')
fig2.savefig(out2, dpi=200, bbox_inches='tight', facecolor='#0f172a')
plt.close()
print(f"Saved: {out2}")
