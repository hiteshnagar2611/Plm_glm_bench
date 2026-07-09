#!/usr/bin/env python3
"""Literature vs V3 benchmark comparison figure."""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ── Load V3 benchmark ────────────────────────────────────────────────────────
df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
if df['ClinVar_label'].dtype in ['float64', 'int64']:
    df['label'] = df['ClinVar_label'].map({1.0: 1, 0.0: 0})
else:
    df['label'] = df['ClinVar_label'].map({'Pathogenic': 1, 'Benign': 0})
valid_vids = set(df['VariationID'])

V3_N_VARS = len(df)
V3_N_GENES = df['GeneSymbol'].nunique()
V3_N_PATH = (df['label'] == 1).sum()
V3_N_BEN = (df['label'] == 0).sum()

# ── Load scores ──────────────────────────────────────────────────────────────
model_configs = {
    'ESM1b-650M':    {'file': 'benchmark_v3/results/esm1b_scores.csv',     'score_col': 'ESM1b_LLR',          'negate': True,  'type': 'protein'},
    'ESM2-650M':     {'file': 'benchmark_v3/results/esm2_650m_scores.csv',  'score_col': 'ESM2_LLR',           'negate': True,  'type': 'protein'},
    'SaProt-650M':   {'file': 'benchmark_v3/results/saprot_scores.csv',     'score_col': 'SaProt_LLR',         'negate': True,  'type': 'protein'},
    'ProtT5-XL':     {'file': 'benchmark_v3/results/prott5_scores.csv',     'score_col': 'ProtT5_score',       'negate': True,  'type': 'protein'},
    'AlphaGenome':   {'file': 'benchmark_v3/results/alphagenome_scores.csv', 'score_col': 'AlphaGenome_delta',  'negate': False, 'type': 'dna'},
    'NT-v2-500M':    {'file': 'benchmark_v3/results/ntv2_scores.csv',       'score_col': 'NTv2_delta',         'negate': False, 'type': 'dna'},
    'HyenaDNA-150M': {'file': 'benchmark_v3/results/hyena_scores.csv',      'score_col': 'HyenaDNA_LLR',       'negate': True,  'type': 'dna'},
}

# ── Font setup ───────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

FIGURES_DIR = 'benchmark_v3/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Compute our V3 metrics ───────────────────────────────────────────────────
our_results = {}
for model_name, config in model_configs.items():
    try:
        scores = pd.read_csv(config['file'])
    except Exception as e:
        print(f"  {model_name}: SKIPPED ({e})")
        continue

    scores = scores[scores['VariationID'].isin(valid_vids)].copy()
    scores = scores.dropna(subset=[config['score_col']])

    if len(scores) < 50:
        continue

    if 'label' not in scores.columns:
        if 'ClinVar_label' in scores.columns:
            if scores['ClinVar_label'].dtype in ['float64', 'int64']:
                scores['label'] = scores['ClinVar_label'].map({1.0: 1, 0.0: 0})
            else:
                scores['label'] = scores['ClinVar_label'].map({'Pathogenic': 1, 'Benign': 0})
        else:
            continue

    y_true = scores['label'].values.astype(int)
    y_score = scores[config['score_col']].values
    if config['negate']:
        y_score = -y_score

    auroc = roc_auc_score(y_true, y_score)
    our_results[model_name] = {
        'auroc': auroc,
        'n_variants': len(scores),
        'n_genes': scores['GeneSymbol'].nunique(),
        'type': config['type'],
    }

# ── Literature comparison data ───────────────────────────────────────────────
lit = {
    'ESM1b-650M': {
        'auroc': 0.905,
        'benchmark': 'ClinVar (Brandes 2023)',
        'n_variants': '36,537',
        'n_genes': '2,765',
        'method': 'LLR',
        'source': 'Nature Genetics 2023',
    },
    'ESM2-650M': {
        'auroc': 0.862,
        'benchmark': 'ClinVar (EVE subset)',
        'n_variants': '~30,000',
        'n_genes': '~2,000',
        'method': 'LLR',
        'source': 'SaProt paper, ICLR 2024',
    },
    'SaProt-650M': {
        'auroc': 0.909,
        'benchmark': 'ClinVar (AF2 struct)',
        'n_variants': '~30,000',
        'n_genes': '~2,000',
        'method': 'LLR + 3Di',
        'source': 'SaProt paper, ICLR 2024',
    },
    'ProtT5-XL': {
        'auroc': 0.610,
        'benchmark': 'ClinVar (various)',
        'n_variants': 'varies',
        'n_genes': 'varies',
        'method': 'Embedding sim',
        'source': 'Elnaggar et al. 2022',
    },
    'AlphaGenome': {
        'auroc': None,
        'benchmark': 'ClinVar (splicing)',
        'n_variants': '~2,000',
        'n_genes': '~50',
        'method': 'Splice delta',
        'source': 'Nature 2026',
    },
    'NT-v2-500M': {
        'auroc': 0.780,
        'benchmark': 'ClinVar (Dalla-Torre)',
        'n_variants': '~100,000',
        'n_genes': '~10,000',
        'method': 'Embed delta',
        'source': 'Nature Methods 2025',
    },
    'HyenaDNA-150M': {
        'auroc': 0.550,
        'benchmark': 'ClinVar (OpenReview)',
        'n_variants': '~50,000',
        'n_genes': '~2,000',
        'method': 'LLR',
        'source': 'OpenReview 2024',
    },
}

# ── Model order (sorted by our V3 AUROC) ────────────────────────────────────
model_order = sorted(our_results.keys(), key=lambda m: our_results[m]['auroc'], reverse=True)

# ── Colors ────────────────────────────────────────────────────────────────────
color_our = '#2196F3'
color_lit = '#FF9800'
color_protein = '#1565C0'
color_dna = '#E65100'

# ── Figure: Horizontal bar chart (no overlap for labels) ─────────────────────
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 1, height_ratios=[4, 4], hspace=0.4)
ax_bar = fig.add_subplot(gs[0])
ax_table = fig.add_subplot(gs[1])

# Horizontal bars: y positions for each model
y_pos = np.arange(len(model_order))
bar_height = 0.35

our_vals = [our_results[m]['auroc'] for m in model_order]
lit_vals = [lit[m]['auroc'] if lit[m]['auroc'] is not None else 0 for m in model_order]
lit_available = [lit[m]['auroc'] is not None for m in model_order]

bars_our = ax_bar.barh(y_pos + bar_height/2, our_vals, bar_height, label='This study (V3)',
                       color=color_our, edgecolor='white', linewidth=0.5, zorder=3)
bars_lit = ax_bar.barh(y_pos - bar_height/2, lit_vals, bar_height, label='Literature',
                       color=color_lit, edgecolor='white', linewidth=0.5, zorder=3)

# Annotate values
for bar, val in zip(bars_our, our_vals):
    ax_bar.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold', color=color_our)

for bar, val, avail in zip(bars_lit, lit_vals, lit_available):
    if avail:
        ax_bar.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold', color=color_lit)
    else:
        ax_bar.text(0.01, bar.get_y() + bar.get_height()/2,
                    'N/A (splicing only)', ha='left', va='center', fontsize=9, fontstyle='italic', color='#999999')

# Model type markers (left of y-axis)
for i, m in enumerate(model_order):
    marker = '\u25CF' if our_results[m]['type'] == 'protein' else '\u25B2'
    color = color_protein if our_results[m]['type'] == 'protein' else color_dna
    ax_bar.text(-0.02, i, marker, ha='right', va='center', fontsize=12, color=color)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(model_order, fontsize=11, fontweight='bold')
ax_bar.set_xlabel('AUROC', fontsize=13, fontweight='bold')
ax_bar.set_xlim(0.35, 1.0)
ax_bar.invert_yaxis()
ax_bar.legend(fontsize=11, loc='lower right')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.grid(axis='x', alpha=0.3, linestyle='--', zorder=0)

# Type legend (below chart)
ax_bar.text(0.55, 0.02, '\u25CF Protein LM', transform=ax_bar.transAxes, fontsize=10, color=color_protein)
ax_bar.text(0.72, 0.02, '\u25B2 DNA/Gene LM', transform=ax_bar.transAxes, fontsize=10, color=color_dna)

# ── Table: Comparison details ─────────────────────────────────────────────────
ax_table.axis('off')
ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)

# Simplified headers for better fit
headers = ['Model', 'Lit Benchmark', 'Our Benchmark', 'Lit Vars', 'Our Vars',
           'Lit Method', 'Our Method', 'Lit AUROC', 'Our AUROC']

table_data = []
for m in model_order:
    l = lit[m]
    lit_auroc_str = f"{l['auroc']:.3f}" if l['auroc'] is not None else "N/A"
    # Our method description
    our_methods = {
        'ESM1b-650M':    'LLR: -logP(mut)/P(wt)',
        'ESM2-650M':     'LLR: -logP(mut)/P(wt)',
        'SaProt-650M':   'LLR+3Di',
        'ProtT5-XL':     'Cosine sim',
        'AlphaGenome':   'Delta: |ref-alt|',
        'NT-v2-500M':    'Hidden state delta',
        'HyenaDNA-150M': 'LLR: -logP(alt)/P(ref)',
    }
    table_data.append([
        m,
        l['benchmark'],
        f"V3: {V3_N_GENES} genes",
        l['n_variants'],
        f"{our_results[m]['n_variants']:,}",
        l['method'],
        our_methods.get(m, ''),
        lit_auroc_str,
        f"{our_results[m]['auroc']:.3f}",
    ])

table = ax_table.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc='center',
    loc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)

# Set column widths manually for better fit
col_widths = [0.12, 0.13, 0.10, 0.08, 0.08, 0.10, 0.12, 0.08, 0.08]
for j, w in enumerate(col_widths):
    for i in range(len(model_order) + 1):
        table[i, j].set_width(w)

# Style header
for j in range(len(headers)):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold', fontsize=10)
    table[0, j].set_edgecolor('#CCCCCC')

# Style rows
for i, m in enumerate(model_order):
    row_color = '#F5F5F5' if i % 2 == 0 else '#FFFFFF'
    for j in range(len(headers)):
        table[i+1, j].set_facecolor(row_color)
        table[i+1, j].set_edgecolor('#DDDDDD')
    # Color AUROC cells by delta
    lit_val = lit[m]['auroc']
    our_val = our_results[m]['auroc']
    if lit_val is not None:
        delta = our_val - lit_val
        color = '#C8E6C9' if delta > -0.05 else '#FFCDD2'
        table[i+1, 7].set_facecolor('#FFF9C4')
        table[i+1, 8].set_facecolor(color)

ax_table.set_title('Benchmark Comparison Details', fontsize=13, fontweight='bold', pad=15)

fig.suptitle(
    f'Literature vs Our Benchmark (V3) \u2014 {V3_N_VARS:,} variants, {V3_N_GENES} genes, '
    f'{V3_N_PATH:,} Pathogenic / {V3_N_BEN:,} Benign',
    fontsize=14, fontweight='bold', y=0.98
)

fig.savefig(f'{FIGURES_DIR}/fig_literature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: benchmark_v3/figures/fig_literature_comparison.png")

# ── Summary printout ─────────────────────────────────────────────────────────
print(f"\nBenchmark V3: {V3_N_VARS:,} variants, {V3_N_GENES} genes ({V3_N_PATH:,} P / {V3_N_BEN:,} B)")
print(f"\n{'Model':<16} {'Our AUROC':>10} {'Lit AUROC':>10} {'Delta':>8}")
print('-' * 50)
for m in model_order:
    lit_val = lit[m]['auroc']
    our_val = our_results[m]['auroc']
    delta_str = f"{our_val - lit_val:+.3f}" if lit_val is not None else "N/A"
    lit_str = f"{lit_val:.3f}" if lit_val is not None else "N/A"
    print(f"{m:<16} {our_val:>10.3f} {lit_str:>10} {delta_str:>8}")
