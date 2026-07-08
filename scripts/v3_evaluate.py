#!/usr/bin/env python3
"""Evaluate and plot all models on benchmark_v3."""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os
import warnings
warnings.filterwarnings('ignore')

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ── Load benchmark ───────────────────────────────────────────────────────────
df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
if df['ClinVar_label'].dtype in ['float64', 'int64']:
    df['label'] = df['ClinVar_label'].map({1.0: 1, 0.0: 0})
else:
    df['label'] = df['ClinVar_label'].map({'Pathogenic': 1, 'Benign': 0})
valid_vids = set(df['VariationID'])

print(f"Benchmark: {len(df)} variants, {df['GeneSymbol'].nunique()} genes")
print(f"Pathogenic: {(df['label']==1).sum()}, Benign: {(df['label']==0).sum()}")

# ── Load scores ──────────────────────────────────────────────────────────────
model_configs = {
    'ESM2-650M': {
        'file': 'benchmark_v3/results/esm2_650m_scores.csv',
        'score_col': 'ESM2_LLR',
        'negate': True,
        'type': 'protein'
    },
    'ESM1b-650M': {
        'file': 'benchmark_v3/results/esm1b_scores.csv',
        'score_col': 'ESM1b_LLR',
        'negate': True,
        'type': 'protein'
    },
    'ProtT5-XL': {
        'file': 'benchmark_v3/results/prott5_scores.csv',
        'score_col': 'ProtT5_score',
        'negate': True,
        'type': 'protein'
    },
    'SaProt-650M': {
        'file': 'benchmark_v3/results/saprot_scores.csv',
        'score_col': 'SaProt_LLR',
        'negate': True,
        'type': 'protein'
    },
    'AlphaGenome': {
        'file': 'benchmark_v3/results/alphagenome_scores.csv',
        'score_col': 'AlphaGenome_delta',
        'negate': False,
        'type': 'dna'
    },
    'HyenaDNA-150M': {
        'file': 'benchmark_v3/results/hyena_scores.csv',
        'score_col': 'HyenaDNA_LLR',
        'negate': True,
        'type': 'dna'
    },
    'NT-v2-500M': {
        'file': 'benchmark_v3/results/ntv2_scores.csv',
        'score_col': 'NTv2_delta',
        'negate': False,
        'type': 'dna'
    },
}

# ── Font setup ───────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

FIGURES_DIR = 'benchmark_v3/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Evaluate each model ──────────────────────────────────────────────────────
all_metrics = {}
all_data = {}

for model_name, config in model_configs.items():
    try:
        scores = pd.read_csv(config['file'])
    except Exception as e:
        print(f"  {model_name}: SKIPPED ({e})")
        continue

    if 'VariationID' in scores.columns:
        scores = scores[scores['VariationID'].isin(valid_vids)].copy()
    scores = scores.dropna(subset=[config['score_col']])

    if len(scores) < 50:
        print(f"  {model_name}: SKIPPED (only {len(scores)} variants)")
        continue

    score_col = config['score_col']
    y_true = scores['label'].values.astype(int)
    y_score = scores[score_col].values

    if config['negate']:
        y_score = -y_score

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    spearman_corr, spearman_p = spearmanr(y_score, y_true)
    mcc = matthews_corrcoef(y_true, (y_score > np.median(y_score)).astype(int))
    recall = recall_score(y_true, (y_score > np.median(y_score)).astype(int))

    gene_aurocs = {}
    for gene, gdata in scores.groupby('GeneSymbol'):
        if gdata['label'].nunique() == 2 and len(gdata) >= 10:
            try:
                gene_aurocs[gene] = roc_auc_score(gdata['label'].values, -gdata[score_col].values if config['negate'] else gdata[score_col].values)
            except:
                pass

    all_metrics[model_name] = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Spearman': spearman_corr,
        'MCC': mcc,
        'Recall': recall,
        'n_variants': len(scores),
        'n_genes': scores['GeneSymbol'].nunique(),
        'n_gene_aurocs': len(gene_aurocs),
        'type': config['type'],
    }
    all_data[model_name] = scores.copy()
    all_data[model_name]['score_negated'] = -y_score if config['negate'] else y_score

    print(f"  {model_name}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, Spearman={spearman_corr:.3f} ({len(scores)} variants)")

# ── Save summary ─────────────────────────────────────────────────────────────
summary = pd.DataFrame(all_metrics).T
summary.to_csv('benchmark_v3/results/benchmark_summary.csv')
print(f"\nSummary saved to benchmark_v3/results/benchmark_summary.csv")

# ── Figure 1: AUROC/AUPRC/Spearman bar chart ────────────────────────────────
model_names = list(all_metrics.keys())
colors = {'protein': '#2196F3', 'dna': '#FF9800'}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax_idx, (metric, title) in enumerate([
    ('AUROC', 'AUROC'), ('AUPRC', 'AUPRC'), ('Spearman', 'Spearman r')
]):
    ax = axes[ax_idx]
    vals = [all_metrics[m][metric] for m in model_names]
    types = [all_metrics[m]['type'] for m in model_names]
    bar_colors = [colors[t] for t in types]
    bars = ax.barh(model_names[::-1], vals[::-1], color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
    ax.set_xlabel(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1 if metric != 'Spearman' else 1)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelsize=11)

fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig1_auroc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 1 saved")

# ── Figure 2: ROC curves ────────────────────────────────────────────────────
from sklearn.metrics import roc_curve
fig, ax = plt.subplots(figsize=(8, 7))
line_styles = {'protein': '-', 'dna': '--'}

for model_name in model_names:
    scores = all_data[model_name]
    y_true = scores['label'].values.astype(int)
    y_score = scores['score_negated'].values
    fpr, tpr, _ = roc_curve(y_true, y_score)
    style = line_styles[all_metrics[model_name]['type']]
    ax.plot(fpr, tpr, style, linewidth=2,
            label=f"{model_name} ({all_metrics[model_name]['AUROC']:.3f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig2_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 2 saved")

# ── Figure 3: Score distributions ───────────────────────────────────────────
n_models = len(model_names)
fig, axes = plt.subplots(n_models, 1, figsize=(10, 3*n_models))
if n_models == 1:
    axes = [axes]

for i, model_name in enumerate(model_names):
    scores = all_data[model_name]
    path_scores = scores[scores['label']==1]['score_negated'].values
    ben_scores = scores[scores['label']==0]['score_negated'].values
    ax = axes[i]
    ax.hist(path_scores, bins=50, alpha=0.6, color='#E53935', label='Pathogenic', density=True)
    ax.hist(ben_scores, bins=50, alpha=0.6, color='#1E88E5', label='Benign', density=True)
    ax.set_ylabel(model_name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

axes[-1].set_xlabel('Score (negated for protein models)', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig3_score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 3 saved")

# ── Figure 4: Per-gene AUROC heatmap ────────────────────────────────────────
gene_auroc_dict = {}
for model_name in model_names:
    scores = all_data[model_name]
    gene_aurocs = {}
    for gene, gdata in scores.groupby('GeneSymbol'):
        if gdata['label'].nunique() == 2 and len(gdata) >= 10:
            try:
                y = gdata['label'].values.astype(int)
                s = gdata['score_negated'].values
                gene_aurocs[gene] = roc_auc_score(y, s)
            except:
                pass
    if gene_aurocs:
        gene_auroc_dict[model_name] = gene_aurocs

all_genes_sets = [set(v.keys()) for v in gene_auroc_dict.values()]
if all_genes_sets:
    common_genes = sorted(set.intersection(*all_genes_sets))
    if len(common_genes) > 5:
        data_matrix = np.zeros((len(model_names), len(common_genes)))
        for i, mn in enumerate(model_names):
            if mn in gene_auroc_dict:
                for j, g in enumerate(common_genes):
                    data_matrix[i, j] = gene_auroc_dict[mn].get(g, 0.5)

        fig, ax = plt.subplots(figsize=(max(16, len(common_genes)*0.12), max(5, len(model_names)*0.6)))
        im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlBu_r', vmin=0.3, vmax=0.9)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xticks(range(len(common_genes)))
        ax.set_xticklabels(common_genes, rotation=90, fontsize=6)
        plt.colorbar(im, ax=ax, label='AUROC')
        ax.set_title(f'Per-Gene AUROC ({len(common_genes)} common genes)', fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{FIGURES_DIR}/fig4_per_gene_auroc.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Figure 4 saved ({len(common_genes)} common genes)")
    else:
        print(f"Figure 4 skipped (only {len(common_genes)} common genes)")
else:
    print("Figure 4 skipped (no common genes)")

# ── Figure 5: Summary table ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, max(3, len(model_names)*0.5 + 1)))
ax.axis('off')

cols = ['AUROC', 'AUPRC', 'Spearman', 'MCC', 'Recall', 'n_variants']
table_data = []
for mn in model_names:
    row = [f'{all_metrics[mn][c]:.3f}' if isinstance(all_metrics[mn][c], float) else str(all_metrics[mn][c]) for c in cols]
    table_data.append(row)

table = ax.table(cellText=table_data, rowLabels=model_names, colLabels=cols,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for j, col in enumerate(cols):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold')
for i, mn in enumerate(model_names):
    color = '#E3F2FD' if all_metrics[mn]['type'] == 'protein' else '#FFF3E0'
    for j in range(len(cols)):
        table[i+1, j].set_facecolor(color)

ax.set_title('Benchmark V3 Summary', fontsize=14, fontweight='bold', pad=20)
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig5_summary_table.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 5 saved")

# ── Figure 6: Spearman scatter ──────────────────────────────────────────────
fig, axes = plt.subplots(1, len(model_names), figsize=(4*len(model_names), 4))
if len(model_names) == 1:
    axes = [axes]

for i, model_name in enumerate(model_names):
    scores = all_data[model_name]
    y_true = scores['label'].values.astype(int)
    y_score = scores['score_negated'].values
    jitter = np.random.normal(0, 0.02, len(y_true))
    ax = axes[i]
    ax.scatter(y_score, y_true + jitter, alpha=0.3, s=10, c=[('#E53935' if t else '#1E88E5') for t in y_true])
    ax.set_title(f"{model_name}\nr={all_metrics[model_name]['Spearman']:.3f}", fontsize=10, fontweight='bold')
    ax.set_ylim(-0.5, 1.5)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Benign', 'Pathogenic'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)

axes[0].set_ylabel('Label', fontsize=11, fontweight='bold')
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig6_spearman_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 6 saved")

# ── Figure 7: Literature comparison ─────────────────────────────────────────
lit_data = {
    'ESM2-650M': 0.880,
    'ESM1b-650M': 0.874,
    'ProtT5-XL': 0.610,
    'SaProt-650M': 0.850,
    'AlphaGenome': 0.770,
    'HyenaDNA-150M': 0.550,
    'NT-v2-500M': 0.630,
}

fig, ax = plt.subplots(figsize=(10, 6))
models_with_lit = [m for m in model_names if m in lit_data]
x = np.arange(len(models_with_lit))
width = 0.35

our_vals = [all_metrics[m]['AUROC'] for m in models_with_lit]
lit_vals = [lit_data[m] for m in models_with_lit]

bars1 = ax.bar(x - width/2, our_vals, width, label='This study', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + width/2, lit_vals, width, label='Literature', color='#FF9800', alpha=0.8)

ax.set_ylabel('AUROC', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_with_lit, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11)
ax.set_ylim(0.4, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar, val in zip(bars1, our_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)
for bar, val in zip(bars2, lit_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig7_literature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 7 saved")

# ── Figure 10: Scoring methods ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

methods = {
    'ESM2-650M': ('Log Likelihood Ratio', 'P(mut|context)/P(wt|context)', 'Protein LM'),
    'ESM1b-650M': ('Log Likelihood Ratio', 'P(mut|context)/P(wt|context)', 'Protein LM'),
    'ProtT5-XL': ('Cosine Similarity', 'sim(wt_repr, mut_repr)', 'Protein LM'),
    'SaProt-650M': ('Log Likelihood Ratio', 'P(mut|AA+3Di)/P(wt|AA+3Di)', 'Protein LM'),
    'AlphaGenome': ('Delta Score', '|ref_output - alt_output|', 'DNA LM'),
    'HyenaDNA-150M': ('Log Likelihood Ratio', 'P(alt|DNA)/P(ref|DNA)', 'DNA LM'),
    'NT-v2-500M': ('Hidden State Delta', 'mean(|wt_h - mut_h|)', 'DNA LM'),
}

table_data = []
for mn in model_names:
    if mn in methods:
        method, formula, mtype = methods[mn]
        table_data.append([mn, mtype, method, formula])

table = ax.table(cellText=table_data,
                 colLabels=['Model', 'Type', 'Method', 'Formula'],
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

for j in range(4):
    table[0, j].set_facecolor('#E3F2FD')
    table[0, j].set_text_props(fontweight='bold')

ax.set_title('Scoring Methods', fontsize=14, fontweight='bold', pad=20)
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig10_scoring_methods.png', dpi=300, bbox_inches='tight')
plt.close()
print("Figure 10 saved")

print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")
