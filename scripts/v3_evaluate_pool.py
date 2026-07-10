#!/usr/bin/env python3
"""Evaluate max pool + cosine scores vs original methods on benchmark_v3."""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
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

# ── Model configs ────────────────────────────────────────────────────────────
# Original scores (use negate=True for higher=pathogenic)
orig_models = {
    'ESM2-650M':    ('benchmark_v3/results/esm2_650m_scores.csv',  'ESM2_LLR',           True,  'protein'),
    'ESM1b-650M':   ('benchmark_v3/results/esm1b_scores.csv',      'ESM1b_LLR',          True,  'protein'),
    'SaProt-650M':  ('benchmark_v3/results/saprot_scores.csv',     'SaProt_LLR',         True,  'protein'),
    'ProtT5-XL':    ('benchmark_v3/results/prott5_scores.csv',     'ProtT5_score',       True,  'protein'),
    'NT-v2-500M':   ('benchmark_v3/results/ntv2_scores.csv',       'NTv2_delta',         False, 'dna'),
    'HyenaDNA-150M':('benchmark_v3/results/hyena_scores.csv',      'HyenaDNA_LLR',       True,  'dna'),
}

# Pooled scores (negate=False for higher=pathogenic since cosine_sim is directional)
pool_models = {
    'ESM2-650M':    ('benchmark_v3/results/esm2_pool_scores.csv',   'ESM2_pool',          False, 'protein'),
    'ESM1b-650M':   ('benchmark_v3/results/esm1b_pool_scores.csv',  'ESM1b_pool',         False, 'protein'),
    'SaProt-650M':  ('benchmark_v3/results/saprot_pool_scores.csv', 'SaProt_pool',        False, 'protein'),
    'ProtT5-XL':    ('benchmark_v3/results/prott5_pool_scores.csv', 'ProtT5_pool',        False, 'protein'),
    'NT-v2-500M':   ('benchmark_v3/results/ntv2_pool_scores.csv',   'NTv2_pool',          False, 'dna'),
    'HyenaDNA-150M':('benchmark_v3/results/hyena_pool_scores.csv',  'Hyena_pool',         False, 'dna'),
}

# ── Font setup ───────────────────────────────────────────────────────────────
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

FIGURES_DIR = 'benchmark_v3/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Evaluate function ────────────────────────────────────────────────────────
def evaluate_scores(df_valid, scores_dict, score_col, negate):
    merged = df_valid.merge(scores_dict[['VariationID', score_col]], on='VariationID', how='left')
    merged = merged.dropna(subset=[score_col])
    if len(merged) < 50:
        return None
    y_true = merged['label'].values.astype(int)
    y_score = merged[score_col].values
    if negate:
        y_score = -y_score
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    spearman, _ = spearmanr(y_score, y_true)
    mcc = matthews_corrcoef(y_true, (y_score > np.median(y_score)).astype(int))
    recall = recall_score(y_true, (y_score > np.median(y_score)).astype(int))
    return {'AUROC': auroc, 'AUPRC': auprc, 'Spearman': spearman, 'MCC': mcc, 'Recall': recall, 'n': len(merged)}

# ── Compute metrics ──────────────────────────────────────────────────────────
results_orig = {}
results_pool = {}

for model_name, (fpath, score_col, negate, mtype) in orig_models.items():
    try:
        scores = pd.read_csv(fpath)
        scores = scores[scores['VariationID'].isin(valid_vids)]
        results = evaluate_scores(df, scores, score_col, negate)
        if results:
            results['type'] = mtype
            results_orig[model_name] = results
            print(f"  {model_name:15s} (orig):  Spearman={results['Spearman']:+.4f}  AUROC={results['AUROC']:.4f}  ({results['n']} vars)")
    except Exception as e:
        print(f"  {model_name:15s} (orig): SKIPPED ({e})")

for model_name, (fpath, score_col, negate, mtype) in pool_models.items():
    try:
        scores = pd.read_csv(fpath)
        scores = scores[scores['VariationID'].isin(valid_vids)]
        results = evaluate_scores(df, scores, score_col, negate)
        if results:
            results['type'] = mtype
            results_pool[model_name] = results
            print(f"  {model_name:15s} (pool): Spearman={results['Spearman']:+.4f}  AUROC={results['AUROC']:.4f}  ({results['n']} vars)")
    except Exception as e:
        print(f"  {model_name:15s} (pool): SKIPPED ({e})")

# ── Figure: Original vs Pool Spearman comparison ────────────────────────────
model_order = [m for m in orig_models if m in results_orig and m in results_pool]
x = np.arange(len(model_order))
bar_height = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Spearman comparison
ax = axes[0]
orig_spearman = [results_orig[m]['Spearman'] for m in model_order]
pool_spearman = [results_pool[m]['Spearman'] for m in model_order]

bars_orig = ax.barh(x + bar_height/2, orig_spearman, bar_height, label='Original (LLR/delta)', color='#2196F3', edgecolor='white')
bars_pool = ax.barh(x - bar_height/2, pool_spearman, bar_height, label='Max Pool + Cosine', color='#FF9800', edgecolor='white')

for bar, val in zip(bars_orig, orig_spearman):
    ax.text(val + 0.005 if val >= 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=10, fontweight='bold', color='#2196F3')
for bar, val in zip(bars_pool, pool_spearman):
    ax.text(val + 0.005 if val >= 0 else val - 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left' if val >= 0 else 'right', va='center', fontsize=10, fontweight='bold', color='#FF9800')

ax.set_yticks(x)
ax.set_yticklabels(model_order, fontsize=11, fontweight='bold')
ax.set_xlabel('Spearman Correlation', fontsize=13, fontweight='bold')
ax.set_title('Spearman: Original vs Max Pool', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# AUROC comparison
ax = axes[1]
orig_auroc = [results_orig[m]['AUROC'] for m in model_order]
pool_auroc = [results_pool[m]['AUROC'] for m in model_order]

bars_orig = ax.barh(x + bar_height/2, orig_auroc, bar_height, label='Original', color='#2196F3', edgecolor='white')
bars_pool = ax.barh(x - bar_height/2, pool_auroc, bar_height, label='Max Pool', color='#FF9800', edgecolor='white')

for bar, val in zip(bars_orig, orig_auroc):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold', color='#2196F3')
for bar, val in zip(bars_pool, pool_auroc):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', ha='left', va='center', fontsize=10, fontweight='bold', color='#FF9800')

ax.set_yticks(x)
ax.set_yticklabels(model_order, fontsize=11, fontweight='bold')
ax.set_xlabel('AUROC', fontsize=13, fontweight='bold')
ax.set_title('AUROC: Original vs Max Pool', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim(0.4, 1.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, linestyle='--')

fig.suptitle('Original Scoring vs Max Pool + Cosine Similarity', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
fig.savefig(f'{FIGURES_DIR}/fig_pool_vs_orig.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: benchmark_v3/figures/fig_pool_vs_orig.png")

# ── Summary table ────────────────────────────────────────────────────────────
print(f"\n{'Model':<16} {'Orig Spearman':>13} {'Pool Spearman':>13} {'Delta':>8}  {'Orig AUROC':>10} {'Pool AUROC':>10}")
print('-' * 80)
for m in model_order:
    o = results_orig[m]
    p = results_pool[m]
    delta = p['Spearman'] - o['Spearman']
    print(f"{m:<16} {o['Spearman']:>+13.4f} {p['Spearman']:>+13.4f} {delta:>+8.4f}  {o['AUROC']:>10.4f} {p['AUROC']:>10.4f}")

print(f"\nEnd: {time.strftime('%Y-%m-%d %H:%M:%S')}")
