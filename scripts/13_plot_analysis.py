#!/usr/bin/env python3
"""Comprehensive analysis and model comparison plots for ClinVar benchmark.

Generates publication-ready figures:
  Part 1: Data exploration (fig1-4)
  Part 2: Model performance (fig5-9)
  Part 3: Summary table (fig10)
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ── Publication-ready style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.2,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

DATA_DIR = 'benchmark_200/data'
RESULTS_DIR = 'benchmark_200/results'
FIG_DIR = 'benchmark_200/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Color palette (colorblind-friendly)
COLORS = {
    'ESM2-650M': '#0072B2',
    'ProtT5-XL': '#E69F00',
    'AlphaGenome': '#009E73',
    'NT-v2': '#CC79A7',
    'SaProt': '#D55E00',
    'HyenaDNA': '#F0E442',
    'Pathogenic': '#D55E00',
    'Benign': '#56B4E9',
}


def load_data():
    """Load all benchmark data."""
    full = pd.read_csv(os.path.join(DATA_DIR, 'clinvar_200_full.csv'))
    genes = pd.read_csv(os.path.join(DATA_DIR, 'selected_200_genes.csv'))

    esm2 = pd.read_csv(os.path.join(RESULTS_DIR, 'esm2_scores.csv'))
    prot = pd.read_csv(os.path.join(RESULTS_DIR, 'prott5_scores.csv'))
    ag = pd.read_csv(os.path.join(RESULTS_DIR, 'alphagenome_scores.csv'))
    saprot = pd.read_csv(os.path.join(RESULTS_DIR, 'saprot_scores.csv'))
    hyena = pd.read_csv(os.path.join(RESULTS_DIR, 'hyena_scores.csv'))
    ntv2 = pd.read_csv(os.path.join(RESULTS_DIR, 'ntv2_scores.csv'))

    # Negate LLR for protein LMs (more negative = more pathogenic → negate)
    esm2['score'] = -esm2['ESM2_LLR']
    prot['score'] = -prot['ESM2_35M_LLR']
    ag['score'] = ag['AlphaGenome_delta']
    saprot['score'] = -saprot['SaProt_LLR']
    hyena['score'] = hyena['HyenaDNA_LLR']
    ntv2['score'] = ntv2['NTv2_delta']

    models = {
        'ESM2-650M': esm2,
        'ProtT5-XL': prot,
        'AlphaGenome': ag,
        'SaProt': saprot,
        'HyenaDNA': hyena,
        'NT-v2': ntv2,
    }
    return full, genes, models


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: DATA EXPLORATION
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_variant_distribution(full, genes):
    """Stacked bar chart: pathogenic vs benign per gene (top 30)."""
    top30 = genes.nlargest(30, 'total')

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(top30))
    bars_p = ax.bar(x, top30['pathogenic_count'], 0.7, label='Pathogenic',
                    color=COLORS['Pathogenic'], edgecolor='white', linewidth=0.3)
    bars_b = ax.bar(x, top30['benign_count'], 0.7, bottom=top30['pathogenic_count'],
                    label='Benign', color=COLORS['Benign'], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(top30['GeneSymbol'], rotation=45, ha='right')
    ax.set_ylabel('Number of Variants')
    ax.set_title('ClinVar Variant Distribution by Gene (Top 30)')
    ax.legend(frameon=False, loc='upper right')
    ax.set_xlim(-0.5, len(top30) - 0.5)

    # Add total count on top
    for i, (_, row) in enumerate(top30.iterrows()):
        ax.text(i, row['total'] + 5, str(int(row['total'])), ha='center', va='bottom',
                fontsize=6, color='#555555')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_variant_distribution.png'))
    plt.close(fig)
    print('  Saved fig1_variant_distribution.png')


def fig2_amino_acid_substitution(full):
    """Heatmap of ref_aa → alt_aa substitution counts."""
    # Use 1-letter codes
    aa_3to1 = {
        'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
        'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
        'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
        'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    }
    aa_order = list('ARNDCEQGHILKMFPSTWYV')

    df = full.copy()
    df['ref_1'] = df['ref_aa'].map(aa_3to1)
    df['alt_1'] = df['alt_aa'].map(aa_3to1)
    df = df.dropna(subset=['ref_1', 'alt_1'])

    # Build substitution matrix
    matrix = pd.DataFrame(0, index=aa_order, columns=aa_order)
    for _, row in df.iterrows():
        r, a = row['ref_1'], row['alt_1']
        if r in matrix.index and a in matrix.columns:
            matrix.loc[r, a] += 1

    # Mask diagonal
    np.fill_diagonal(matrix.values, 0)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    mask = np.eye(len(aa_order), dtype=bool)
    sns.heatmap(matrix, mask=mask, cmap='YlOrRd', annot=True, fmt='d',
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Count', 'shrink': 0.7},
                annot_kws={'size': 6})
    ax.set_xlabel('Alternate Amino Acid')
    ax.set_ylabel('Reference Amino Acid')
    ax.set_title('Amino Acid Substitution Matrix')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_amino_acid_substitution.png'))
    plt.close(fig)
    print('  Saved fig2_amino_acid_substitution.png')


def fig3_protein_position_distribution(full):
    """Histogram of aa_position for pathogenic vs benign."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    path = full[full['ClinVar_label'] == 1]['aa_position'].dropna()
    ben = full[full['ClinVar_label'] == 0]['aa_position'].dropna()

    bins = np.linspace(0, min(path.max(), ben.max(), 2500), 50)
    ax.hist(path, bins=bins, alpha=0.6, label=f'Pathogenic (n={len(path)})',
            color=COLORS['Pathogenic'], edgecolor='white', linewidth=0.3, density=True)
    ax.hist(ben, bins=bins, alpha=0.6, label=f'Benign (n={len(ben)})',
            color=COLORS['Benign'], edgecolor='white', linewidth=0.3, density=True)

    ax.set_xlabel('Amino Acid Position')
    ax.set_ylabel('Density')
    ax.set_title('Variant Position Distribution along Protein')
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_protein_position_distribution.png'))
    plt.close(fig)
    print('  Saved fig3_protein_position_distribution.png')


def fig4_gene_balance(genes):
    """Scatter: pathogenic_count vs benign_count per gene."""
    fig, ax = plt.subplots(figsize=(5, 4))

    ax.scatter(genes['benign_count'], genes['pathogenic_count'],
               s=20, alpha=0.6, edgecolors='white', linewidth=0.3,
               color='#4C72B0')

    # Balance diagonal
    max_val = max(genes['pathogenic_count'].max(), genes['benign_count'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], '--', color='gray', linewidth=0.8, alpha=0.5, label='Balanced')

    # Label outliers
    for _, row in genes.nlargest(5, 'total').iterrows():
        ax.annotate(row['GeneSymbol'],
                    (row['benign_count'], row['pathogenic_count']),
                    fontsize=6, ha='left', va='bottom', color='#333')

    ax.set_xlabel('Benign Variants')
    ax.set_ylabel('Pathogenic Variants')
    ax.set_title('Pathogenic vs Benign Variant Counts per Gene')
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlim(-5, max_val)
    ax.set_ylim(-5, max_val)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_gene_balance.png'))
    plt.close(fig)
    print('  Saved fig4_gene_balance.png')


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_auroc_auprc(y_true, scores):
    """Compute AUROC and AUPRC."""
    from sklearn.metrics import average_precision_score
    mask = ~np.isnan(scores)
    y_true = y_true[mask]
    scores = scores[mask]
    if len(y_true) < 10:
        return np.nan, np.nan
    return roc_auc_score(y_true, scores), average_precision_score(y_true, scores)


def fig5_auroc_comparison(models, full):
    """Grouped bar chart comparing AUROC, AUPRC, and Spearman r across models."""
    from scipy import stats as sp_stats

    labels = full[['VariationID', 'ClinVar_label']].drop_duplicates(subset='VariationID')

    results = []
    for name, df in models.items():
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')
        if len(merged) > 10:
            auroc, auprc = _compute_auroc_auprc(merged['ClinVar_label'].values, merged['score'].values)
            r, _ = sp_stats.spearmanr(merged['score'], merged['ClinVar_label'])
            results.append({'Model': name, 'AUROC': auroc, 'AUPRC': auprc, 'Spearman_r': r, 'N': len(merged)})

    res_df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(res_df))
    w = 0.25

    bars1 = ax.bar(x - w, res_df['AUROC'], w, label='AUROC', color='#0072B2', edgecolor='white', linewidth=0.3)
    bars2 = ax.bar(x, res_df['AUPRC'], w, label='AUPRC', color='#E69F00', edgecolor='white', linewidth=0.3)
    bars3 = ax.bar(x + w, res_df['Spearman_r'], w, label='Spearman r', color='#009E73', edgecolor='white', linewidth=0.3)

    # Add value labels
    for i, row in res_df.iterrows():
        ax.text(i - w, row['AUROC'] + 0.01, f"{row['AUROC']:.3f}", ha='center', va='bottom', fontsize=6, fontweight='bold')
        ax.text(i, row['AUPRC'] + 0.01, f"{row['AUPRC']:.3f}", ha='center', va='bottom', fontsize=6, fontweight='bold')
        offset = 0.01 if row['Spearman_r'] >= 0 else -0.03
        va = 'bottom' if row['Spearman_r'] >= 0 else 'top'
        ax.text(i + w, row['Spearman_r'] + offset, f"{row['Spearman_r']:.3f}", ha='center', va=va, fontsize=6, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(res_df['Model'], rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_ylim(-0.35, 1.08)
    ax.set_title('Model Performance Comparison (ClinVar 200-Gene Benchmark)', fontsize=11, fontweight='bold')
    ax.legend(frameon=False, loc='lower left', ncol=3, fontsize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.4, label='_nolegend_')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_auroc_comparison.png'))
    plt.close(fig)
    print('  Saved fig5_auroc_comparison.png')


def fig6_roc_curves(models, full):
    """Overlaid ROC curves for all models."""
    labels = full[['VariationID', 'ClinVar_label']].drop_duplicates(subset='VariationID')

    fig, ax = plt.subplots(figsize=(4.5, 4))

    for name, df in models.items():
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')
        if len(merged) > 10:
            y_true = merged['ClinVar_label'].values
            y_score = merged['score'].values
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auroc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{name} (AUC={auroc:.3f})',
                    color=COLORS.get(name, '#333'), linewidth=1.5)

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_roc_curves.png'))
    plt.close(fig)
    print('  Saved fig6_roc_curves.png')


def fig7_score_distributions(models, full):
    """2x2 grid: score distributions for benign vs pathogenic per model."""
    labels = full[['VariationID', 'ClinVar_label']].drop_duplicates(subset='VariationID')
    model_names = list(models.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        df = models[name]
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')

        if len(merged) < 10:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        ben = merged[merged['ClinVar_label'] == 0]['score']
        pat = merged[merged['ClinVar_label'] == 1]['score']

        bins = np.linspace(
            min(merged['score'].quantile(0.01), merged['score'].quantile(0.01)),
            max(merged['score'].quantile(0.99), merged['score'].quantile(0.99)),
            50
        )
        ax.hist(ben, bins=bins, alpha=0.6, label='Benign', color=COLORS['Benign'],
                edgecolor='white', linewidth=0.3, density=True)
        ax.hist(pat, bins=bins, alpha=0.6, label='Pathogenic', color=COLORS['Pathogenic'],
                edgecolor='white', linewidth=0.3, density=True)

        ax.set_title(name)
        ax.set_xlabel('Score')
        ax.legend(frameon=False, fontsize=7)

    axes[0].set_ylabel('Density')
    fig.suptitle('Score Distributions by Clinical Significance', fontsize=11, y=1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_score_distributions.png'))
    plt.close(fig)
    print('  Saved fig7_score_distributions.png')


def fig8_per_gene_auroc(models, full):
    """Heatmap of per-gene AUROC for each model."""
    labels = full[['VariationID', 'ClinVar_label', 'GeneSymbol']].drop_duplicates(subset='VariationID')

    # Get genes with at least 2 variants per class
    gene_counts = labels.groupby(['GeneSymbol', 'ClinVar_label']).size().unstack(fill_value=0)
    valid_genes = gene_counts[(gene_counts.get(0, 0) >= 2) & (gene_counts.get(1, 0) >= 2)].index

    # Compute per-gene AUROC for each model
    auroc_data = {}
    for name, df in models.items():
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')
        gene_auroc = {}
        for gene in valid_genes:
            g = merged[merged['GeneSymbol'] == gene]
            if len(g) >= 4 and g['ClinVar_label'].nunique() == 2:
                try:
                    gene_auroc[gene] = roc_auc_score(g['ClinVar_label'], g['score'])
                except:
                    pass
        auroc_data[name] = gene_auroc

    # Build matrix — keep NaN for genes without data (don't dropna)
    matrix = pd.DataFrame(auroc_data)
    # Only keep genes where at least 3 models have data
    matrix = matrix.dropna(thresh=3)
    # Sort by mean AUROC
    matrix = matrix.loc[matrix.mean(axis=1).sort_values(ascending=False).index]

    if matrix.empty:
        print('  Skipped fig8: insufficient per-gene data')
        return

    n_genes = len(matrix)
    fig_height = max(6, n_genes * 0.35)
    fig, ax = plt.subplots(figsize=(5 + len(models) * 0.8, fig_height))
    sns.heatmap(matrix, cmap='RdYlGn', center=0.7, vmin=0.4, vmax=1.0,
                annot=True, fmt='.2f', linewidths=0.4, linecolor='white',
                ax=ax, cbar_kws={'label': 'AUROC', 'shrink': 0.5},
                annot_kws={'size': 6}, mask=matrix.isna())
    ax.set_title(f'Per-Gene AUROC by Model ({n_genes} genes)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Gene')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='right')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig8_per_gene_auroc.png'))
    plt.close(fig)
    print('  Saved fig8_per_gene_auroc.png')


def fig9_spearman_scatter(models, full):
    """Scatter: ClinVar_label vs model score for each model (with Spearman r)."""
    labels = full[['VariationID', 'ClinVar_label']].drop_duplicates(subset='VariationID')
    model_names = list(models.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        df = models[name]
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')

        if len(merged) < 10:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Add jitter
        jitter = np.random.uniform(-0.1, 0.1, len(merged))
        y = merged['ClinVar_label'].values + jitter
        x = merged['score'].values

        ax.scatter(x[y <= 0.5], y[y <= 0.5], s=5, alpha=0.3, color=COLORS['Benign'], label='Benign')
        ax.scatter(x[y > 0.5], y[y > 0.5], s=5, alpha=0.3, color=COLORS['Pathogenic'], label='Pathogenic')

        r, p = stats.spearmanr(merged['score'], merged['ClinVar_label'])
        ax.text(0.05, 0.95, f'Spearman r={r:.3f}\np={p:.1e}',
                transform=ax.transAxes, fontsize=7, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_title(name)
        ax.set_xlabel('Score')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Benign', 'Pathogenic'])

    axes[0].set_ylabel('Clinical Significance')
    fig.suptitle('Score vs Clinical Significance', fontsize=11, y=1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig9_spearman_scatter.png'))
    plt.close(fig)
    print('  Saved fig9_spearman_scatter.png')


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def fig10_summary_table(models, full):
    """Table image of benchmark metrics."""
    from sklearn.metrics import matthews_corrcoef, recall_score

    labels = full[['VariationID', 'ClinVar_label']].drop_duplicates(subset='VariationID')

    rows = []
    for name, df in models.items():
        merged = labels.merge(df[['VariationID', 'score']], on='VariationID', how='inner')
        if len(merged) < 10:
            continue
        y = merged['ClinVar_label'].values
        s = merged['score'].values

        auroc = roc_auc_score(y, s)
        auprc = auc(*roc_curve(y, s)[:2])
        r, p = stats.spearmanr(s, y)
        preds = (s > 0).astype(int)
        mcc = matthews_corrcoef(y, preds)
        recall = recall_score(y, preds)

        rows.append({
            'Model': name,
            'N': len(merged),
            'Path': int(y.sum()),
            'Ben': int((1 - y).sum()),
            'AUROC': f'{auroc:.4f}',
            'AUPRC': f'{auprc:.4f}',
            'Spearman r': f'{r:.3f}',
            'MCC': f'{mcc:.3f}',
            'Recall': f'{recall:.3f}',
        })

    res_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.5 * len(res_df)))
    ax.axis('off')
    ax.set_title('ClinVar Benchmark Results — 200-Gene Subset', fontsize=12, fontweight='bold', pad=15)

    table = ax.table(
        cellText=res_df.values,
        colLabels=res_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for j in range(len(res_df.columns)):
        cell = table[0, j]
        cell.set_facecolor('#2C3E50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(res_df)):
        color = '#F8F9FA' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(res_df.columns)):
            table[i + 1, j].set_facecolor(color)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig10_summary_table.png'))
    plt.close(fig)
    print('  Saved fig10_summary_table.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('Loading data...', flush=True)
    full, genes, models = load_data()
    print(f'  Full: {len(full)} variants, {full["GeneSymbol"].nunique()} genes', flush=True)
    print(f'  Models: {list(models.keys())}', flush=True)

    print('\n── Part 1: Data Exploration ──', flush=True)
    fig1_variant_distribution(full, genes)
    fig2_amino_acid_substitution(full)
    fig3_protein_position_distribution(full)
    fig4_gene_balance(genes)

    print('\n── Part 2: Model Performance ──', flush=True)
    fig5_auroc_comparison(models, full)
    fig6_roc_curves(models, full)
    fig7_score_distributions(models, full)
    fig8_per_gene_auroc(models, full)
    fig9_spearman_scatter(models, full)

    print('\n── Part 3: Summary ──', flush=True)
    fig10_summary_table(models, full)

    print(f'\nDone! All figures saved to {FIG_DIR}/', flush=True)


if __name__ == '__main__':
    main()
