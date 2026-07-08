#!/usr/bin/env python3
"""Evaluate model performance on ClinVar variant pathogenicity prediction.

Computes metrics: AUROC, AUPRC, Spearman correlation, MCC, Recall.
Generates summary table and comparison plots.
"""

import pandas as pd
import numpy as np
import os
import sys
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, recall_score

def compute_metrics(y_true, scores, model_name):
    """Compute all metrics for a set of scores."""
    y_true = np.array(y_true)
    scores = np.array(scores)

    # Remove NaN
    mask = ~np.isnan(scores)
    y_true = y_true[mask]
    scores = scores[mask]

    if len(y_true) < 10:
        return None

    # AUROC
    auroc = roc_auc_score(y_true, scores)

    # AUPRC
    auprc = average_precision_score(y_true, scores)

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(y_true, scores)

    # MCC (binarize at 0 threshold for LLR-based scores)
    preds = (scores > 0).astype(int)
    mcc = matthews_corrcoef(y_true, preds)

    # Recall (sensitivity for pathogenic)
    recall = recall_score(y_true, preds)

    return {
        'Model': model_name,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Spearman_r': spearman_corr,
        'Spearman_p': spearman_p,
        'MCC': mcc,
        'Recall': recall,
        'N_variants': len(y_true),
        'N_pathogenic': int(y_true.sum()),
        'N_benign': int((1 - y_true).sum()),
    }

def main():
    data_dir = "benchmark_200/data"
    results_dir = "benchmark_200/results"

    # Load variant labels
    print("Loading variant labels...", flush=True)
    full_data = pd.read_csv(os.path.join(data_dir, "clinvar_200_full.csv"))
    labels = full_data[['VariationID', 'ClinVar_label']].copy()
    labels = labels.drop_duplicates(subset='VariationID')
    print(f"  Total unique variants: {len(labels)}", flush=True)

    all_metrics = []

    # ESM2-650M scores
    esm2_path = os.path.join(results_dir, "esm2_scores.csv")
    if os.path.exists(esm2_path):
        print("\nProcessing ESM2-650M scores...", flush=True)
        esm2 = pd.read_csv(esm2_path)
        merged = labels.merge(esm2[['VariationID', 'ESM2_LLR']], on='VariationID', how='inner')
        if len(merged) > 0:
            # Negate LLR: more negative = more pathogenic, so negate for standard convention
            metrics = compute_metrics(merged['ClinVar_label'], -merged['ESM2_LLR'], 'ESM2-650M')
            if metrics:
                all_metrics.append(metrics)
                print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  ESM2 scores not found: {esm2_path}", flush=True)

    # ProtT5 / ESM2-35M scores
    prott5_path = os.path.join(results_dir, "prott5_scores.csv")
    if os.path.exists(prott5_path):
        print("\nProcessing ProtT5/ESM2-35M scores...", flush=True)
        prott5 = pd.read_csv(prott5_path)
        # Check which column exists
        score_col = 'ESM2_35M_LLR' if 'ESM2_35M_LLR' in prott5.columns else 'ProtT5_LLR'
        if score_col in prott5.columns:
            merged = labels.merge(prott5[['VariationID', score_col]], on='VariationID', how='inner')
            if len(merged) > 0:
                # Negate LLR: more negative = more pathogenic
                metrics = compute_metrics(merged['ClinVar_label'], -merged[score_col], 'ProtT5-XL')
                if metrics:
                    all_metrics.append(metrics)
                    print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  ProtT5 scores not found: {prott5_path}", flush=True)

    # AlphaGenome scores
    ag_path = os.path.join(results_dir, "alphagenome_scores.csv")
    if os.path.exists(ag_path):
        print("\nProcessing AlphaGenome scores...", flush=True)
        ag = pd.read_csv(ag_path)
        merged = labels.merge(ag[['VariationID', 'AlphaGenome_delta']], on='VariationID', how='inner')
        if len(merged) > 0:
            # AlphaGenome delta: higher = more different from reference
            # For pathogenic variants, delta should be higher
            metrics = compute_metrics(merged['ClinVar_label'], merged['AlphaGenome_delta'], 'AlphaGenome')
            if metrics:
                all_metrics.append(metrics)
                print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  AlphaGenome scores not found: {ag_path}", flush=True)

    # NT-v2 scores
    ntv2_path = os.path.join(results_dir, "ntv2_scores.csv")
    if os.path.exists(ntv2_path):
        print("\nProcessing NT-v2 scores...", flush=True)
        ntv2 = pd.read_csv(ntv2_path)
        score_col = [c for c in ntv2.columns if 'LLR' in c or 'delta' in c.lower() or 'score' in c.lower()][0]
        merged = labels.merge(ntv2[['VariationID', score_col]], on='VariationID', how='inner')
        if len(merged) > 0:
            metrics = compute_metrics(merged['ClinVar_label'], merged[score_col], 'NT-v2')
            if metrics:
                all_metrics.append(metrics)
                print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  NT-v2 scores not found: {ntv2_path}", flush=True)

    # HyenaDNA scores
    hyena_path = os.path.join(results_dir, "hyena_scores.csv")
    if os.path.exists(hyena_path):
        print("\nProcessing HyenaDNA scores...", flush=True)
        hyena = pd.read_csv(hyena_path)
        score_col = [c for c in hyena.columns if 'LLR' in c or 'delta' in c.lower() or 'score' in c.lower()][0]
        merged = labels.merge(hyena[['VariationID', score_col]], on='VariationID', how='inner')
        if len(merged) > 0:
            metrics = compute_metrics(merged['ClinVar_label'], merged[score_col], 'HyenaDNA')
            if metrics:
                all_metrics.append(metrics)
                print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  HyenaDNA scores not found: {hyena_path}", flush=True)

    # SaProt scores
    saprot_path = os.path.join(results_dir, "saprot_scores.csv")
    if os.path.exists(saprot_path):
        print("\nProcessing SaProt scores...", flush=True)
        saprot = pd.read_csv(saprot_path)
        score_col = [c for c in saprot.columns if 'LLR' in c or 'score' in c.lower()][0]
        merged = labels.merge(saprot[['VariationID', score_col]], on='VariationID', how='inner')
        if len(merged) > 0:
            metrics = compute_metrics(merged['ClinVar_label'], merged[score_col], 'SaProt')
            if metrics:
                all_metrics.append(metrics)
                print(f"  AUROC: {metrics['AUROC']:.4f}", flush=True)
    else:
        print(f"\n  SaProt scores not found: {saprot_path}", flush=True)

    # Summary table
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_path = os.path.join(results_dir, "benchmark_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*80}", flush=True)
        print("BENCHMARK SUMMARY", flush=True)
        print(f"{'='*80}", flush=True)
        print(summary_df.to_string(index=False), flush=True)
        print(f"\nSaved: {summary_path}", flush=True)
    else:
        print("\nNo model results found!", flush=True)

if __name__ == "__main__":
    main()
