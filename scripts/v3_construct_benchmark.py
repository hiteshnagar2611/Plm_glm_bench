#!/usr/bin/env python3
"""Construct benchmark_v3.csv from ClinVar missense data.

Filtering:
1. Only genes with protein sequences < 1001 aa
2. Remove stop-gain (Ter/*) variants — true missense only
3. No position filtering (AlphaFold has full coverage)
"""

import pandas as pd
from pathlib import Path

OUT_DIR = Path('benchmark_v3/data')

# Load all ClinVar missense variants
full = pd.read_csv('data/processed/clinvar_missense_protein.csv')
print(f"Starting: {len(full)} variants, {full['GeneSymbol'].unique().size} genes")

# Load protein sequences (filtered < 1001 aa)
prots = pd.read_csv(OUT_DIR / 'protein_sequences.csv')
valid_genes = set(prots['gene'])
print(f"Genes with protein seq (< 1001 aa): {len(valid_genes)}")

# Filter to valid genes
df = full[full['GeneSymbol'].isin(valid_genes)].copy()
print(f"After gene filter: {len(df)} variants, {df['GeneSymbol'].unique().size} genes")

# Remove stop-gain variants
stop_mask = df['alt_aa'].isin(['Ter', '*']) | df['alt_aa'].str.contains(r'\*', regex=True)
n_stop = stop_mask.sum()
df = df[~stop_mask].copy()
print(f"Removed {n_stop} stop-gain variants: {len(df)} variants remaining")

# All remaining variants are standard missense (alt_aa are 3-letter codes like Ser, Thr)
print(f"After stop-gain removal: {len(df)} variants, {df['GeneSymbol'].unique().size} genes")

# Class distribution (labels may be numeric 1.0/0.0 or string)
if df['ClinVar_label'].dtype in ['float64', 'int64']:
    n_path = (df['ClinVar_label'] == 1.0).sum()
    n_ben = (df['ClinVar_label'] == 0.0).sum()
else:
    n_path = (df['ClinVar_label'] == 'Pathogenic').sum()
    n_ben = (df['ClinVar_label'] == 'Benign').sum()
print(f"\nClass distribution:")
print(f"  Pathogenic: {n_path} ({100*n_path/len(df):.1f}%)")
print(f"  Benign: {n_ben} ({100*n_ben/len(df):.1f}%)")

# Per-gene stats
def count_path(x):
    return (x == 1.0).sum() if x.dtype in ['float64', 'int64'] else (x == 'Pathogenic').sum()
def count_ben(x):
    return (x == 0.0).sum() if x.dtype in ['float64', 'int64'] else (x == 'Benign').sum()

gene_stats = df.groupby('GeneSymbol').agg(
    n_variants=('VariationID', 'count'),
    n_path=('ClinVar_label', count_path),
    n_ben=('ClinVar_label', count_ben),
    chromosomes=('Chromosome', 'first')
).reset_index()

genes_both = gene_stats[(gene_stats['n_path'] > 0) & (gene_stats['n_ben'] > 0)]
print(f"\nGenes with both P and B: {len(genes_both)}/{len(gene_stats)}")

# Save
df.to_csv(OUT_DIR / 'benchmark_v3.csv', index=False)
print(f"\nSaved benchmark_v3.csv: {len(df)} variants, {df['GeneSymbol'].unique().size} genes")

# Summary
print(f"\n{'='*60}")
print(f"Benchmark V3 Summary")
print(f"{'='*60}")
print(f"Total variants: {len(df)}")
print(f"Total genes: {df['GeneSymbol'].unique().size}")
print(f"Pathogenic: {n_path}")
print(f"Benign: {n_ben}")
print(f"Genes with both classes: {len(genes_both)}")
print(f"Filtering: no position filter (AlphaFold full coverage)")
print(f"{'='*60}")
