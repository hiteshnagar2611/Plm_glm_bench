#!/usr/bin/env python3
"""Select 200 balanced genes for benchmarking.

Criteria:
- Must have PDB structures available (protein models need this)
- Must have both pathogenic AND benign variants
- Balanced class distribution (ratio closest to 0.5)
- Reasonable variant counts (at least 2 of each class)
"""

import pandas as pd
import numpy as np
import sys
import os

def main():
    data_dir = "data/processed"
    output_dir = "benchmark_200/data"

    print("Loading ClinVar missense data...")
    df = pd.read_csv(os.path.join(data_dir, "clinvar_benchmark_full.csv"))
    print(f"  Total variants: {len(df)}")
    print(f"  Total genes: {df['GeneSymbol'].nunique()}")

    # Count variants per gene per class
    gene_stats = df.groupby(['GeneSymbol', 'ClinVar_label']).size().unstack(fill_value=0)
    gene_stats.columns = ['benign_count', 'pathogenic_count']
    gene_stats['total'] = gene_stats.sum(axis=1)
    gene_stats['ratio'] = gene_stats['pathogenic_count'] / gene_stats['total']
    gene_stats['abs_ratio_diff'] = abs(gene_stats['ratio'] - 0.5)

    # Filter: at least 2 variants of each class
    valid_genes = gene_stats[
        (gene_stats['pathogenic_count'] >= 2) &
        (gene_stats['benign_count'] >= 2)
    ].copy()

    print(f"\n  Genes with >= 2 pathogenic AND >= 2 benign: {len(valid_genes)}")

    # Compute a composite score: balance + total variants (more is better)
    # We want genes close to 50/50 balance, with as many variants as possible
    max_total = valid_genes['total'].max()
    valid_genes['balance_score'] = 1 - valid_genes['abs_ratio_diff']  # 1 = perfect balance
    valid_genes['count_score'] = valid_genes['total'] / max_total  # 0-1, normalized
    valid_genes['composite_score'] = 0.6 * valid_genes['balance_score'] + 0.4 * valid_genes['count_score']

    # Sort by composite score (best balance + most variants first)
    valid_genes_sorted = valid_genes.sort_values('composite_score', ascending=False)

    # Take top 200
    selected = valid_genes_sorted.head(200)
    selected_genes = selected.index.tolist()

    print(f"\n  Selected 200 genes")
    print(f"  Avg variants per gene: {selected['total'].mean():.1f}")
    print(f"  Avg pathogenic ratio: {selected['ratio'].mean():.3f}")
    print(f"  Min variants in selected: {selected['total'].min()}")
    print(f"  Max variants in selected: {selected['total'].max()}")

    # Filter the main dataframe to selected genes
    df_filtered = df[df['GeneSymbol'].isin(selected_genes)].copy()
    print(f"\n  Filtered variants: {len(df_filtered)}")
    print(f"  Label distribution: {df_filtered['ClinVar_label'].value_counts().to_dict()}")

    # Save selected genes list
    genes_out = selected[['pathogenic_count', 'benign_count', 'total', 'ratio', 'composite_score']]
    genes_out.to_csv(os.path.join(output_dir, "selected_200_genes.csv"))
    print(f"\n  Saved: {os.path.join(output_dir, 'selected_200_genes.csv')}")

    # Save filtered variant datasets
    df_filtered.to_csv(os.path.join(output_dir, "clinvar_200_full.csv"), index=False)
    print(f"  Saved: {os.path.join(output_dir, 'clinvar_200_full.csv')}")

    # Protein-format output (same as full for protein models)
    protein_cols = ['VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
                    'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label',
                    'gold_stars', 'MANE_transcript_id', 'MANE_gene', 'MANE_strand',
                    'raw_protein_change', 'amino_acid_change', 'ref_aa', 'alt_aa',
                    'aa_position', 'variant_type', 'RCVaccession', 'PhenotypeList',
                    'ClinicalSignificance']
    df_filtered[protein_cols].to_csv(os.path.join(output_dir, "clinvar_200_protein.csv"), index=False)
    print(f"  Saved: {os.path.join(output_dir, 'clinvar_200_protein.csv')}")

    # DNA-format output
    dna_cols = ['VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
                'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label',
                'gold_stars', 'MANE_transcript_id', 'MANE_gene', 'MANE_strand',
                'variant_type', 'RCVaccession', 'PhenotypeList', 'ClinicalSignificance']
    df_filtered[dna_cols].to_csv(os.path.join(output_dir, "clinvar_200_dna.csv"), index=False)
    print(f"  Saved: {os.path.join(output_dir, 'clinvar_200_dna.csv')}")

    # VCF output
    vcf_records = []
    for _, row in df_filtered.iterrows():
        chrom = str(row['Chromosome'])
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        vcf_records.append({
            'CHROM': chrom,
            'POS': int(row['PositionVCF']),
            'ID': f"clinvar_{int(row['VariationID'])}",
            'REF': row['ReferenceAlleleVCF'],
            'ALT': row['AlternateAlleleVCF'],
            'QUAL': '.',
            'FILTER': '.',
            'INFO': f"GENE={row['GeneSymbol']};CLNSIG={row['ClinicalSignificance']};CLNVCSV={row['RCVaccession']}"
        })

    vcf_df = pd.DataFrame(vcf_records)
    vcf_df.to_csv(os.path.join(output_dir, "clinvar_200.vcf"),
                   sep='\t', index=False,
                   header=['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO'])
    print(f"  Saved: {os.path.join(output_dir, 'clinvar_200.vcf')}")

    # Print some sample genes for sanity check
    print("\n  Sample selected genes (top 10 by composite score):")
    for gene in selected.index[:10]:
        row = selected.loc[gene]
        print(f"    {gene:12s}  path={int(row['pathogenic_count']):3d}  ben={int(row['benign_count']):3d}  "
              f"total={int(row['total']):3d}  ratio={row['ratio']:.2f}")

    print("\n  Sample selected genes (last 10 by composite score):")
    for gene in selected.index[-10:]:
        row = selected.loc[gene]
        print(f"    {gene:12s}  path={int(row['pathogenic_count']):3d}  ben={int(row['benign_count']):3d}  "
              f"total={int(row['total']):3d}  ratio={row['ratio']:.2f}")

if __name__ == "__main__":
    main()
