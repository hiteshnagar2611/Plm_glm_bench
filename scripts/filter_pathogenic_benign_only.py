#!/usr/bin/env python3
"""Filter to only Pathogenic and Benign (no Likely)."""
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# Load full dataset
df = pd.read_csv(PROCESSED_DIR / "clinvar_benchmark_full.csv", low_memory=False)
print(f"Original: {len(df):,} variants")

# Keep only Pathogenic (1) and Benign (0)
df = df[df['ClinVar_label'].isin([0, 1])].reset_index(drop=True)
print(f"After filtering: {len(df):,} variants")
print(f"  Benign: {len(df[df['ClinVar_label']==0]):,}")
print(f"  Pathogenic: {len(df[df['ClinVar_label']==1]):,}")

# Save full
df.to_csv(PROCESSED_DIR / "clinvar_benchmark_full.csv", index=False)

# Save protein (missense + synonymous)
protein = df[df['variant_type'].isin(['missense', 'synonymous'])]
protein.to_csv(PROCESSED_DIR / "clinvar_missense_protein.csv", index=False)
print(f"\nProtein dataset: {len(protein):,}")

# Save DNA
dna_cols = ['VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
            'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label', 'gold_stars',
            'MANE_transcript_id', 'MANE_strand', 'variant_type',
            'RCVaccession', 'PhenotypeList']
df[dna_cols].to_csv(PROCESSED_DIR / "clinvar_benchmark_dna.csv", index=False)

# Save VCF
with open(PROCESSED_DIR / "clinvar_benchmark.vcf", 'w') as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write("##source=ClinVar\n")
    f.write("##reference=GRCh38\n")
    f.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"Clinical significance\">\n")
    f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
    f.write("##INFO=<ID=AA,Number=1,Type=String,Description=\"Amino acid change\">\n")
    f.write("##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Variant type\">\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for _, row in df.iterrows():
        vid = f"rs{int(row['VariationID'])}" if pd.notna(row['VariationID']) else '.'
        info = f"CLNSIG={'Pathogenic' if row['ClinVar_label']==1 else 'Benign'};GENE={row['GeneSymbol']};TYPE={row['variant_type']}"
        if pd.notna(row.get('amino_acid_change')):
            info += f";AA={row['amino_acid_change']}"
        f.write(f"{row['Chromosome']}\t{int(row['PositionVCF'])}\t{vid}\t{row['ReferenceAlleleVCF']}\t{row['AlternateAlleleVCF']}\t.\tPASS\t{info}\n")

print("\n✓ All files updated (Pathogenic + Benign only)")
