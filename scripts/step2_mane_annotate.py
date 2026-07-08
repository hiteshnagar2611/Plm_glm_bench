#!/usr/bin/env python3
"""
Step 2: Add MANE annotations and generate final outputs
"""
import gzip
import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

def load_mane():
    """Load MANE annotations efficiently."""
    print("Loading MANE annotations...")
    mane_path = RAW_DIR / "MANE.GRCh38.v1.5.refseq_genomic.gff.gz"
    
    with gzip.open(mane_path, "rt") as f:
        lines = f.readlines()
    
    columns = ["Chromosome", "Source", "Feature", "Start", "End", "Score", "Strand", "Frame", "Attributes"]
    data = []
    
    for line in lines:
        if not line.startswith("#"):
            fields = line.strip().split("\t")
            data.append(fields)
    
    mane_df = pd.DataFrame(data, columns=columns)
    
    # Parse attributes
    def parse_attributes(attr_string):
        attrs = {}
        for field in ['ID', 'Parent', 'transcript_id', 'gene', 'tag']:
            match = re.search(f'{field}=([^;]+)', attr_string)
            if match:
                attrs[field] = match.group(1)
        return attrs
    
    parsed_attrs = mane_df['Attributes'].apply(parse_attributes)
    parsed_df = pd.DataFrame(parsed_attrs.tolist())
    mane_df = pd.concat([mane_df, parsed_df], axis=1)
    
    # Filter MANE Select transcripts
    mane_select = mane_df[
        (mane_df['tag'] == 'MANE Select') & 
        (mane_df['Feature'] == 'mRNA')
    ].copy()
    
    # Clean chromosome (remove 'chr' prefix)
    mane_select['Chrom_clean'] = mane_select['Chromosome'].str.replace('chr', '')
    mane_select['Start_int'] = mane_select['Start'].astype(int)
    mane_select['End_int'] = mane_select['End'].astype(int)
    
    print(f"Loaded {len(mane_select)} MANE Select transcripts")
    return mane_select

def annotate_with_mane_vectorized(clinvar_df, mane_select):
    """Annotate variants using vectorized operations for speed."""
    print("Annotating with MANE transcripts...")
    
    # Initialize columns
    clinvar_df['MANE_transcript_id'] = None
    clinvar_df['MANE_gene'] = None
    clinvar_df['MANE_strand'] = None
    
    # Process by chromosome for efficiency
    for chrom in clinvar_df['Chromosome'].unique():
        chrom_variants = clinvar_df[clinvar_df['Chromosome'] == chrom].index
        chrom_mane = mane_select[mane_select['Chrom_clean'] == chrom]
        
        if chrom_mane.empty:
            continue
        
        for idx in chrom_variants:
            pos = int(clinvar_df.at[idx, 'PositionVCF'])
            matches = chrom_mane[
                (chrom_mane['Start_int'] <= pos) & 
                (chrom_mane['End_int'] >= pos)
            ]
            
            if not matches.empty:
                match = matches.iloc[0]
                clinvar_df.at[idx, 'MANE_transcript_id'] = match.get('transcript_id', None)
                clinvar_df.at[idx, 'MANE_gene'] = match.get('gene', None)
                clinvar_df.at[idx, 'MANE_strand'] = match.get('Strand', None)
    
    annotated = clinvar_df['MANE_transcript_id'].notna().sum()
    print(f"Annotated {annotated}/{len(clinvar_df)} variants")
    return clinvar_df

def generate_outputs(clinvar_df):
    """Generate final output files."""
    print("\nGenerating output files...")
    
    # Create amino acid change string
    def make_aa_change(row):
        if pd.notna(row['ref_aa']) and pd.notna(row['aa_position']) and pd.notna(row['alt_aa']):
            return f"{row['ref_aa']}{int(row['aa_position'])}{row['alt_aa']}"
        return None
    
    clinvar_df['amino_acid_change'] = clinvar_df.apply(make_aa_change, axis=1)
    
    # 1. Full benchmark
    print("[1/4] Full benchmark...")
    full_columns = [
        'VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
        'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label', 'gold_stars',
        'MANE_transcript_id', 'MANE_gene', 'MANE_strand',
        'raw_protein_change', 'amino_acid_change', 'ref_aa', 'alt_aa', 'aa_position',
        'variant_type', 'RCVaccession', 'PhenotypeList', 'ClinicalSignificance'
    ]
    clinvar_df[full_columns].to_csv(PROCESSED_DIR / "clinvar_benchmark_full.csv", index=False)
    
    # 2. Missense + synonymous for protein models
    print("[2/4] Protein model dataset...")
    protein_df = clinvar_df[clinvar_df['variant_type'].isin(['missense', 'synonymous'])]
    protein_df[full_columns].to_csv(PROCESSED_DIR / "clinvar_missense_protein.csv", index=False)
    
    # 3. DNA model dataset
    print("[3/4] DNA model dataset...")
    dna_columns = [
        'VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
        'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label', 'gold_stars',
        'MANE_transcript_id', 'MANE_strand', 'variant_type',
        'RCVaccession', 'PhenotypeList'
    ]
    clinvar_df[dna_columns].to_csv(PROCESSED_DIR / "clinvar_benchmark_dna.csv", index=False)
    
    # 4. VCF format
    print("[4/4] VCF format...")
    vcf_path = PROCESSED_DIR / "clinvar_benchmark.vcf"
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=ClinVar\n")
        f.write("##reference=GRCh38\n")
        f.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"Clinical significance\">\n")
        f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
        f.write("##INFO=<ID=AA,Number=1,Type=String,Description=\"Amino acid change\">\n")
        f.write("##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Variant type\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        for _, row in clinvar_df.iterrows():
            chrom = row['Chromosome']
            pos = int(row['PositionVCF'])
            vid = f"rs{int(row['VariationID'])}" if pd.notna(row['VariationID']) else '.'
            ref = row['ReferenceAlleleVCF']
            alt = row['AlternateAlleleVCF']
            info_parts = [
                f"CLNSIG={row['ClinicalSignificance']}",
                f"GENE={row['GeneSymbol']}",
                f"TYPE={row['variant_type']}"
            ]
            if pd.notna(row['amino_acid_change']):
                info_parts.append(f"AA={row['amino_acid_change']}")
            info = ";".join(info_parts)
            f.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total variants: {len(clinvar_df):,}")
    print(f"\nBy clinical significance:")
    print(f"  Pathogenic:        {len(clinvar_df[clinvar_df['ClinVar_label']==1]):,}")
    print(f"  Likely pathogenic: {len(clinvar_df[clinvar_df['ClinVar_label']==1.1]):,}")
    print(f"  Benign:            {len(clinvar_df[clinvar_df['ClinVar_label']==0]):,}")
    print(f"  Likely benign:     {len(clinvar_df[clinvar_df['ClinVar_label']==0.1]):,}")
    print(f"\nBy variant type:")
    for vtype in clinvar_df['variant_type'].unique():
        print(f"  {vtype}: {len(clinvar_df[clinvar_df['variant_type']==vtype]):,}")
    print("="*60)

def main():
    # Load intermediate data
    print("Loading intermediate data...")
    clinvar_df = pd.read_csv(PROCESSED_DIR / "clinvar_filtered_intermediate.csv", low_memory=False)
    print(f"Loaded {len(clinvar_df):,} variants")
    
    # Load MANE
    mane_select = load_mane()
    
    # Annotate
    clinvar_df = annotate_with_mane_vectorized(clinvar_df, mane_select)
    
    # Generate outputs
    generate_outputs(clinvar_df)
    
    print("\n✓ Pipeline completed!")
    print(f"Output files in: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
