#!/usr/bin/env python3
"""
ClinVar Benchmark Dataset Filter for PLM and Genome Model Evaluation

This script filters ClinVar data to create a benchmark dataset for comparing
protein language models (ESM2, ESM3, SaProt, ProtT5) and genome models
(AlphaGenome, EVO2, NT-v2, HyenaDNA) on pathogenic prediction tasks.

Data sources:
- ClinVar: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/
- MANE: https://ftp.ncbi.nlm.nih.gov/refseq/MANE/
"""

import os
import gzip
import re
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from Bio.SeqUtils import seq1


# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# URLs for data download
CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
MANE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.5/MANE.GRCh38.v1.5.refseq_genomic.gff.gz"

# Clinical significance mapping
CLINICAL_SIGNIFICANCE_TO_LABEL = {
    'Benign': 0,
    'Likely benign': 0.1,
    'Benign/Likely benign': 0.1,
    'Pathogenic': 1,
    'Likely pathogenic': 1.1,
    'Pathogenic/Likely pathogenic': 1.1,
    'Uncertain significance': 2
}

# Review status to gold stars mapping
REVIEW_STATUS_TO_GOLD_STARS = {
    'criteria provided, single submitter': 1,
    'criteria provided, multiple submitters, no conflicts': 2,
    'criteria provided, conflicting interpretations': 1,
    'no assertion criteria provided': np.nan,
    'reviewed by expert panel': 3,
    'no assertion provided': np.nan,
    'no interpretation for the single variant': np.nan,
    'practice guideline': 4,
}

# Valid amino acids for parsing
VALID_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY*')


# ============================================================================
# Download Functions
# ============================================================================

def download_file(url: str, output_path: Path, force: bool = False) -> None:
    """Download a file from URL to local path."""
    if output_path.exists() and not force:
        print(f"File already exists: {output_path}")
        return
    
    print(f"Downloading {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to: {output_path}")


def download_clinvar() -> Path:
    """Download ClinVar variant_summary.txt.gz."""
    output_path = RAW_DIR / "variant_summary.txt.gz"
    download_file(CLINVAR_URL, output_path)
    return output_path


def download_mane() -> Path:
    """Download MANE GFF annotation file."""
    output_path = RAW_DIR / "MANE.GRCh38.v1.5.refseq_genomic.gff.gz"
    download_file(MANE_URL, output_path)
    return output_path


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_hgvs_protein_change(variant_name: str) -> dict:
    """
    Parse protein change from HGVS nomenclature in variant Name field.
    
    Example: "NM_000546.6(CDKN2A):c.248C>T (p.Ala83Val)"
    Returns: {'start_aa': 'A', 'end_pos': 83, 'alt': 'V', 'is_missense': True}
    """
    protein_pattern = re.compile(r'\(p\.(.+)\)')
    match = protein_pattern.search(variant_name)
    
    if not match:
        return {
            'raw_protein_change': None,
            'start_aa': None,
            'end_pos': None,
            'alt': None,
            'is_missense': False,
            'is_synonymous': False,
            'is_stop_gain': False,
        }
    
    raw_change = match.group(1)
    
    try:
        # Extract position (last digits before the change)
        last_digit_pos = list(re.finditer(r'\d', raw_change))[-1].start()
        ref_part = raw_change[:(last_digit_pos + 1)]
        alt_part = raw_change[(last_digit_pos + 1):]
        
        # Parse reference AA and position
        ref_parts = ref_part.split('_')
        if len(ref_parts) == 1:
            ref_aa = seq1(ref_parts[0][:3])
            pos = int(ref_parts[0][3:])
        else:
            ref_aa = seq1(ref_parts[0][:3])
            pos = int(ref_parts[0][3:])
        
        # Parse alternate
        if alt_part == '=':
            alt_aa = '='
            is_synonymous = True
            is_missense = False
            is_stop_gain = False
        elif alt_part == '*':
            alt_aa = '*'
            is_synonymous = False
            is_missense = False
            is_stop_gain = True
        elif alt_part.endswith('fs'):
            alt_aa = 'fs'
            is_synonymous = False
            is_missense = False
            is_stop_gain = False
        else:
            # Regular amino acid change
            alt_aa = seq1(alt_part[:3]) if len(alt_part) >= 3 else alt_part
            is_synonymous = False
            is_missense = ref_aa != alt_aa and alt_aa != '*' and alt_aa != 'fs'
            is_stop_gain = alt_aa == '*'
        
        return {
            'raw_protein_change': raw_change,
            'start_aa': ref_aa,
            'end_pos': pos,
            'alt': alt_aa,
            'is_missense': is_missense,
            'is_synonymous': is_synonymous,
            'is_stop_gain': is_stop_gain,
        }
    except Exception as e:
        return {
            'raw_protein_change': raw_change,
            'start_aa': None,
            'end_pos': None,
            'alt': None,
            'is_missense': False,
            'is_synonymous': False,
            'is_stop_gain': False,
        }


def parse_refseq_id(variant_name: str) -> Optional[str]:
    """Extract RefSeq transcript ID from variant name."""
    pattern = re.compile(r'(NM_\d+\.?\d*)')
    match = pattern.search(variant_name)
    return match.group(1) if match else None


# ============================================================================
# MANE Annotation Functions
# ============================================================================

def load_mane(mane_path: Path) -> pd.DataFrame:
    """Load and parse MANE GFF annotation file."""
    print("Loading MANE annotations...")
    
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
    fields_to_extract = ['ID', 'Parent', 'Dbxref', 'Name', 'description', 'gbkey', 'gene', 
                         'gene_biotype', 'product', 'tag', 'transcript_id']
    
    def parse_attributes(attr_string):
        attrs = {}
        for field in fields_to_extract:
            match = re.search(f'{field}=([^;]+)', attr_string)
            if match:
                attrs[field] = match.group(1)
        # Extract Ensembl ID
        if 'Dbxref' in attrs:
            for val in attrs['Dbxref'].split(','):
                if val.startswith('Ensembl:'):
                    attrs['Ensembl'] = val.split(':')[1]
                    break
        return attrs
    
    parsed_attrs = mane_df['Attributes'].apply(parse_attributes)
    parsed_df = pd.DataFrame(parsed_attrs.tolist())
    mane_df = pd.concat([mane_df, parsed_df], axis=1)
    
    print(f"Loaded {len(mane_df)} MANE features")
    return mane_df


def get_mane_select_transcripts(mane_df: pd.DataFrame) -> pd.DataFrame:
    """Extract MANE Select transcripts (one per gene)."""
    # Filter for MANE Select status
    mane_select = mane_df[mane_df['tag'] == 'MANE Select'].copy()
    
    # Get mRNA features (transcripts)
    mrna_features = mane_select[mane_select['Feature'] == 'mRNA'].copy()
    
    print(f"Found {len(mrna_features)} MANE Select transcripts")
    return mrna_features


def annotate_with_mane(clinvar_df: pd.DataFrame, mane_df: pd.DataFrame) -> pd.DataFrame:
    """Annotate ClinVar variants with MANE transcript information."""
    print("Annotating variants with MANE transcripts...")
    
    # Get MANE Select transcripts
    mane_select = get_mane_select_transcripts(mane_df)
    
    # Create a lookup for chromosome (remove 'chr' prefix if present)
    mane_select['Chrom_clean'] = mane_select['Chromosome'].str.replace('chr', '')
    
    # Build annotation columns
    annotation_cols = ['MANE_transcript_id', 'MANE_gene', 'MANE_strand']
    for col in annotation_cols:
        clinvar_df[col] = None
    
    # Annotate each variant
    for idx, row in clinvar_df.iterrows():
        chrom = str(row['Chromosome'])
        pos = int(row['PositionVCF'])
        
        # Find matching MANE transcripts
        matches = mane_select[
            (mane_select['Chrom_clean'] == chrom) &
            (mane_select['Start'].astype(int) <= pos) &
            (mane_select['End'].astype(int) >= pos)
        ]
        
        if not matches.empty:
            # Take the first match
            match = matches.iloc[0]
            clinvar_df.at[idx, 'MANE_transcript_id'] = match.get('transcript_id', None)
            clinvar_df.at[idx, 'MANE_gene'] = match.get('gene', None)
            clinvar_df.at[idx, 'MANE_strand'] = match.get('Strand', None)
    
    annotated = clinvar_df['MANE_transcript_id'].notna().sum()
    print(f"Annotated {annotated}/{len(clinvar_df)} variants with MANE transcripts")
    
    return clinvar_df


# ============================================================================
# Main Filtering Pipeline
# ============================================================================

def filter_clinvar(variant_summary_path: Path) -> pd.DataFrame:
    """
    Filter ClinVar data for benchmark dataset.
    
    Steps:
    1. Load variant_summary.txt.gz
    2. Filter for GRCh38 assembly
    3. Filter for single nucleotide variants (SNVs)
    4. Exclude somatic variants
    5. Filter for standard chromosomes (1-22, X, Y)
    6. Filter for pathogenic/benign with review stars >= 1
    7. Validate REF/ALT are single nucleotides
    """
    print("=" * 60)
    print("ClinVar Benchmark Dataset Filter")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/7] Loading ClinVar variant_summary...")
    clinvar = pd.read_csv(
        variant_summary_path,
        sep='\t',
        dtype={'Chromosome': str},
        low_memory=False,
        compression='gzip'
    )
    print(f"Loaded {len(clinvar):,} variants")
    
    # Step 2: Filter for GRCh38
    print("\n[2/7] Filtering for GRCh38 assembly...")
    clinvar = clinvar[clinvar['Assembly'] == 'GRCh38']
    print(f"GRCh38 variants: {len(clinvar):,}")
    
    # Step 3: Filter for SNVs
    print("\n[3/7] Filtering for single nucleotide variants...")
    clinvar = clinvar[clinvar['Type'] == 'single nucleotide variant']
    print(f"SNVs: {len(clinvar):,}")
    
    # Step 4: Exclude somatic variants
    print("\n[4/7] Excluding somatic variants...")
    clinvar = clinvar[clinvar['OriginSimple'] != 'somatic'].reset_index(drop=True)
    print(f"After excluding somatic: {len(clinvar):,}")
    
    # Step 5: Filter for standard chromosomes
    print("\n[5/7] Filtering for standard chromosomes (1-22, X, Y)...")
    basic_chromosomes = list(map(str, range(1, 23))) + ['X', 'Y']
    clinvar = clinvar[clinvar['Chromosome'].isin(basic_chromosomes)]
    print(f"Standard chromosomes: {len(clinvar):,}")
    
    # Step 6: Filter for clinical significance and review status
    print("\n[6/7] Filtering for pathogenic/benign with review stars >= 1...")
    clinvar['ClinVar_label'] = clinvar['ClinicalSignificance'].map(CLINICAL_SIGNIFICANCE_TO_LABEL)
    clinvar = clinvar[clinvar['ClinVar_label'].isin([0, 0.1, 1, 1.1])]
    
    clinvar['gold_stars'] = clinvar['ReviewStatus'].map(REVIEW_STATUS_TO_GOLD_STARS)
    clinvar = clinvar[clinvar['gold_stars'].isin([1, 2, 3, 4])]
    
    print(f"Pathogenic/Likely pathogenic: {len(clinvar[clinvar['ClinVar_label'].isin([1, 1.1])]):,}")
    print(f"Benign/Likely benign: {len(clinvar[clinvar['ClinVar_label'].isin([0, 0.1])]):,}")
    
    # Step 7: Validate REF/ALT are single nucleotides
    print("\n[7/7] Validating single nucleotide REF/ALT...")
    valid_nucleotides = {'A', 'T', 'C', 'G'}
    clinvar = clinvar[
        (clinvar['ReferenceAlleleVCF'].isin(valid_nucleotides)) &
        (clinvar['AlternateAlleleVCF'].isin(valid_nucleotides))
    ].reset_index(drop=True)
    
    print(f"\nFinal filtered dataset: {len(clinvar):,} variants")
    print("=" * 60)
    
    return clinvar


def add_protein_annotations(clinvar_df: pd.DataFrame) -> pd.DataFrame:
    """Add protein-level annotations from HGVS nomenclature."""
    print("\nAdding protein annotations...")
    
    # Parse protein changes
    protein_info = clinvar_df['Name'].apply(parse_hgvs_protein_change)
    protein_df = pd.DataFrame(protein_info.tolist())
    
    # Add columns
    clinvar_df['raw_protein_change'] = protein_df['raw_protein_change']
    clinvar_df['ref_aa'] = protein_df['start_aa']
    clinvar_df['aa_position'] = protein_df['end_pos']
    clinvar_df['alt_aa'] = protein_df['alt']
    clinvar_df['is_missense'] = protein_df['is_missense']
    clinvar_df['is_synonymous'] = protein_df['is_synonymous']
    clinvar_df['is_stop_gain'] = protein_df['is_stop_gain']
    
    # Parse RefSeq ID
    clinvar_df['refseq_id'] = clinvar_df['Name'].apply(parse_refseq_id)
    
    # Create amino acid change string (e.g., "R123H")
    clinvar_df['amino_acid_change'] = clinvar_df.apply(
        lambda row: f"{row['ref_aa']}{row['aa_position']}{row['alt_aa']}" 
        if pd.notna(row['ref_aa']) and pd.notna(row['aa_position']) and pd.notna(row['alt_aa'])
        else None,
        axis=1
    )
    
    # Classify variant type
    clinvar_df['variant_type'] = 'other'
    clinvar_df.loc[clinvar_df['is_missense'], 'variant_type'] = 'missense'
    clinvar_df.loc[clinvar_df['is_synonymous'], 'variant_type'] = 'synonymous'
    clinvar_df.loc[clinvar_df['is_stop_gain'], 'variant_type'] = 'stop_gain'
    
    # Print statistics
    print(f"Missense variants: {clinvar_df['is_missense'].sum():,}")
    print(f"Synonymous variants: {clinvar_df['is_synonymous'].sum():,}")
    print(f"Stop gain variants: {clinvar_df['is_stop_gain'].sum():,}")
    
    return clinvar_df


def generate_outputs(clinvar_df: pd.DataFrame) -> None:
    """Generate output files for protein and DNA models."""
    print("\n" + "=" * 60)
    print("Generating Output Files")
    print("=" * 60)
    
    # Define column mappings for cleaner output
    output_columns = [
        'VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
        'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label', 'gold_stars',
        'refseq_id', 'MANE_transcript_id', 'MANE_gene', 'MANE_strand',
        'raw_protein_change', 'amino_acid_change', 'ref_aa', 'alt_aa', 'aa_position',
        'variant_type', 'is_missense', 'is_synonymous', 'is_stop_gain',
        'RCVaccession', 'PhenotypeList', 'ReviewStatus', 'ClinicalSignificance'
    ]
    
    # 1. Full benchmark dataset
    print("\n[1/4] Generating full benchmark dataset...")
    full_df = clinvar_df[output_columns].copy()
    full_path = PROCESSED_DIR / "clinvar_benchmark_full.csv"
    full_df.to_csv(full_path, index=False)
    print(f"Saved: {full_path} ({len(full_df):,} variants)")
    
    # 2. Missense + Synonymous for protein models
    print("\n[2/4] Generating protein model dataset (missense + synonymous)...")
    protein_df = clinvar_df[
        (clinvar_df['variant_type'].isin(['missense', 'synonymous']))
    ][output_columns].copy()
    protein_path = PROCESSED_DIR / "clinvar_missense_protein.csv"
    protein_df.to_csv(protein_path, index=False)
    print(f"Saved: {protein_path} ({len(protein_df):,} variants)")
    print(f"  - Missense: {len(protein_df[protein_df['variant_type'] == 'missense']):,}")
    print(f"  - Synonymous: {len(protein_df[protein_df['variant_type'] == 'synonymous']):,}")
    
    # 3. DNA model dataset
    print("\n[3/4] Generating DNA model dataset...")
    dna_columns = [
        'VariationID', 'GeneSymbol', 'Chromosome', 'PositionVCF',
        'ReferenceAlleleVCF', 'AlternateAlleleVCF', 'ClinVar_label', 'gold_stars',
        'MANE_transcript_id', 'MANE_strand', 'variant_type',
        'RCVaccession', 'PhenotypeList'
    ]
    dna_df = clinvar_df[dna_columns].copy()
    dna_path = PROCESSED_DIR / "clinvar_benchmark_dna.csv"
    dna_df.to_csv(dna_path, index=False)
    print(f"Saved: {dna_path} ({len(dna_df):,} variants)")
    
    # 4. VCF format
    print("\n[4/4] Generating VCF format...")
    vcf_df = pd.DataFrame({
        '#CHROM': clinvar_df['Chromosome'],
        'POS': clinvar_df['PositionVCF'],
        'ID': clinvar_df['VariationID'].apply(lambda x: f"rs{x}" if pd.notna(x) else '.'),
        'REF': clinvar_df['ReferenceAlleleVCF'],
        'ALT': clinvar_df['AlternateAlleleVCF'],
        'QUAL': '.',
        'PASS': 'PASS',
        'INFO': clinvar_df.apply(
            lambda row: f"CLNSIG={row['ClinicalSignificance']};GENE={row['GeneSymbol']};"
                       f"AA={row['amino_acid_change'] if pd.notna(row['amino_acid_change']) else '.'};"
                       f"TYPE={row['variant_type']}",
            axis=1
        ),
        'FORMAT': '.',
    })
    
    vcf_path = PROCESSED_DIR / "clinvar_benchmark.vcf"
    
    # Write VCF with header
    with open(vcf_path, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=ClinVar\n")
        f.write("##reference=GRCh38\n")
        f.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"Clinical significance\">\n")
        f.write("##INFO=<ID=GENE,Number=1,Type=String,Description=\"Gene symbol\">\n")
        f.write("##INFO=<ID=AA,Number=1,Type=String,Description=\"Amino acid change\">\n")
        f.write("##INFO=<ID=TYPE,Number=1,Type=String,Description=\"Variant type\">\n")
        vcf_df.to_csv(f, sep='\t', index=False)
    
    print(f"Saved: {vcf_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"\nTotal variants: {len(clinvar_df):,}")
    print(f"\nBy clinical significance:")
    print(f"  Pathogenic:        {len(clinvar_df[clinvar_df['ClinVar_label'] == 1]):,}")
    print(f"  Likely pathogenic: {len(clinvar_df[clinvar_df['ClinVar_label'] == 1.1]):,}")
    print(f"  Benign:            {len(clinvar_df[clinvar_df['ClinVar_label'] == 0]):,}")
    print(f"  Likely benign:     {len(clinvar_df[clinvar_df['ClinVar_label'] == 0.1]):,}")
    print(f"\nBy variant type:")
    print(f"  Missense:    {len(clinvar_df[clinvar_df['variant_type'] == 'missense']):,}")
    print(f"  Synonymous:  {len(clinvar_df[clinvar_df['variant_type'] == 'synonymous']):,}")
    print(f"  Stop gain:   {len(clinvar_df[clinvar_df['variant_type'] == 'stop_gain']):,}")
    print(f"  Other:       {len(clinvar_df[clinvar_df['variant_type'] == 'other']):,}")
    print(f"\nBy review status (gold stars):")
    for stars in [1, 2, 3, 4]:
        print(f"  {stars} star(s): {len(clinvar_df[clinvar_df['gold_stars'] == stars]):,}")
    print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main pipeline execution."""
    # Create directories
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download data
    variant_summary_path = download_clinvar()
    mane_path = download_mane()
    
    # Filter ClinVar
    clinvar_df = filter_clinvar(variant_summary_path)
    
    # Load MANE and annotate
    mane_df = load_mane(mane_path)
    clinvar_df = annotate_with_mane(clinvar_df, mane_df)
    
    # Add protein annotations
    clinvar_df = add_protein_annotations(clinvar_df)
    
    # Generate outputs
    generate_outputs(clinvar_df)
    
    print("\n✓ Pipeline completed successfully!")
    print(f"Output files are in: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
