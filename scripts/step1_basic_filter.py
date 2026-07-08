#!/usr/bin/env python3
"""
Step 1: Basic ClinVar filtering (fast)
"""
import os
import gzip
import re
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CLINICAL_SIGNIFICANCE_TO_LABEL = {
    'Benign': 0, 'Likely benign': 0.1, 'Benign/Likely benign': 0.1,
    'Pathogenic': 1, 'Likely pathogenic': 1.1, 'Pathogenic/Likely pathogenic': 1.1,
}

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

def parse_hgvs_protein_change(variant_name):
    protein_pattern = re.compile(r'\(p\.(.+)\)')
    match = protein_pattern.search(variant_name)
    if not match:
        return None, None, None, None, None
    
    raw_change = match.group(1)
    try:
        last_digit_pos = list(re.finditer(r'\d', raw_change))[-1].start()
        ref_part = raw_change[:(last_digit_pos + 1)]
        alt_part = raw_change[(last_digit_pos + 1):]
        
        ref_parts = ref_part.split('_')
        ref_aa = ref_parts[0][:3]
        pos = int(ref_parts[0][3:])
        
        if alt_part == '=':
            return ref_aa, pos, '=', 'synonymous', raw_change
        elif alt_part == '*':
            return ref_aa, pos, '*', 'stop_gain', raw_change
        elif alt_part.endswith('fs'):
            return ref_aa, pos, 'fs', 'frameshift', raw_change
        else:
            alt_aa = alt_part[:3] if len(alt_part) >= 3 else alt_part
            return ref_aa, pos, alt_aa, 'missense', raw_change
    except:
        return None, None, None, None, raw_change

def main():
    print("Loading ClinVar variant_summary.txt.gz...")
    clinvar = pd.read_csv(
        RAW_DIR / "variant_summary.txt.gz",
        sep='\t', dtype={'Chromosome': str}, low_memory=False, compression='gzip'
    )
    print(f"Loaded {len(clinvar):,} variants")
    
    # Filter GRCh38
    clinvar = clinvar[clinvar['Assembly'] == 'GRCh38']
    print(f"GRCh38: {len(clinvar):,}")
    
    # Filter SNVs
    clinvar = clinvar[clinvar['Type'] == 'single nucleotide variant']
    print(f"SNVs: {len(clinvar):,}")
    
    # Exclude somatic
    clinvar = clinvar[clinvar['OriginSimple'] != 'somatic'].reset_index(drop=True)
    print(f"Non-somatic: {len(clinvar):,}")
    
    # Standard chromosomes
    basic_chromosomes = list(map(str, range(1, 23))) + ['X', 'Y']
    clinvar = clinvar[clinvar['Chromosome'].isin(basic_chromosomes)]
    print(f"Standard chrom: {len(clinvar):,}")
    
    # Clinical significance
    clinvar['ClinVar_label'] = clinvar['ClinicalSignificance'].map(CLINICAL_SIGNIFICANCE_TO_LABEL)
    clinvar = clinvar[clinvar['ClinVar_label'].isin([0, 0.1, 1, 1.1])]
    
    # Review status
    clinvar['gold_stars'] = clinvar['ReviewStatus'].map(REVIEW_STATUS_TO_GOLD_STARS)
    clinvar = clinvar[clinvar['gold_stars'].isin([1, 2, 3, 4])]
    
    # Valid nucleotides
    valid_nucleotides = {'A', 'T', 'C', 'G'}
    clinvar = clinvar[
        (clinvar['ReferenceAlleleVCF'].isin(valid_nucleotides)) &
        (clinvar['AlternateAlleleVCF'].isin(valid_nucleotides))
    ].reset_index(drop=True)
    
    print(f"\nFiltered: {len(clinvar):,}")
    print(f"Pathogenic: {len(clinvar[clinvar['ClinVar_label'].isin([1, 1.1])]):,}")
    print(f"Benign: {len(clinvar[clinvar['ClinVar_label'].isin([0, 0.1])]):,}")
    
    # Parse protein changes
    print("\nParsing protein changes...")
    protein_info = clinvar['Name'].apply(lambda x: pd.Series(parse_hgvs_protein_change(x)))
    clinvar['ref_aa'] = protein_info[0]
    clinvar['aa_position'] = protein_info[1]
    clinvar['alt_aa'] = protein_info[2]
    clinvar['variant_type'] = protein_info[3]
    clinvar['raw_protein_change'] = protein_info[4]
    
    clinvar['variant_type'] = clinvar['variant_type'].fillna('other')
    
    print(f"\nVariant types:")
    print(clinvar['variant_type'].value_counts())
    
    # Save intermediate
    output_path = PROCESSED_DIR / "clinvar_filtered_intermediate.csv"
    clinvar.to_csv(output_path, index=False)
    print(f"\nSaved intermediate: {output_path}")

if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    main()
