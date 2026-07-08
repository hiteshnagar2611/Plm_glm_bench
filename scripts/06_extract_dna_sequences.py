#!/usr/bin/env python3
"""Extract DNA sequences for the 200 selected genes.

Extracts 6000 bp flanking sequences around each variant from the hg38 reference genome.
"""

import pandas as pd
import pysam
import os
import sys

def extract_sequence(samfile, chrom, start, end):
    """Extract sequence from reference genome."""
    # Ensure chromosome name has 'chr' prefix
    if not chrom.startswith('chr'):
        chrom = 'chr' + chrom

    try:
        # pysam is 0-based, half-open
        seq = samfile.fetch(chrom, start, end)
        return seq.upper()
    except Exception as e:
        # Try without 'chr' prefix
        if chrom.startswith('chr'):
            try:
                seq = samfile.fetch(chrom[3:], start, end)
                return seq.upper()
            except:
                pass
        return None

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/data/dna_sequences"
    ref_path = "benchmark_200/data/hg38.fa"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading variant data...")
    df = pd.read_csv(os.path.join(data_dir, "clinvar_200_dna.csv"))
    print(f"  Total variants: {len(df)}")
    print(f"  Total genes: {df['GeneSymbol'].nunique()}")

    # Check if reference genome index exists
    fai_path = ref_path + ".fai"
    if not os.path.exists(fai_path):
        print("\nIndexing reference genome...")
        pysam.faidx(ref_path)
        print("  Done.")

    print("\nOpening reference genome...")
    samfile = pysam.FastaFile(ref_path)

    # Extract sequences for each variant
    # Context window: 6000 bp centered on variant
    # For each variant: 3000 bp upstream + variant + 3000 bp downstream
    context_size = 6000

    records = []
    errors = []

    for idx, row in df.iterrows():
        chrom = str(row['Chromosome'])
        pos = int(row['PositionVCF'])  # 1-based VCF position
        ref = row['ReferenceAlleleVCF']
        alt = row['AlternateAlleleVCF']
        gene = row['GeneSymbol']

        # Calculate flanking regions (0-based for pysam)
        # VCF POS is 1-based, so 0-based start is POS-1
        var_start = pos - 1  # 0-based
        var_end = var_start + len(ref)

        half_window = context_size // 2
        seq_start = max(0, var_start - half_window)
        seq_end = var_end + half_window

        seq = extract_sequence(samfile, chrom, seq_start, seq_end)

        if seq is None:
            errors.append(f"  {gene} {chrom}:{pos} {ref}>{alt} - sequence extraction failed")
            continue

        # Verify reference allele in extracted sequence
        var_idx = var_start - seq_start
        extracted_ref = seq[var_idx:var_idx+len(ref)]

        if extracted_ref != ref:
            errors.append(f"  {gene} {chrom}:{pos} {ref}>{alt} - REF mismatch: expected {ref}, got {extracted_ref}")
            continue

        # Create mutant sequence
        mutant_seq = seq[:var_idx] + alt + seq[var_idx+len(ref):]

        # Extract wild-type and mutant around the variant (for context)
        wt_context = seq  # full 6000 bp
        mut_context = mutant_seq

        records.append({
            'VariationID': int(row['VariationID']),
            'GeneSymbol': gene,
            'Chromosome': chrom,
            'PositionVCF': pos,
            'ReferenceAlleleVCF': ref,
            'AlternateAlleleVCF': alt,
            'ClinVar_label': row['ClinVar_label'],
            'context_start': seq_start + 1,  # back to 1-based
            'context_end': seq_end,
            'wildtype_sequence': wt_context,
            'mutant_sequence': mut_context,
            'variant_relative_pos': var_idx
        })

        if (idx + 1) % 500 == 0:
            print(f"  [{idx+1}/{len(df)}] Processed {len(records)} variants, {len(errors)} errors")

    samfile.close()

    print(f"\n  Successfully extracted: {len(records)}")
    print(f"  Errors: {len(errors)}")

    if errors:
        print("\n  First 10 errors:")
        for e in errors[:10]:
            print(e)

    # Save to CSV
    result_df = pd.DataFrame(records)
    output_path = os.path.join(data_dir, "dna_sequences.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")

    # Summary
    print(f"\n  Sequence length distribution:")
    print(f"    All sequences: {context_size} bp")

    # Save individual FASTA files per gene
    fasta_dir = os.path.join(output_dir, "per_gene")
    os.makedirs(fasta_dir, exist_ok=True)

    # Group by gene
    gene_groups = result_df.groupby('GeneSymbol')
    for gene, group in gene_groups:
        # Save wild-type sequences (one per gene, using first variant)
        first_row = group.iloc[0]
        fasta_path = os.path.join(fasta_dir, f"{gene}_wt.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">{gene}_wildtype\n")
            seq = first_row['wildtype_sequence']
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')

    print(f"  Saved per-gene FASTA files: {fasta_dir}")

if __name__ == "__main__":
    main()
