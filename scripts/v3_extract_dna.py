#!/usr/bin/env python3
"""Extract 6001bp DNA context sequences for benchmark_v3 variants.

Extracts sequences centered on each variant from hg38 reference genome.
"""

import pandas as pd
import pysam
import time
from pathlib import Path

OUT_DIR = Path('benchmark_v3/data')
REF_PATH = 'benchmark_200/data/hg38.fa'

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv(OUT_DIR / 'benchmark_v3.csv')
print(f"Variants: {len(df)}, Genes: {df['GeneSymbol'].unique().size}")

# Check index
fai_path = REF_PATH + ".fai"
import os
if not os.path.exists(fai_path):
    print("Indexing reference genome...")
    pysam.faidx(REF_PATH)

samfile = pysam.FastaFile(REF_PATH)

context_size = 6000  # 3000bp each side
records = []
errors = []
t0 = time.time()

for idx, row in df.iterrows():
    chrom = str(row['Chromosome'])
    pos = int(row['PositionVCF'])  # 1-based VCF
    ref = row['ReferenceAlleleVCF']
    alt = row['AlternateAlleleVCF']

    # 0-based coordinates
    var_start = pos - 1
    var_end = var_start + len(ref)
    half_window = context_size // 2
    seq_start = max(0, var_start - half_window)
    seq_end = var_end + half_window

    try:
        if not chrom.startswith('chr'):
            chrom_fetch = 'chr' + chrom
        else:
            chrom_fetch = chrom

        seq = samfile.fetch(chrom_fetch, seq_start, seq_end).upper()
    except Exception:
        try:
            seq = samfile.fetch(chrom, seq_start, seq_end).upper()
        except Exception as e:
            errors.append(f"{row['GeneSymbol']} {chrom}:{pos} {ref}>{alt} - {e}")
            continue

    # Verify ref allele
    var_idx = var_start - seq_start
    extracted_ref = seq[var_idx:var_idx+len(ref)]

    if extracted_ref != ref:
        errors.append(f"{row['GeneSymbol']} {chrom}:{pos} REF mismatch: {ref} vs {extracted_ref}")
        continue

    records.append({
        'VariationID': int(row['VariationID']),
        'GeneSymbol': row['GeneSymbol'],
        'seq': seq
    })

    if (idx + 1) % 500 == 0:
        print(f"  [{idx+1}/{len(df)}] extracted={len(records)}, errors={len(errors)}")

samfile.close()

out = pd.DataFrame(records)
out.to_csv(OUT_DIR / 'dna_sequences.csv', index=False)

print(f"\nDone: {len(out)} sequences, {len(errors)} errors")
print(f"Time: {time.time()-t0:.0f}s")
if errors[:5]:
    print("Errors:")
    for e in errors[:5]:
        print(f"  {e}")
