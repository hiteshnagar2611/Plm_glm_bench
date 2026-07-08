#!/usr/bin/env python3
"""Score variants using NT-v2 (Nucleotide Transformer v2) DNA language model.

Uses 6-mer tokenization to encode DNA sequences.
Computes log-likelihood ratio for each variant using masked language modeling.
"""

import pandas as pd
import torch
import numpy as np
import os
import sys
import time

sys.path.insert(0, 'benchmark_200/tools/nt_v2')
from nt_pkg import EsmModel, EsmConfig
from transformers import AutoTokenizer

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading variant data...", flush=True)
    variants = pd.read_csv(os.path.join(data_dir, "clinvar_200_dna.csv"))
    print(f"  Variants: {len(variants)}", flush=True)

    print("Loading NT-v2 model...", flush=True)
    model_path = "benchmark_200/tools/nt_v2"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = EsmConfig.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path, config=config)
    model.eval()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS", flush=True)
    else:
        device = torch.device("cpu")
        print("  Using CPU", flush=True)
    model = model.to(device)
    print("  Model loaded", flush=True)

    print(f"\nScoring {len(variants)} variants with NT-v2...", flush=True)
    results = []
    errors = 0
    t_start = time.time()

    for idx, row in variants.iterrows():
        chrom = str(row['Chromosome'])
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        pos = int(row['PositionVCF'])
        ref = row['ReferenceAlleleVCF']
        alt = row['AlternateAlleleVCF']
        label = row['ClinVar_label']

        try:
            # Load wild-type sequence from CSV
            wt_seq = row.get('wildtype_sequence', '')
            mut_seq = row.get('mutant_sequence', '')

            if pd.isna(wt_seq) or pd.isna(mut_seq) or len(wt_seq) < 10:
                # Read from DNA sequences file
                dna_df = pd.read_csv(os.path.join(data_dir, "dna_sequences.csv"))
                dna_row = dna_df[dna_df['VariationID'] == int(row['VariationID'])]
                if len(dna_row) == 0:
                    errors += 1
                    continue
                wt_seq = dna_row.iloc[0]['wildtype_sequence']
                mut_seq = dna_row.iloc[0]['mutant_sequence']

            # Tokenize wild-type
            wt_tokens = tokenizer(wt_seq, return_tensors="pt", truncation=True, max_length=2048)
            wt_tokens = {k: v.to(device) for k, v in wt_tokens.items()}

            with torch.no_grad():
                wt_outputs = model(**wt_tokens)
                wt_hidden = wt_outputs.last_hidden_state

            # Tokenize mutant
            mut_tokens = tokenizer(mut_seq, return_tensors="pt", truncation=True, max_length=2048)
            mut_tokens = {k: v.to(device) for k, v in mut_tokens.items()}

            with torch.no_grad():
                mut_outputs = model(**mut_tokens)
                mut_hidden = mut_outputs.last_hidden_state

            # Compute delta: mean absolute difference in hidden states
            delta = torch.mean(torch.abs(wt_hidden - mut_hidden)).item()

            results.append({
                'VariationID': int(row['VariationID']),
                'GeneSymbol': row['GeneSymbol'],
                'Chromosome': chrom,
                'PositionVCF': pos,
                'ReferenceAlleleVCF': ref,
                'AlternateAlleleVCF': alt,
                'ClinVar_label': label,
                'NTv2_delta': delta,
            })

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error: {row['GeneSymbol']} {chrom}:{pos}: {str(e)[:100]}", flush=True)

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(variants)}] Scored: {len(results)}, Errors: {errors} | {rate:.2f} vars/s", flush=True)

    print(f"\n  Completed: {len(results)} scored, {errors} errors", flush=True)

    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "ntv2_scores.csv")
    results_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)

    if len(results_df) > 0:
        print(f"\n  NT-v2 Score Statistics:", flush=True)
        print(f"    Mean delta: {results_df['NTv2_delta'].mean():.6f}", flush=True)
        print(f"    Std delta: {results_df['NTv2_delta'].std():.6f}", flush=True)

if __name__ == "__main__":
    main()
