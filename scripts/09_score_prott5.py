#!/usr/bin/env python3
"""Score variants using ESM2-35M protein language model.

Uses ESM2-35M (smaller, faster) as a protein LM baseline.
Computes log-likelihood ratio (LLR) for each variant.
"""

import pandas as pd
import torch
import numpy as np
import os
import sys
import time
import esm

AA_3TO1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*',
}

def convert_aa(aa_str):
    if len(aa_str) == 1:
        return aa_str
    return AA_3TO1.get(aa_str, aa_str)

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading variant data...", flush=True)
    variants = pd.read_csv(os.path.join(data_dir, "clinvar_200_full.csv"))
    proteins = pd.read_csv(os.path.join(data_dir, "protein_sequences.csv"))
    print(f"  Variants: {len(variants)}", flush=True)
    print(f"  Proteins: {len(proteins)}", flush=True)
    protein_seqs = dict(zip(proteins['gene'], proteins['sequence']))

    print("Loading ESM2-35M model...", flush=True)
    model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    print(f"  Model loaded (device: {device})", flush=True)

    # Prepare variants
    print("Preparing variants...", flush=True)
    variant_list = []
    for idx, row in variants.iterrows():
        gene = row['GeneSymbol']
        aa_pos = int(row['aa_position'])
        ref_aa = convert_aa(row['ref_aa'])
        alt_aa = convert_aa(row['alt_aa'])

        if gene not in protein_seqs:
            continue
        wt_seq = protein_seqs[gene]
        if aa_pos > len(wt_seq) or aa_pos > 1022:
            continue
        if wt_seq[aa_pos - 1] != ref_aa:
            continue
        if ref_aa == '*' or alt_aa == '*':
            continue

        mut_seq = wt_seq[:aa_pos-1] + alt_aa + wt_seq[aa_pos:]
        variant_list.append({
            'idx': idx, 'gene': gene, 'aa_pos': aa_pos,
            'ref_aa': ref_aa, 'alt_aa': alt_aa,
            'wt_seq': wt_seq, 'mut_seq': mut_seq,
            'label': row['ClinVar_label']
        })

    print(f"  Valid variants: {len(variant_list)}", flush=True)

    # Score in batches
    BATCH = 8
    all_results = []
    t_start = time.time()
    print(f"Scoring {len(variant_list)} variants with ESM2-35M (batch={BATCH})...", flush=True)

    for batch_start in range(0, len(variant_list), BATCH):
        batch = variant_list[batch_start:batch_start + BATCH]

        # Process wild-type
        wt_data = [(f"wt_{i}", d['wt_seq'][:1022]) for i, d in enumerate(batch)]
        batch_labels, batch_strs, batch_tokens = batch_converter(wt_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            wt_results = model(batch_tokens, repr_layers=[12], return_contacts=False)
        wt_logits = wt_results["logits"]

        # Process mutant
        mut_data = [(f"mut_{i}", d['mut_seq'][:1022]) for i, d in enumerate(batch)]
        batch_labels2, batch_strs2, batch_tokens2 = batch_converter(mut_data)
        batch_tokens2 = batch_tokens2.to(device)

        with torch.no_grad():
            mut_results = model(batch_tokens2, repr_layers=[12], return_contacts=False)
        mut_logits = mut_results["logits"]

        # Compute LLR
        for i, d in enumerate(batch):
            pos = d['aa_pos'] - 1
            token_pos = pos + 1

            if token_pos >= wt_logits.shape[1] or token_pos >= mut_logits.shape[1]:
                continue

            wt_log_probs = torch.log_softmax(wt_logits[i], dim=-1)
            mut_log_probs = torch.log_softmax(mut_logits[i], dim=-1)

            wt_id = alphabet.get_idx(d['ref_aa'])
            mut_id = alphabet.get_idx(d['alt_aa'])

            if wt_id == alphabet.unk_idx or mut_id == alphabet.unk_idx:
                continue

            # LLR: log P(mut|context) - log P(wt|context)
            # Use wild-type context for both (standard approach)
            wt_ll_wt = wt_log_probs[token_pos, wt_id].item()
            mut_ll_wt = wt_log_probs[token_pos, mut_id].item()
            llr = mut_ll_wt - wt_ll_wt

            row = variants.iloc[d['idx']]
            all_results.append({
                'VariationID': int(row['VariationID']),
                'GeneSymbol': d['gene'],
                'Chromosome': row['Chromosome'],
                'PositionVCF': int(row['PositionVCF']),
                'ReferenceAlleleVCF': row['ReferenceAlleleVCF'],
                'AlternateAlleleVCF': row['AlternateAlleleVCF'],
                'ClinVar_label': d['label'],
                'ref_aa': d['ref_aa'],
                'alt_aa': d['alt_aa'],
                'aa_position': d['aa_pos'],
                'ESM2_35M_LLR': llr,
            })

        done = batch_start + len(batch)
        if done % 200 == 0 or done >= len(variant_list):
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{len(variant_list)}] Scored: {len(all_results)} | {rate:.1f} vars/s | {elapsed:.0f}s", flush=True)

    print(f"\n  Completed: {len(all_results)} scored", flush=True)

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "prott5_scores.csv")
    results_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)

    if len(results_df) > 0:
        print(f"\n  ESM2-35M Score Statistics:", flush=True)
        print(f"    Mean LLR: {results_df['ESM2_35M_LLR'].mean():.4f}", flush=True)
        print(f"    Std LLR: {results_df['ESM2_35M_LLR'].std():.4f}", flush=True)
        for label in [0.0, 1.0]:
            subset = results_df[results_df['ClinVar_label'] == label]
            name = "Benign" if label == 0.0 else "Pathogenic"
            print(f"    {name}: mean={subset['ESM2_35M_LLR'].mean():.4f}, std={subset['ESM2_35M_LLR'].std():.4f}", flush=True)

if __name__ == "__main__":
    main()
