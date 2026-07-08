#!/usr/bin/env python3
"""Score variants using ESM2 protein language model.

Uses native ESM library for faster inference.
Computes log-likelihood ratio (LLR) for each variant:
LLR = log P(mutant | context) - log P(wild-type | context)
"""

import pandas as pd
import torch
import numpy as np
import os
import sys
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

    print("Loading ESM2 model (650M)...", flush=True)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using MPS (Apple Silicon)", flush=True)
    else:
        device = torch.device("cpu")
        print("  Using CPU", flush=True)
    model = model.to(device)
    esm_layers = 33
    print("  Model loaded", flush=True)

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
    print(f"Scoring {len(variant_list)} variants (batch_size={BATCH})...", flush=True)

    import time
    t_start = time.time()
    for batch_start in range(0, len(variant_list), BATCH):
        batch = variant_list[batch_start:batch_start + BATCH]

        # Build (label, sequence) pairs for wild-type
        wt_data = [(f"wt_{i}", d['wt_seq'][:1022]) for i, d in enumerate(batch)]

        batch_labels, batch_strs, batch_tokens = batch_converter(wt_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[esm_layers], return_contacts=False)
        logits = results["logits"]

        for i, d in enumerate(batch):
            pos = d['aa_pos'] - 1  # 0-based
            token_pos = pos + 1  # +1 for BOS

            if token_pos >= logits.shape[1]:
                continue

            log_probs = torch.log_softmax(logits[i], dim=-1)

            wt_id = alphabet.get_idx(d['ref_aa'])
            mut_id = alphabet.get_idx(d['alt_aa'])

            if wt_id == alphabet.unk_idx or mut_id == alphabet.unk_idx:
                continue

            wt_ll = log_probs[token_pos, wt_id].item()
            mut_ll = log_probs[token_pos, mut_id].item()
            llr = mut_ll - wt_ll

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
                'ESM2_LLR': llr,
                'ESM2_wt_ll': wt_ll,
                'ESM2_mut_ll': mut_ll
            })

        done = batch_start + len(batch)
        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        if done % 200 == 0 or done >= len(variant_list):
            print(f"  [{done}/{len(variant_list)}] Scored: {len(all_results)} | {rate:.1f} vars/s | {elapsed:.0f}s elapsed", flush=True)

    print(f"\n  Completed: {len(all_results)} scored", flush=True)

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join(output_dir, "esm2_scores.csv")
    results_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)

    if len(results_df) > 0:
        print(f"\n  ESM2 Score Statistics:", flush=True)
        print(f"    Mean LLR: {results_df['ESM2_LLR'].mean():.4f}", flush=True)
        print(f"    Std LLR: {results_df['ESM2_LLR'].std():.4f}", flush=True)
        for label in [0.0, 1.0]:
            subset = results_df[results_df['ClinVar_label'] == label]
            name = "Benign" if label == 0.0 else "Pathogenic"
            print(f"    {name}: mean={subset['ESM2_LLR'].mean():.4f}, std={subset['ESM2_LLR'].std():.4f}", flush=True)

if __name__ == "__main__":
    main()
