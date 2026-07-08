#!/usr/bin/env python3
"""Score benchmark_v3 with ESM2-650M (standalone)."""

import pandas as pd
import torch
import esm
import time
import sys

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
prots = pd.read_csv('benchmark_v3/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}", flush=True)

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()
print("Model loaded", flush=True)

results = []
t0 = time.time()
skipped = 0

for gene, group in df.groupby('GeneSymbol'):
    seq = gene_seq.get(gene)
    if not seq or len(seq) > 1022:
        skipped += len(group)
        continue

    for _, row in group.iterrows():
        pos = int(row['aa_position']) - 1
        if pos >= len(seq):
            skipped += 1
            continue

        variant_seq = seq[:pos] + row['alt_aa'].upper()[0] + seq[pos+1:]
        _, _, wt_tokens = batch_converter([('wt', seq)])
        _, _, mut_tokens = batch_converter([('mut', variant_seq)])

        with torch.no_grad():
            wt_out = model(wt_tokens.to(device), repr_layers=[33])
            mut_out = model(mut_tokens.to(device), repr_layers=[33])

        wt_logits = wt_out["logits"][0, pos+1]
        mut_logits = mut_out["logits"][0, pos+1]

        wt_prob = torch.softmax(wt_logits, dim=-1)
        mut_prob = torch.softmax(mut_logits, dim=-1)

        wt_ll = torch.log(wt_prob[alphabet.get_idx(seq[pos])]).item()
        mut_ll = torch.log(mut_prob[alphabet.get_idx(variant_seq[pos])]).item()

        results.append({
            'VariationID': row['VariationID'],
            'GeneSymbol': gene,
            'ClinVar_label': row['ClinVar_label'],
            'ref_aa': row['ref_aa'],
            'alt_aa': row['alt_aa'],
            'aa_position': row['aa_position'],
            'ESM2_LLR': mut_ll - wt_ll
        })

    if len(results) % 500 == 0 and len(results) > 0:
        elapsed = time.time() - t0
        print(f"{len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s, skipped={skipped}", flush=True)

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/esm2_650m_scores.csv', index=False)
print(f"Done: {len(out)} variants in {time.time()-t0:.0f}s, skipped={skipped}", flush=True)
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
