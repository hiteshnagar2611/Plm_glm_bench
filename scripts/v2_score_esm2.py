#!/usr/bin/env python3
"""Score benchmark_v2 with ESM2-650M."""

import pandas as pd
import torch
import esm
import time

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v2/data/benchmark_v2.csv')
prots = pd.read_csv('benchmark_v2/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()

results = []
batch_size = 8
t0 = time.time()

for gene, group in df.groupby('GeneSymbol'):
    seq = gene_seq.get(gene)
    if not seq or len(seq) > 1022:
        continue
    
    for _, row in group.iterrows():
        pos = int(row['aa_position']) - 1
        if pos >= len(seq):
            continue
        
        variant_seq = seq[:pos] + row['alt_aa'].upper()[0] + seq[pos+1:]
        wt_data = [("wt", seq)]
        mut_data = [("mut", variant_seq)]
        
        _, _, wt_tokens = batch_converter(wt_data)
        _, _, mut_tokens = batch_converter(mut_data)
        
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
    
    elapsed = time.time() - t0
    n = len(results)
    if n % 100 == 0 and n > 0:
        print(f"  {n} variants, {elapsed:.0f}s, {n/elapsed:.1f} vars/s")

out = pd.DataFrame(results)
out.to_csv('benchmark_v2/results/esm2_650m_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
