#!/usr/bin/env python3
"""Score benchmark_v3 with ProtT5-XL on CPU."""

import pandas as pd
import torch
import time
import warnings
warnings.filterwarnings('ignore')
from transformers import T5EncoderModel, T5Tokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
prots = pd.read_csv('benchmark_v3/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))

device = 'cpu'
print(f"Device: {device}")

print("Loading ProtT5-XL...")
tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_uniref50').eval().to(device)
print("Model loaded")

results = []
t0 = time.time()

for gene, group in df.groupby('GeneSymbol'):
    seq = gene_seq.get(gene)
    if not seq:
        continue

    seq_spaced = ' '.join(list(seq))

    for _, row in group.iterrows():
        pos = int(row['aa_position']) - 1
        if pos >= len(seq):
            continue

        alt = row['alt_aa'].upper()[0]
        mut_seq = seq[:pos] + alt + seq[pos+1:]
        mut_seq_spaced = ' '.join(list(mut_seq))

        with torch.no_grad():
            wt_ids = tokenizer(seq_spaced, return_tensors='pt', max_length=1024, truncation=True).input_ids.to(device)
            wt_out = model(wt_ids)
            wt_repr = wt_out.last_hidden_state[0, pos+1]

            mut_ids = tokenizer(mut_seq_spaced, return_tensors='pt', max_length=1024, truncation=True).input_ids.to(device)
            mut_out = model(mut_ids)
            mut_repr = mut_out.last_hidden_state[0, pos+1]

        cos_sim = torch.nn.functional.cosine_similarity(wt_repr.unsqueeze(0), mut_repr.unsqueeze(0)).item()

        results.append({
            'VariationID': row['VariationID'],
            'GeneSymbol': gene,
            'ClinVar_label': row['ClinVar_label'],
            'ref_aa': row['ref_aa'],
            'alt_aa': row['alt_aa'],
            'aa_position': row['aa_position'],
            'ProtT5_score': cos_sim
        })

    if len(results) % 200 == 0 and len(results) > 0:
        elapsed = time.time() - t0
        print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s")

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/prott5_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
