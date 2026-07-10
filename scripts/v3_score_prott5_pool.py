#!/usr/bin/env python3
"""Score benchmark_v3 with ProtT5-XL: max pool + cosine similarity."""

import pandas as pd
import torch
import torch.nn.functional as F
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

with torch.no_grad():
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

            try:
                wt_ids = tokenizer(seq_spaced, return_tensors='pt', max_length=1024, truncation=True)
                mut_ids = tokenizer(mut_seq_spaced, return_tensors='pt', max_length=1024, truncation=True)

                wt_ids = {k: v.to(device) for k, v in wt_ids.items()}
                mut_ids = {k: v.to(device) for k, v in mut_ids.items()}

                wt_out = model(**wt_ids)
                mut_out = model(**mut_ids)

                wt_hidden = wt_out.last_hidden_state[0]   # [seq_len, hidden]
                mut_hidden = mut_out.last_hidden_state[0]

                wt_pool = wt_hidden[1:-1].max(dim=0).values   # exclude BOS/EOS
                mut_pool = mut_hidden[1:-1].max(dim=0).values

                diff = mut_pool - wt_pool
                score = F.cosine_similarity(diff.unsqueeze(0), wt_pool.unsqueeze(0)).item()

                results.append({
                    'VariationID': row['VariationID'],
                    'GeneSymbol': gene,
                    'ClinVar_label': row['ClinVar_label'],
                    'ref_aa': row['ref_aa'],
                    'alt_aa': row['alt_aa'],
                    'aa_position': row['aa_position'],
                    'ProtT5_pool': score
                })
            except Exception as e:
                if len(results) < 3:
                    print(f"  Error: {e}")

            if len(results) % 200 == 0 and len(results) > 0:
                elapsed = time.time() - t0
                print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s", flush=True)

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/prott5_pool_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
