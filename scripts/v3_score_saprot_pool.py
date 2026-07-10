#!/usr/bin/env python3
"""Score benchmark_v3 with SaProt-650M: max pool + cosine similarity."""

import pandas as pd
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
prots = pd.read_csv('benchmark_v3/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))

af_3di_dir = Path('benchmark_v3/data/alphafold_3di')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading SaProt...")
tokenizer = AutoTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
model = AutoModelForMaskedLM.from_pretrained('westlake-repl/SaProt_650M_AF2').eval().to(device)

def get_3di(gene):
    three_di_file = af_3di_dir / gene
    if three_di_file.exists():
        with open(three_di_file) as f:
            lines = f.readlines()
            if lines:
                return lines[0].strip()
    return None

results = []
t0 = time.time()
skipped = 0

with torch.no_grad():
    for gene, group in df.groupby('GeneSymbol'):
        seq = gene_seq.get(gene)
        three_di = get_3di(gene)
        if not seq or not three_di or len(seq) > 1022:
            skipped += len(group)
            continue

        three_di = three_di.lower()[:len(seq)]

        for _, row in group.iterrows():
            pos = int(row['aa_position']) - 1
            if pos >= len(seq):
                skipped += 1
                continue

            wt_interleaved = ''.join(a + d for a, d in zip(seq, three_di))
            mut_aa = row['alt_aa'].upper()[0]
            mut_seq = seq[:pos] + mut_aa + seq[pos+1:]
            mut_interleaved = ''.join(a + d for a, d in zip(mut_seq, three_di))

            try:
                wt_ids = tokenizer(wt_interleaved, return_tensors='pt', max_length=2048, truncation=True, padding='max_length')
                mut_ids = tokenizer(mut_interleaved, return_tensors='pt', max_length=2048, truncation=True, padding='max_length')

                wt_ids = {k: v.to(device) for k, v in wt_ids.items()}
                mut_ids = {k: v.to(device) for k, v in mut_ids.items()}

                wt_out = model(**wt_ids, output_hidden_states=True)
                mut_out = model(**mut_ids, output_hidden_states=True)

                wt_hidden = wt_out.hidden_states[-1][0]
                mut_hidden = mut_out.hidden_states[-1][0]

                wt_pool = wt_hidden[1:-1].max(dim=0).values
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
                    'SaProt_pool': score
                })
            except Exception as e:
                if len(results) < 3:
                    print(f"  Error: {e}")

            if len(results) % 500 == 0 and len(results) > 0:
                elapsed = time.time() - t0
                print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s", flush=True)

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/saprot_pool_scores.csv', index=False)
print(f"\nDone: {len(out)} variants, {skipped} skipped, in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
