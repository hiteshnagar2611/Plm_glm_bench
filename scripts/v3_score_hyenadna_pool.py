#!/usr/bin/env python3
"""Score benchmark_v3 with HyenaDNA-150M: max pool + cosine similarity."""

import pandas as pd
import torch
import torch.nn.functional as F
import time
from multimolecule import DnaTokenizer, HyenaDnaForCausalLM

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
dna = pd.read_csv('benchmark_v3/data/dna_sequences.csv')
dna_map = dict(zip(dna['VariationID'], dna['seq']))

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading HyenaDNA...")
tokenizer = DnaTokenizer()
model = HyenaDnaForCausalLM.from_pretrained(
    'multimolecule/hyenadna-medium',
    trust_remote_code=True
).eval().to(device)

results = []
t0 = time.time()

with torch.no_grad():
    for _, row in df.iterrows():
        vid = row['VariationID']
        seq = dna_map.get(vid)
        if not seq:
            continue

        ref = row['ReferenceAlleleVCF']
        alt = row['AlternateAlleleVCF']
        variant_pos = 3000

        ref_seq = seq[:variant_pos] + ref + seq[variant_pos+1:]
        alt_seq = seq[:variant_pos] + alt + seq[variant_pos+1:]

        try:
            ref_ids = tokenizer(ref_seq, return_tensors='pt', max_length=6003, truncation=True, padding='max_length')
            alt_ids = tokenizer(alt_seq, return_tensors='pt', max_length=6003, truncation=True, padding='max_length')

            ref_ids = {k: v.to(device) for k, v in ref_ids.items()}
            alt_ids = {k: v.to(device) for k, v in alt_ids.items()}

            ref_out = model(**ref_ids, output_hidden_states=True)
            alt_out = model(**alt_ids, output_hidden_states=True)

            wt_hidden = ref_out.hidden_states[-1][0]   # [seq_len, hidden]
            mut_hidden = alt_out.hidden_states[-1][0]

            wt_pool = wt_hidden[1:-1].max(dim=0).values
            mut_pool = mut_hidden[1:-1].max(dim=0).values

            diff = mut_pool - wt_pool
            score = F.cosine_similarity(diff.unsqueeze(0), wt_pool.unsqueeze(0)).item()

            results.append({
                'VariationID': vid,
                'GeneSymbol': row['GeneSymbol'],
                'ClinVar_label': row['ClinVar_label'],
                'ref_aa': row['ref_aa'],
                'alt_aa': row['alt_aa'],
                'aa_position': row['aa_position'],
                'Hyena_pool': score
            })
        except Exception as e:
            if len(results) < 3:
                print(f"  Error: {e}")

        if len(results) % 500 == 0 and len(results) > 0:
            elapsed = time.time() - t0
            print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s", flush=True)

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/hyena_pool_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
