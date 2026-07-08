#!/usr/bin/env python3
"""Score benchmark_v3 with HyenaDNA-150M."""

import pandas as pd
import torch
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

        token_pos = variant_pos + 1

        ref_ids = tokenizer(ref_seq, return_tensors='pt', max_length=6003, truncation=True, padding='max_length')
        alt_ids = tokenizer(alt_seq, return_tensors='pt', max_length=6003, truncation=True, padding='max_length')

        ref_ids = {k: v.to(device) for k, v in ref_ids.items()}
        alt_ids = {k: v.to(device) for k, v in alt_ids.items()}

        ref_out = model(**ref_ids)
        alt_out = model(**alt_ids)

        ref_logits = ref_out.logits[0, token_pos]
        alt_logits = alt_out.logits[0, token_pos]

        ref_prob = torch.softmax(ref_logits, dim=-1)
        alt_prob = torch.softmax(alt_logits, dim=-1)

        ref_token = tokenizer(ref)['input_ids'][1]
        alt_token = tokenizer(alt)['input_ids'][1]

        ref_ll = torch.log(ref_prob[ref_token]).item()
        alt_ll = torch.log(alt_prob[alt_token]).item()

        results.append({
            'VariationID': vid,
            'GeneSymbol': row['GeneSymbol'],
            'ClinVar_label': row['ClinVar_label'],
            'ref_aa': row['ref_aa'],
            'alt_aa': row['alt_aa'],
            'aa_position': row['aa_position'],
            'HyenaDNA_LLR': alt_ll - ref_ll
        })

        if len(results) % 500 == 0 and len(results) > 0:
            elapsed = time.time() - t0
            print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s")

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/hyena_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
