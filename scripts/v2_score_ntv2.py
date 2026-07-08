#!/usr/bin/env python3
"""Score benchmark_v2 with NT-v2 500M on CPU."""

import pandas as pd
import torch
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'benchmark_200/tools/nt_v2')
from nt_pkg import EsmModel, EsmConfig
from transformers import AutoTokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v2/data/benchmark_v2.csv')
dna = pd.read_csv('benchmark_v2/data/dna_sequences.csv')
dna_map = dict(zip(dna['VariationID'], dna['seq']))

device = 'cpu'
print(f"Device: {device}")

model_path = "benchmark_200/tools/nt_v2"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = EsmConfig.from_pretrained(model_path)
model = EsmModel.from_pretrained(model_path, config=config).eval().to(device)
print("Model loaded")

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
            wt_tokens = tokenizer(ref_seq, return_tensors='pt', truncation=True, max_length=2048)
            wt_out = model(**wt_tokens)
            wt_hidden = wt_out.last_hidden_state

            mut_tokens = tokenizer(alt_seq, return_tensors='pt', truncation=True, max_length=2048)
            mut_out = model(**mut_tokens)
            mut_hidden = mut_out.last_hidden_state

            delta = torch.mean(torch.abs(wt_hidden - mut_hidden)).item()

            results.append({
                'VariationID': vid,
                'GeneSymbol': row['GeneSymbol'],
                'ClinVar_label': row['ClinVar_label'],
                'ref_aa': row['ref_aa'],
                'alt_aa': row['alt_aa'],
                'aa_position': row['aa_position'],
                'NTv2_delta': delta
            })
        except Exception as e:
            if len(results) < 3:
                print(f"  Error: {e}")

        if len(results) % 200 == 0 and len(results) > 0:
            elapsed = time.time() - t0
            print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s")

out = pd.DataFrame(results)
out.to_csv('benchmark_v2/results/ntv2_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
