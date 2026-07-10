#!/usr/bin/env python3
"""Score benchmark_v3 with NT-v2-500M: max pool + cosine similarity.

Uses HuggingFace transformers directly (no local nt_pkg dependency).
"""

import pandas as pd
import torch
import torch.nn.functional as F
import time
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForMaskedLM, AutoTokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
dna = pd.read_csv('benchmark_v3/data/dna_sequences.csv')
dna_map = dict(zip(dna['VariationID'], dna['seq']))

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading NT-v2-500M from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained(
    'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
    trust_remote_code=True
)
model = AutoModelForMaskedLM.from_pretrained(
    'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species',
    trust_remote_code=True
).eval().to(device)
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
            wt_tokens = tokenizer(ref_seq, return_tensors='pt', truncation=True, max_length=2048, padding='max_length')
            mut_tokens = tokenizer(alt_seq, return_tensors='pt', truncation=True, max_length=2048, padding='max_length')

            wt_tokens = {k: v.to(device) for k, v in wt_tokens.items()}
            mut_tokens = {k: v.to(device) for k, v in mut_tokens.items()}

            wt_out = model(**wt_tokens, output_hidden_states=True)
            mut_out = model(**mut_tokens, output_hidden_states=True)

            wt_hidden = wt_out.hidden_states[-1][0]   # [seq_len, hidden]
            mut_hidden = mut_out.hidden_states[-1][0]

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
                'NTv2_pool': score
            })
        except Exception as e:
            if len(results) < 3:
                print(f"  Error: {e}")

        if len(results) % 500 == 0 and len(results) > 0:
            elapsed = time.time() - t0
            print(f"  {len(results)} variants, {elapsed:.0f}s, {len(results)/elapsed:.1f} vars/s", flush=True)

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/ntv2_pool_scores.csv', index=False)
print(f"\nDone: {len(out)} variants in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
