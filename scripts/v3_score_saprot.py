#!/usr/bin/env python3
"""Score benchmark_v3 with SaProt-650M using Foldseek 3Di + HuggingFace.

SaProt requires interleaved AA+3Di input. 3Di codes are derived from
AlphaFold predicted structures (full-length).
3Di from Foldseek is UPPERCASE; SaProt tokenizer expects lowercase.
"""

import pandas as pd
import torch
import subprocess
import time
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
prots = pd.read_csv('benchmark_v3/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))

FOLDSEEK = Path('benchmark_200/tools/foldseek/bin/foldseek')
af_3di_dir = Path('benchmark_v3/data/alphafold_3di')
saprot_3di_dir = Path('benchmark_v3/data/saprot_3di')
saprot_3di_dir.mkdir(parents=True, exist_ok=True)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading SaProt...")
tokenizer = AutoTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
model = AutoModelForMaskedLM.from_pretrained('westlake-repl/SaProt_650M_AF2').eval().to(device)

def get_3di(gene):
    """Get 3Di sequence from AlphaFold 3Di directory."""
    three_di_file = af_3di_dir / gene
    if three_di_file.exists():
        with open(three_di_file) as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split('\t')
                if len(parts) >= 3:
                    return parts[2]
    return ''

def make_saprot_seq(aa_seq, three_di):
    """Create SaProt interleaved AA+3Di sequence (3Di must be lowercase)."""
    result = []
    for i, aa in enumerate(aa_seq):
        result.append(aa)
        if i < len(three_di):
            result.append(three_di[i].lower())
        else:
            result.append('o')
    return ''.join(result)

results = []
t0 = time.time()
skipped = 0
no_3di = 0
source_counts = {}

for gene, group in df.groupby('GeneSymbol'):
    seq = gene_seq.get(gene)
    if not seq or len(seq) > 1022:
        skipped += len(group)
        continue

    three_di = get_3di(gene)
    if not three_di:
        no_3di += len(group)
        skipped += len(group)
        continue

    pdb_len = len(three_di)
    seq_len = len(seq)
    effective_len = min(pdb_len, seq_len)
    seq_trunc = seq[:effective_len]
    three_di_trunc = three_di[:effective_len]
    wt_saprot = make_saprot_seq(seq_trunc, three_di_trunc)
    wt_saprot_trunc = wt_saprot[:1022]

    for _, row in group.iterrows():
        pos = int(row['aa_position']) - 1
        if pos >= effective_len or pos < 0:
            skipped += 1
            continue

        alt = row['alt_aa'].upper()[0]
        ref_aa_char = seq_trunc[pos]
        di_code = three_di_trunc[pos]

        wt_token = ref_aa_char + di_code.lower()
        mut_token = alt + di_code.lower()

        wt_token_id = tokenizer.convert_tokens_to_ids(wt_token)
        mut_token_id = tokenizer.convert_tokens_to_ids(mut_token)

        if wt_token_id == tokenizer.unk_token_id or mut_token_id == tokenizer.unk_token_id:
            skipped += 1
            continue

        wt_ids = tokenizer(wt_saprot_trunc, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            wt_out = model(wt_ids)

        tok_pos = pos + 1
        if tok_pos >= wt_out.logits.shape[1]:
            skipped += 1
            continue

        logits = wt_out.logits[0, tok_pos]
        ll = torch.log_softmax(logits, dim=-1)

        wt_score = ll[wt_token_id].item()
        mut_score = ll[mut_token_id].item()

        results.append({
            'VariationID': row['VariationID'],
            'GeneSymbol': gene,
            'ClinVar_label': row['ClinVar_label'],
            'ref_aa': row['ref_aa'],
            'alt_aa': row['alt_aa'],
            'aa_position': row['aa_position'],
            'SaProt_LLR': mut_score - wt_score
        })

    if len(results) % 500 == 0 and len(results) > 0:
        elapsed = time.time() - t0
        print(f"  {len(results)} variants, {elapsed:.0f}s, skipped={skipped}")

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/saprot_scores.csv', index=False)
print(f"\nDone: {len(out)} variants, skipped={skipped}")
print(f"  Time: {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
