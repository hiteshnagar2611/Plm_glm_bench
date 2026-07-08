#!/usr/bin/env python3
"""Score benchmark_v2 with SaProt-650M using Foldseek 3Di + HuggingFace.

SaProt requires interleaved AA+3Di input. 3Di codes are derived from
OpenFold predicted structures (full-length) when available, falling back
to partial PDB structures.

3Di from Foldseek is UPPERCASE; SaProt tokenizer expects lowercase.
"""

import pandas as pd
import torch
import subprocess
import time
from pathlib import Path
from transformers import AutoModelForMaskedLM, AutoTokenizer

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v2/data/benchmark_v2.csv')
prots = pd.read_csv('benchmark_v2/data/protein_sequences.csv')
gene_seq = dict(zip(prots['gene'], prots['sequence']))
pdb_map = pd.read_csv('data/processed/pdb_mapping_all.csv')
gene_to_pdb = dict(zip(pdb_map['gene'], pdb_map['pdb_id']))

FOLDSEEK = Path('benchmark_200/tools/foldseek/bin/foldseek')
pdb_dirs = [Path('benchmark_200/data/pdb_structures'), Path('benchmark_200/data/pdb_structures_all')]
fs_dir = Path('benchmark_v2/data/saprot_3di')
fs_dir.mkdir(parents=True, exist_ok=True)
openfold_3di_dir = Path('benchmark_v2/data/alphafold_3di')

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Device: {device}")

print("Loading SaProt...")
tokenizer = AutoTokenizer.from_pretrained('westlake-repl/SaProt_650M_AF2')
model = AutoModelForMaskedLM.from_pretrained('westlake-repl/SaProt_650M_AF2').eval().to(device)

def get_3di(gene, pdb_id):
    """Get 3Di sequence, preferring OpenFold full-length over partial PDB."""
    three_di = ''
    source = 'none'

    # 1. Try OpenFold predicted 3Di (full-length)
    openfold_file = openfold_3di_dir / gene
    if openfold_file.exists():
        with open(openfold_file) as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split('\t')
                if len(parts) >= 3:
                    three_di = parts[2]
                    source = 'openfold'

    # 2. Fall back to PDB 3Di (partial)
    if not three_di and pdb_id:
        for d in pdb_dirs:
            pdb_file = d / f'{pdb_id}.pdb'
            if pdb_file.exists():
                db_path = fs_dir / pdb_id
                if not db_path.exists() or not (fs_dir / f'{pdb_id}.dbtype').exists():
                    subprocess.run([str(FOLDSEEK), 'structureto3didescriptor', str(pdb_file), str(db_path)],
                                 capture_output=True, timeout=30)
                desc_file = fs_dir / pdb_id
                if desc_file.exists():
                    with open(desc_file) as f:
                        lines = f.readlines()
                        if lines:
                            parts = lines[0].strip().split('\t')
                            if len(parts) >= 3:
                                three_di = parts[2]
                                source = 'pdb'
                break

    return three_di, source

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
pos_out = 0
unk_out = 0
source_counts = {'openfold': 0, 'pdb': 0}

for gene, group in df.groupby('GeneSymbol'):
    pdb_id = gene_to_pdb.get(gene)

    seq = gene_seq.get(gene)
    if not seq or len(seq) > 1022:
        skipped += len(group)
        continue

    three_di, source = get_3di(gene, pdb_id)
    if not three_di:
        no_3di += len(group)
        skipped += len(group)
        continue

    source_counts[source] = source_counts.get(source, 0) + 1
    pdb_len = len(three_di)

    # Truncate protein to 3Di length
    seq_trunc = seq[:pdb_len]

    # Build SaProt sequence for truncated region
    wt_saprot = make_saprot_seq(seq_trunc, three_di)
    wt_saprot_trunc = wt_saprot[:1022]

    for _, row in group.iterrows():
        pos = int(row['aa_position']) - 1
        if pos >= pdb_len:
            pos_out += 1
            skipped += 1
            continue

        alt = row['alt_aa'].upper()[0]
        ref_aa_char = seq_trunc[pos]
        di_code = three_di[pos]

        wt_token = ref_aa_char + di_code.lower()
        mut_token = alt + di_code.lower()

        wt_token_id = tokenizer.convert_tokens_to_ids(wt_token)
        mut_token_id = tokenizer.convert_tokens_to_ids(mut_token)

        if wt_token_id == tokenizer.unk_token_id or mut_token_id == tokenizer.unk_token_id:
            unk_out += 1
            skipped += 1
            continue

        wt_ids = tokenizer(wt_saprot_trunc, return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            wt_out = model(wt_ids)

        tok_pos = pos + 1  # CLS token at position 0
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
            'PDB_ID': pdb_id,
            '3Di_source': source,
            'SaProt_LLR': mut_score - wt_score
        })

    if len(results) % 100 == 0 and len(results) > 0:
        elapsed = time.time() - t0
        print(f"  {len(results)} variants, {elapsed:.0f}s, skipped={skipped} (no_3di={no_3di}, pos_out={pos_out}, unk={unk_out})")

out = pd.DataFrame(results)
out.to_csv('benchmark_v2/results/saprot_scores.csv', index=False)
print(f"\nDone: {len(out)} variants, skipped={skipped}")
print(f"  no_3di={no_3di}, pos_out_pdb={pos_out}, unk_token={unk_out}")
print(f"  3Di sources: {source_counts}")
print(f"  Time: {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
