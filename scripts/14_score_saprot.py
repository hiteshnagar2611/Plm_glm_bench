#!/usr/bin/env python3
"""Score variants using SaProt 650M (structure-aware protein language model).

Uses HuggingFace transformers to load westlake-repl/SaProt_650M_AF2.
Requires Foldseek to generate 3Di structure sequences from PDB files.

SaProt input format: interleaved AA + lowercase(3Di) codes.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = 'westlake-repl/SaProt_650M_AF2'
FOLDSEEK_BIN = 'benchmark_200/tools/foldseek/bin/foldseek'
PDB_DIR = 'benchmark_200/data/pdb_structures'
DATA_DIR = 'benchmark_200/data'
RESULTS_DIR = 'benchmark_200/results'
LOG_FILE = os.path.join(RESULTS_DIR, '14_score_saprot_log.txt')

AA_3to1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
}


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()


def extract_3di_from_pdb(pdb_path):
    """Run Foldseek structureto3didescriptor on a PDB file."""
    tmp_out = pdb_path + '.3di.txt'
    try:
        result = subprocess.run(
            [FOLDSEEK_BIN, 'structureto3didescriptor', pdb_path, tmp_out],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            return None, None

        with open(tmp_out, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3 and len(parts[1]) == len(parts[2]):
                    aa_seq = parts[1]
                    s3di_seq = parts[2]
                    if len(aa_seq) > 0 and all(c in 'ARNDCEQGHILKMFPSTWYV' for c in aa_seq):
                        return aa_seq, s3di_seq
        return None, None
    except Exception:
        return None, None
    finally:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)


def build_structure_dict(pdb_dir):
    """Build gene → (aa_seq, s3di_seq) dict from PDB files."""
    struct_dict = {}
    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith('.pdb')])

    log(f'Extracting 3Di sequences from {len(pdb_files)} PDB files...')
    errors = 0
    for i, pdb_file in enumerate(pdb_files):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        aa_seq, s3di_seq = extract_3di_from_pdb(pdb_path)
        if aa_seq is not None:
            pdb_id = pdb_file.replace('.pdb', '')
            struct_dict[pdb_id] = (aa_seq, s3di_seq)
        else:
            errors += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(pdb_files):
            log(f'  [{i+1}/{len(pdb_files)}] Extracted: {len(struct_dict)}, Errors: {errors}')

    return struct_dict


def make_saprot_seq(aa_seq, s3di_seq, aa_pos, alt_aa_1):
    """Create SaProt wildtype and mutant interleaved sequences."""
    if aa_pos >= len(aa_seq) or aa_pos >= len(s3di_seq):
        return None, None

    wt_parts = []
    mut_parts = []
    for i in range(len(aa_seq)):
        s3di_char = s3di_seq[i].lower() if s3di_seq[i] != '-' else '#'
        wt_parts.append(aa_seq[i] + s3di_char)
        if i == aa_pos:
            mut_parts.append(alt_aa_1 + s3di_char)
        else:
            mut_parts.append(aa_seq[i] + s3di_char)

    return ''.join(wt_parts), ''.join(mut_parts)


def score_variant(model, tokenizer, wt_seq, mut_seq, device='mps'):
    """Compute log-likelihood ratio for a variant using token-level comparison."""
    max_len = 1024  # SaProt max length

    with torch.no_grad():
        wt_inputs = tokenizer(wt_seq, return_tensors='pt', truncation=True, max_length=max_len)
        wt_inputs = {k: v.to(device) for k, v in wt_inputs.items()}
        wt_logits = model(**wt_inputs).logits

        mut_inputs = tokenizer(mut_seq, return_tensors='pt', truncation=True, max_length=max_len)
        mut_inputs = {k: v.to(device) for k, v in mut_inputs.items()}
        mut_logits = model(**mut_inputs).logits

    wt_tokens = wt_inputs['input_ids'][0].tolist()
    mut_tokens = mut_inputs['input_ids'][0].tolist()

    # Find first position where tokens differ (skip CLS=0 and EOS)
    mut_token_pos = None
    for i in range(1, min(len(wt_tokens), len(mut_tokens)) - 1):
        if wt_tokens[i] != mut_tokens[i]:
            mut_token_pos = i
            break

    if mut_token_pos is None:
        return 0.0

    wt_log_probs = torch.log_softmax(wt_logits[0, mut_token_pos], dim=-1)
    mut_log_probs = torch.log_softmax(mut_logits[0, mut_token_pos], dim=-1)

    wt_ll = wt_log_probs[wt_tokens[mut_token_pos]].item()
    mut_ll = mut_log_probs[mut_tokens[mut_token_pos]].item()

    return mut_ll - wt_ll


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Clear old log
    with open(LOG_FILE, 'w') as f:
        pass

    log('SaProt 650M Variant Scoring')
    log('=' * 60)

    # Load variant data
    log('Loading variant data...')
    full_data = pd.read_csv(os.path.join(DATA_DIR, 'clinvar_200_full.csv'))
    pdb_map = pd.read_csv(os.path.join(DATA_DIR, 'pdb_mapping.csv'))

    gene_to_pdb = dict(zip(pdb_map['gene'], pdb_map['pdb_id']))
    log(f'  Variants: {len(full_data)}')
    log(f'  Genes with PDB: {len(gene_to_pdb)}')

    # Extract 3Di sequences from PDB files
    struct_dict = build_structure_dict(PDB_DIR)
    log(f'  Structures extracted: {len(struct_dict)}')

    # Build gene → structure mapping
    gene_struct = {}
    for gene, pdb_id in gene_to_pdb.items():
        base_id = pdb_id.split('_')[0] if '_' in pdb_id else pdb_id
        if base_id in struct_dict:
            gene_struct[gene] = struct_dict[base_id]
        elif pdb_id in struct_dict:
            gene_struct[gene] = struct_dict[pdb_id]

    log(f'  Genes with structure data: {len(gene_struct)}')

    # Load model
    log(f'Loading SaProt model: {MODEL_ID}...')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log(f'  Using device: {device}')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()
    log(f'  Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)')

    # Score variants
    log(f'\nScoring variants...')
    results = []
    errors = 0
    skipped_no_struct = 0
    skipped_bad_aa = 0
    t_start = time.time()
    last_log_time = t_start

    for idx, row in full_data.iterrows():
        gene = row['GeneSymbol']
        variation_id = row['VariationID']

        # Get structure data
        if gene not in gene_struct:
            skipped_no_struct += 1
            continue

        aa_seq, s3di_seq = gene_struct[gene]

        # Get amino acid info
        ref_aa_3 = row['ref_aa']
        alt_aa_3 = row['alt_aa']
        aa_pos = int(row['aa_position']) - 1  # Convert to 0-indexed

        ref_aa_1 = AA_3to1.get(ref_aa_3)
        alt_aa_1 = AA_3to1.get(alt_aa_3)

        if ref_aa_1 is None or alt_aa_1 is None:
            skipped_bad_aa += 1
            continue

        # Verify reference matches
        if aa_pos < len(aa_seq) and aa_seq[aa_pos] != ref_aa_1:
            found = False
            for offset in range(-3, 4):
                check_pos = aa_pos + offset
                if 0 <= check_pos < len(aa_seq) and aa_seq[check_pos] == ref_aa_1:
                    aa_pos = check_pos
                    found = True
                    break
            if not found:
                skipped_bad_aa += 1
                continue

        # Create SaProt sequences
        wt_seq, mut_seq = make_saprot_seq(aa_seq, s3di_seq, aa_pos, alt_aa_1)
        if wt_seq is None:
            errors += 1
            continue

        # Score
        try:
            llr = score_variant(model, tokenizer, wt_seq, mut_seq, device=device)
        except Exception as e:
            errors += 1
            if errors <= 3:
                log(f'  Error scoring {variation_id}: {e}')
            continue

        results.append({
            'VariationID': variation_id,
            'GeneSymbol': gene,
            'Chromosome': row['Chromosome'],
            'PositionVCF': row['PositionVCF'],
            'ReferenceAlleleVCF': row['ReferenceAlleleVCF'],
            'AlternateAlleleVCF': row['AlternateAlleleVCF'],
            'ClinVar_label': row['ClinVar_label'],
            'ref_aa': ref_aa_3,
            'alt_aa': alt_aa_3,
            'aa_position': row['aa_position'],
            'SaProt_LLR': llr,
        })

        n_scored = len(results)
        now = time.time()
        if n_scored % 100 == 0 or (now - last_log_time) > 30:
            elapsed = now - t_start
            rate = n_scored / elapsed if elapsed > 0 else 0
            remaining = len(full_data) - idx - 1
            eta = remaining / rate / 3600 if rate > 0 else 0
            log(f'  [{n_scored}/{len(full_data)}] Scored: {n_scored}, '
                f'Errors: {errors}, No struct: {skipped_no_struct} | {rate:.2f} vars/s (ETA: {eta:.1f}h)')
            last_log_time = now

    elapsed = time.time() - t_start
    log(f'\nCompleted in {elapsed:.0f}s')
    log(f'  Scored: {len(results)}')
    log(f'  Errors: {errors}')
    log(f'  No structure: {skipped_no_struct}')
    log(f'  Bad AA: {skipped_bad_aa}')

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR, 'saprot_scores.csv')
        results_df.to_csv(out_path, index=False)
        log(f'\nSaved: {out_path}')

        from scipy import stats as sp_stats
        path_scores = results_df[results_df['ClinVar_label'] == 1]['SaProt_LLR']
        ben_scores = results_df[results_df['ClinVar_label'] == 0]['SaProt_LLR']
        log(f'  Pathogenic mean LLR: {path_scores.mean():.4f}')
        log(f'  Benign mean LLR: {ben_scores.mean():.4f}')
        r, p = sp_stats.spearmanr(results_df['SaProt_LLR'], results_df['ClinVar_label'])
        log(f'  Spearman r: {r:.4f} (p={p:.2e})')
    else:
        log('No results to save!')


if __name__ == '__main__':
    main()
