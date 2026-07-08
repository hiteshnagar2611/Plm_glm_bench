#!/usr/bin/env python3
"""Score variants using HyenaDNA 150M (genome language model).

Uses multimolecule package to load HyenaDNA medium CausalLM model.
Computes log-likelihood ratio for each variant using 6001bp DNA context.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import torch

# Import multimolecule FIRST to register architecture
import multimolecule
from multimolecule import DnaTokenizer, HyenaDnaForCausalLM

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_ID = 'multimolecule/hyenadna-medium'
DATA_DIR = 'benchmark_200/data'
RESULTS_DIR = 'benchmark_200/results'
LOG_FILE = os.path.join(RESULTS_DIR, '15_score_hyenadna_log.txt')


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def score_variant(model, tokenizer, wt_seq, mut_seq, variant_pos, device='mps'):
    """Compute log-likelihood ratio for a variant.

    variant_pos: 0-indexed position of the variant in the sequence (0..len-1)
    DnaTokenizer adds CLS (pos 0) and EOS, so token_pos = variant_pos + 1
    """
    max_len = 6003  # CLS + 6001 DNA bases + EOS

    # Tokenize both sequences
    wt_tokens = tokenizer(wt_seq, return_tensors='pt', truncation=True, max_length=max_len)
    mut_tokens = tokenizer(mut_seq, return_tensors='pt', truncation=True, max_length=max_len)

    wt_input = {k: v.to(device) for k, v in wt_tokens.items()}
    mut_input = {k: v.to(device) for k, v in mut_tokens.items()}

    with torch.no_grad():
        wt_logits = model(**wt_input).logits  # (1, seq_len, vocab_size)
        mut_logits = model(**mut_input).logits

    input_ids_wt = wt_input['input_ids'][0]
    input_ids_mut = mut_input['input_ids'][0]

    # DnaTokenizer prepends CLS token, so DNA base at position N is at token N+1
    token_pos = variant_pos + 1
    if token_pos >= len(input_ids_wt) or token_pos >= len(input_ids_mut):
        return 0.0

    wt_base = input_ids_wt[token_pos].item()
    mut_base = input_ids_mut[token_pos].item()

    if wt_base == mut_base:
        return 0.0

    # Log-likelihood at variant position
    wt_log_probs = torch.log_softmax(wt_logits[0, token_pos], dim=-1)
    mut_log_probs = torch.log_softmax(mut_logits[0, token_pos], dim=-1)

    wt_ll = wt_log_probs[wt_base].item()
    mut_ll = mut_log_probs[mut_base].item()

    return mut_ll - wt_ll


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log('HyenaDNA 150M Variant Scoring')
    log('=' * 60)

    # Load variant data
    log('Loading variant data...')
    dna_data = pd.read_csv(os.path.join(DATA_DIR, 'dna_sequences.csv'))
    log(f'  DNA sequences: {len(dna_data)}')

    # Load model
    log(f'Loading HyenaDNA model: {MODEL_ID}...')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    log(f'  Using device: {device}')

    tokenizer = DnaTokenizer.from_pretrained(MODEL_ID)
    model = HyenaDnaForCausalLM.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()
    log(f'  Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)')

    # Test tokenizer
    test_seq = dna_data.iloc[0]['wildtype_sequence']
    test_tokens = tokenizer(test_seq, return_tensors='pt', truncation=True, max_length=6003)
    log(f'  Test tokenization: {len(test_seq)}bp → {test_tokens["input_ids"].shape[1]} tokens')

    # Score variants
    log(f'\nScoring {len(dna_data)} variants...')
    results = []
    errors = 0
    t_start = time.time()

    for idx, row in dna_data.iterrows():
        variation_id = row['VariationID']
        wt_seq = str(row['wildtype_sequence'])
        mut_seq = str(row['mutant_sequence'])
        variant_pos = int(row['variant_relative_pos'])

        # Validate
        if len(wt_seq) != len(mut_seq):
            errors += 1
            continue
        if variant_pos < 0 or variant_pos >= len(wt_seq):
            errors += 1
            continue
        if wt_seq[variant_pos] == mut_seq[variant_pos]:
            errors += 1
            continue

        # Score
        try:
            llr = score_variant(model, tokenizer, wt_seq, mut_seq, variant_pos, device=device)
        except Exception as e:
            errors += 1
            continue

        results.append({
            'VariationID': variation_id,
            'GeneSymbol': row['GeneSymbol'],
            'Chromosome': row['Chromosome'],
            'PositionVCF': row['PositionVCF'],
            'ReferenceAlleleVCF': row['ReferenceAlleleVCF'],
            'AlternateAlleleVCF': row['AlternateAlleleVCF'],
            'ClinVar_label': row['ClinVar_label'],
            'HyenaDNA_LLR': llr,
        })

        n_scored = len(results)
        if n_scored % 200 == 0 or n_scored == 1:
            elapsed = time.time() - t_start
            rate = n_scored / elapsed if elapsed > 0 else 0
            eta = (len(dna_data) - n_scored) / rate if rate > 0 else 0
            log(f'  [{n_scored}/{len(dna_data)}] Scored: {n_scored}, '
                f'Errors: {errors} | {rate:.2f} vars/s (ETA: {eta/3600:.1f}h)')

    elapsed = time.time() - t_start
    log(f'\nCompleted in {elapsed:.0f}s')
    log(f'  Scored: {len(results)}')
    log(f'  Errors: {errors}')

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        out_path = os.path.join(RESULTS_DIR, 'hyena_scores.csv')
        results_df.to_csv(out_path, index=False)
        log(f'\nSaved: {out_path}')

        # Quick stats
        from scipy import stats as sp_stats
        path_scores = results_df[results_df['ClinVar_label'] == 1]['HyenaDNA_LLR']
        ben_scores = results_df[results_df['ClinVar_label'] == 0]['HyenaDNA_LLR']
        log(f'  Pathogenic mean LLR: {path_scores.mean():.4f}')
        log(f'  Benign mean LLR: {ben_scores.mean():.4f}')
        r, p = sp_stats.spearmanr(results_df['HyenaDNA_LLR'], results_df['ClinVar_label'])
        log(f'  Spearman r: {r:.4f} (p={p:.2e})')
    else:
        log('No results to save!')


if __name__ == '__main__':
    main()
