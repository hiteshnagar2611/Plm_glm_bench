#!/usr/bin/env python3
"""Score benchmark_v2 variants with ESM3 using BOTH sequence + structure input.
Uses AlphaFold DB predicted structures (pLDDT > 70)."""

import os, time
os.environ['HF_TOKEN'] = os.environ.get('HF_TOKEN', '')

import torch
import pandas as pd
from pathlib import Path
from huggingface_hub import login
login(token=os.environ['HF_TOKEN'])
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
BENCH_DIR = BASE_DIR / 'benchmark_v2'
RESULTS_DIR = BENCH_DIR / 'results'
LOG = RESULTS_DIR / 'esm3_struct_log.txt'

def log(msg):
    print(msg, flush=True)
    with open(LOG, 'a') as f:
        f.write(msg + '\n')

log("Loading ESM3 model...")
t0 = time.time()
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESM3.from_pretrained('esm3_sm_open_v1').to('cpu', dtype=torch.float32)
log(f"Model loaded in {time.time()-t0:.1f}s")

tok = model.tokenizers.sequence
config = LogitsConfig(sequence=True)
AA_TO_ID = {aa: tok.encode(aa)[1] for aa in 'ACDEFGHIKLMNPQRSTVWY'}

def parse_pdb_backbone(pdb_path):
    residues = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                atom = line[12:16].strip()
                resn = int(line[22:26].strip())
                if atom in ('N', 'CA', 'C'):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    b = float(line[60:66])
                    if resn not in residues:
                        residues[resn] = {'plddt': b}
                    residues[resn][atom] = [x, y, z]
    coords = []
    plddt = []
    for resn in sorted(residues.keys()):
        r = residues[resn]
        if 'N' in r and 'CA' in r and 'C' in r:
            coords.append([r['N'], r['CA'], r['C']])
            plddt.append(r['plddt'])
    return torch.tensor(coords, dtype=torch.float32), torch.tensor(plddt, dtype=torch.float32)

def score_variant_struct(wt_seq, mut_seq, pdb_path):
    with torch.no_grad():
        # WT
        wt_coords, wt_plddt = parse_pdb_backbone(pdb_path)
        n = min(len(wt_seq), len(wt_coords))
        wt_prot = ESMProtein(sequence=wt_seq[:n], coordinates=wt_coords[:n], plddt=wt_plddt[:n])
        wt_enc = model.encode(wt_prot)
        wt_out = model.logits(wt_enc, config)
        wt_lp = torch.nn.functional.log_softmax(wt_out.logits.sequence.float(), dim=-1)

        # MUT
        mut_coords, mut_plddt = parse_pdb_backbone(pdb_path)
        n2 = min(len(mut_seq), len(mut_coords))
        mut_prot = ESMProtein(sequence=mut_seq[:n2], coordinates=mut_coords[:n2], plddt=mut_plddt[:n2])
        mut_enc = model.encode(mut_prot)
        mut_out = model.logits(mut_enc, config)
        mut_lp = torch.nn.functional.log_softmax(mut_out.logits.sequence.float(), dim=-1)

    return wt_lp[0], mut_lp[0]

def score_variant_seqonly(wt_seq, mut_seq):
    with torch.no_grad():
        wt_enc = model.encode(ESMProtein(sequence=wt_seq))
        wt_out = model.logits(wt_enc, config)
        wt_lp = torch.nn.functional.log_softmax(wt_out.logits.sequence.float(), dim=-1)

        mut_enc = model.encode(ESMProtein(sequence=mut_seq))
        mut_out = model.logits(mut_enc, config)
        mut_lp = torch.nn.functional.log_softmax(mut_out.logits.sequence.float(), dim=-1)

    return wt_lp[0], mut_lp[0]

benchmark = pd.read_csv(BENCH_DIR / 'data' / 'benchmark_v2.csv')
benchmark['variant_id'] = benchmark['GeneSymbol'] + '_' + benchmark['aa_position'].astype(int).astype(str) + '_' + benchmark['ref_aa'] + '_' + benchmark['alt_aa']
benchmark['aa_pos'] = benchmark['aa_position'].astype(int)
benchmark['gene'] = benchmark['GeneSymbol']
benchmark['label'] = benchmark['ClinVar_label'].str.lower()

AA3_TO_1 = {'Ala':'A','Arg':'R','Asn':'N','Asp':'D','Cys':'C','Glu':'E','Gln':'Q',
             'Gly':'G','His':'H','Ile':'I','Leu':'L','Lys':'K','Met':'M','Phe':'F',
             'Pro':'P','Ser':'S','Thr':'T','Trp':'W','Tyr':'Y','Val':'V','Ter':'*'}
benchmark['ref_aa_1'] = benchmark['ref_aa'].map(lambda x: AA3_TO_1.get(x, x))
benchmark['alt_aa_1'] = benchmark['alt_aa'].map(lambda x: AA3_TO_1.get(x, x))

proteins = pd.read_csv(BENCH_DIR / 'data' / 'protein_sequences.csv')
protein_map = dict(zip(proteins['gene'], proteins['sequence']))
pdb_dir = BENCH_DIR / 'data' / 'alphafold_structures'
log(f"Loaded {len(benchmark)} variants, {len(protein_map)} proteins, PDB dir exists: {pdb_dir.exists()}")

results = []
t_start = time.time()
for idx, row in benchmark.iterrows():
    gene = row['gene']
    aa_pos = int(row['aa_pos'])
    alt_aa = row['alt_aa_1']
    label = row['label']

    if gene not in protein_map:
        continue
    wt_seq = protein_map[gene]
    if aa_pos < 1 or aa_pos > len(wt_seq):
        continue
    if alt_aa not in AA_TO_ID:
        continue

    mut_seq = wt_seq[:aa_pos-1] + alt_aa + wt_seq[aa_pos:]
    pdb_path = pdb_dir / f'{gene}.pdb'

    try:
        t_v = time.time()
        if pdb_path.exists():
            wt_lp, mut_lp = score_variant_struct(wt_seq, mut_seq, pdb_path)
        else:
            wt_lp, mut_lp = score_variant_seqonly(wt_seq, mut_seq)

        # BOS at index 0, protein pos = index
        wt_logprob = wt_lp[aa_pos, AA_TO_ID[wt_seq[aa_pos-1]]].item()
        mut_logprob = mut_lp[aa_pos, AA_TO_ID[alt_aa]].item()
        llr = mut_logprob - wt_logprob
        dt = time.time() - t_v
        used_struct = pdb_path.exists()
    except Exception as e:
        log(f"  Error {gene} pos {aa_pos}: {e}")
        continue

    results.append({
        'variant_id': row['variant_id'],
        'gene': gene,
        'aa_pos': aa_pos,
        'ref_aa': row['ref_aa_1'],
        'alt_aa': alt_aa,
        'label': label,
        'llr': llr,
        'used_structure': used_struct,
    })

    if len(results) % 10 == 0 or len(results) <= 3:
        elapsed = time.time() - t_start
        rate = len(results) / elapsed
        eta = (793 - len(results)) / rate / 60 if rate > 0 else 0
        log(f"  [{len(results)}/793] {gene} {row['ref_aa_1']}{aa_pos}{alt_aa} LLR={llr:.4f} [{'struct' if used_struct else 'seq'}] [{dt:.1f}s] ETA={eta:.0f}min")

df = pd.DataFrame(results)
df.to_csv(RESULTS_DIR / 'esm3_struct_scores.csv', index=False)
total = time.time() - t_start
n_struct = df['used_structure'].sum()
log(f"\nDone! {len(df)} variants ({n_struct} with structure) in {total:.0f}s ({total/60:.1f} min)")
log(f"Pathogenic mean: {df[df['label']=='pathogenic']['llr'].mean():.4f}")
log(f"Benign mean: {df[df['label']=='benign']['llr'].mean():.4f}")
