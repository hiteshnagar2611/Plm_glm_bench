#!/usr/bin/env python3
"""Download AlphaFold DB structures for all benchmark_v3 genes.

Uses AlphaFold API to get the correct version URL for each protein.
"""

import pandas as pd
import requests
import time
import shutil
from pathlib import Path

OUT_DIR = Path('benchmark_v3/data')
AF_DIR = OUT_DIR / 'alphafold_structures'
AF_DIR.mkdir(parents=True, exist_ok=True)

prots = pd.read_csv(OUT_DIR / 'protein_sequences.csv')
genes = sorted(prots['gene'].unique())
print(f"Genes: {len(genes)}")

# Step 1: Copy from v2
v2_af_dir = Path('benchmark_v2/data/alphafold_structures')
copied = 0
if v2_af_dir.exists():
    for gene in genes:
        src = v2_af_dir / f"{gene}.pdb"
        dst = AF_DIR / f"{gene}.pdb"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
print(f"Copied from v2: {copied}")

# Step 2: Load UniProt mappings
mapping_file = OUT_DIR / 'gene_to_uniprot.csv'
if mapping_file.exists():
    mapping_df = pd.read_csv(mapping_file)
    gene_to_uniprot = dict(zip(mapping_df['gene'], mapping_df['uniprot']))
else:
    gene_to_uniprot = {}

print(f"Known UniProt accessions: {len(gene_to_uniprot)}")

# Step 3: Download remaining
to_download = [g for g in genes if not (AF_DIR / f"{g}.pdb").exists()]
print(f"Need to download: {len(to_download)}")

downloaded = 0
failed = 0

for idx, gene in enumerate(to_download):
    uniprot = gene_to_uniprot.get(gene)

    if not uniprot:
        # Try API to find it
        try:
            api_url = f"https://rest.uniprot.org/uniprotkb/search?query=(gene:{gene} AND organism_id:9606)&format=json&size=1"
            resp = requests.get(api_url, timeout=30)
            if resp.status_code == 200:
                results = resp.json().get('results', [])
                if results:
                    uniprot = results[0].get('primaryAccession', '')
        except:
            pass

    if not uniprot:
        failed += 1
        continue

    # Use API to get correct PDB URL (handles versioning)
    try:
        api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
        resp = requests.get(api_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                pdb_url = data[0].get('pdbUrl', '')
                if pdb_url:
                    pdb_resp = requests.get(pdb_url, timeout=60)
                    if pdb_resp.status_code == 200 and 'ATOM' in pdb_resp.text:
                        with open(AF_DIR / f"{gene}.pdb", 'w') as f:
                            f.write(pdb_resp.text)
                        downloaded += 1
                    else:
                        failed += 1
                else:
                    failed += 1
            else:
                failed += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

    if (idx + 1) % 50 == 0:
        print(f"  {idx+1}/{len(to_download)}: downloaded={downloaded}, failed={failed}")

    time.sleep(0.2)

total = len(list(AF_DIR.glob('*.pdb')))
print(f"\nDone: downloaded={downloaded}, failed={failed}")
print(f"Total structures: {total}")
