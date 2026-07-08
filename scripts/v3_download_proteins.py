#!/usr/bin/env python3
"""Download full-length NCBI protein sequences for all 1,377 ClinVar genes.

Uses NCBI elink (NM_ -> NP_) and efetch to download protein sequences.
Filters to proteins < 1001 aa for ESM2 compatibility.
"""

import pandas as pd
import requests
import time
from pathlib import Path
from Bio import SeqIO
from io import StringIO

OUT_DIR = Path('benchmark_v3/data')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load all ClinVar missense variants
full = pd.read_csv('data/processed/clinvar_missense_protein.csv')
print(f"Loaded {len(full)} variants, {full['GeneSymbol'].nunique()} genes")

# Get gene -> transcript mapping
gene_transcripts = full.groupby('GeneSymbol')['MANE_transcript_id'].first().reset_index()
gene_transcript_map = dict(zip(gene_transcripts['GeneSymbol'], gene_transcripts['MANE_transcript_id']))

# Also check v2 data for existing mappings
v2_prots = pd.read_csv('benchmark_v2/data/protein_sequences.csv')
v2_map = {}
for _, row in v2_prots.iterrows():
    v2_map[row['gene']] = {
        'transcript': row.get('transcript', ''),
        'protein_id': row.get('protein_id', ''),
        'sequence': row['sequence'],
        'length': row['length']
    }

genes_needed = sorted(full['GeneSymbol'].unique())
print(f"Total genes: {len(genes_needed)}")

# Separate genes we already have from those we need to fetch
existing_genes = set(v2_map.keys())
genes_to_fetch = [g for g in genes_needed if g not in existing_genes]
print(f"Already have: {len(existing_genes)}")
print(f"Need to fetch: {len(genes_to_fetch)}")

new_seqs = {}
failed = []

for idx, gene in enumerate(genes_to_fetch):
    transcript = gene_transcript_map.get(gene)
    if not transcript:
        failed.append((gene, 'no transcript'))
        continue

    base_accession = transcript.split('.')[0]

    # Step 1: elink to get protein accession
    try:
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'nucleotide',
            'db': 'protein',
            'id': base_accession,
            'linkname': 'nuccore_protein',
            'retmode': 'json'
        }
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            failed.append((gene, f'HTTP {resp.status_code}'))
            continue

        data = resp.json()
        linksets = data.get('linksets', [])
        if not linksets or 'linksetdbs' not in linksets[0]:
            failed.append((gene, 'no linksetdbs'))
            continue

        links = linksets[0]['linksetdbs'][0].get('links', [])
        if not links:
            failed.append((gene, 'no protein links'))
            continue

        protein_id = str(links[0])

        # Step 2: efetch protein sequence
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params2 = {
            'db': 'protein',
            'id': protein_id,
            'rettype': 'fasta',
            'retmode': 'text'
        }
        resp2 = requests.get(url2, params=params2, timeout=30)
        if resp2.status_code == 200 and resp2.text.strip():
            records = list(SeqIO.parse(StringIO(resp2.text), "fasta"))
            if records:
                seq = str(records[0].seq)
                new_seqs[gene] = {
                    'gene': gene,
                    'transcript': transcript,
                    'protein_id': protein_id,
                    'sequence': seq,
                    'length': len(seq)
                }

    except Exception as e:
        failed.append((gene, str(e)))

    if (idx + 1) % 50 == 0:
        print(f'  Fetched {idx+1}/{len(genes_to_fetch)}, found {len(new_seqs)}, failed {len(failed)}')

    time.sleep(0.35)

print(f'\nFetched: {len(new_seqs)}')
print(f'Failed: {len(failed)}')
if failed:
    print('Failed genes:')
    for g, r in failed[:20]:
        print(f'  {g}: {r}')

# Combine with existing
all_seqs = []
for gene in genes_needed:
    if gene in v2_map:
        info = v2_map[gene]
        all_seqs.append({
            'gene': gene,
            'transcript': info['transcript'],
            'protein_id': info['protein_id'],
            'sequence': info['sequence'],
            'length': info['length']
        })
    elif gene in new_seqs:
        all_seqs.append(new_seqs[gene])

prot_df = pd.DataFrame(all_seqs)
print(f'\nTotal proteins: {len(prot_df)}')
print(f'Length stats:')
lengths = prot_df['length'].values
print(f'  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}')

under_1001 = sum(1 for l in lengths if l < 1001)
print(f'  < 1001 aa: {under_1001}/{len(lengths)}')

# Save all (before filtering)
prot_df.to_csv(OUT_DIR / 'protein_sequences_all.csv', index=False)
print(f'\nSaved {OUT_DIR}/protein_sequences_all.csv ({len(prot_df)} proteins)')

# Filter to < 1001 aa
prot_filtered = prot_df[prot_df['length'] < 1001].copy()
prot_filtered.to_csv(OUT_DIR / 'protein_sequences.csv', index=False)
print(f'Saved {OUT_DIR}/protein_sequences.csv ({len(prot_filtered)} proteins, < 1001 aa)')

# FASTA
with open(OUT_DIR / 'proteins.fasta', 'w') as f:
    for _, row in prot_filtered.iterrows():
        f.write(f">{row['gene']}\n{row['sequence']}\n")
print(f'Saved {OUT_DIR}/proteins.fasta')

# Summary
kept_genes = set(prot_filtered['gene'])
filtered_out = set(prot_df['gene']) - kept_genes
print(f'\nSummary:')
print(f'  Genes kept: {len(kept_genes)}')
print(f'  Genes filtered out (> 1001 aa): {len(filtered_out)}')
if filtered_out:
    long_genes = prot_df[prot_df['gene'].isin(filtered_out)].sort_values('length', ascending=False)
    print(f'  Top 10 longest filtered genes:')
    for _, row in long_genes.head(10).iterrows():
        print(f'    {row["gene"]:12s} {row["length"]:5d} aa')
