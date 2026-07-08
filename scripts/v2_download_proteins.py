#!/usr/bin/env python3
"""Download full-length NCBI protein sequences for benchmark_v2 genes."""

import pandas as pd
import requests
import time
from Bio import SeqIO
from io import StringIO

df = pd.read_csv('benchmark_v2/data/benchmark_v2.csv')
existing = pd.read_csv('benchmark_200/data/protein_sequences.csv')
existing_genes = set(existing['gene'].values)

genes_needed = sorted(df['GeneSymbol'].unique())
genes_to_fetch = [g for g in genes_needed if g not in existing_genes]

print(f'Total genes: {len(genes_needed)}')
print(f'Already have: {len(genes_needed) - len(genes_to_fetch)}')
print(f'Need to fetch: {len(genes_to_fetch)}')

# Load gene -> transcript mapping from benchmark_v2
gene_transcripts = df.groupby('GeneSymbol')['MANE_transcript_id'].first().reset_index()
gene_transcript_map = dict(zip(gene_transcripts['GeneSymbol'], gene_transcripts['MANE_transcript_id']))

# Also load from full dataset for genes not in benchmark_v2
full = pd.read_csv('data/processed/clinvar_benchmark_full.csv')
full_gt = full.groupby('GeneSymbol')['MANE_transcript_id'].first().reset_index()
full_gt_map = dict(zip(full_gt['GeneSymbol'], full_gt['MANE_transcript_id']))
gene_transcript_map.update(full_gt_map)

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
    for g, r in failed[:10]:
        print(f'  {g}: {r}')

# Combine with existing
all_seqs = []
for _, row in existing.iterrows():
    all_seqs.append({
        'gene': row['gene'],
        'transcript': row.get('transcript', ''),
        'protein_id': row.get('protein_id', ''),
        'sequence': row['sequence'],
        'length': row['length']
    })

for gene, info in new_seqs.items():
    all_seqs.append(info)

# Also fetch existing genes that don't have full info
existing_missing = [g for g in genes_needed if g in existing_genes]
print(f'\nExisting genes: {len(existing_missing)}')

prot_df = pd.DataFrame(all_seqs)
prot_df = prot_df.drop_duplicates(subset='gene', keep='last')

# Filter to genes we need
prot_df = prot_df[prot_df['gene'].isin(genes_needed)]

print(f'\nTotal proteins: {len(prot_df)}')
print(f'Length stats:')
lengths = prot_df['length'].values
print(f'  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {sum(lengths)/len(lengths):.0f}')

under_1001 = sum(1 for l in lengths if l < 1001)
print(f'  < 1001 aa: {under_1001}/{len(lengths)}')

# Save
prot_df.to_csv('benchmark_v2/data/protein_sequences.csv', index=False)
print(f'\nSaved benchmark_v2/data/protein_sequences.csv')

# FASTA
with open('benchmark_v2/data/proteins.fasta', 'w') as f:
    for _, row in prot_df.iterrows():
        f.write(f">{row['gene']}\n{row['sequence']}\n")
print(f'Saved benchmark_v2/data/proteins.fasta')
