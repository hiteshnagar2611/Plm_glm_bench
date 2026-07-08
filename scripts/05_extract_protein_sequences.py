#!/usr/bin/env python3
"""Extract protein sequences for the 200 selected genes.

Uses NCBI efetch to download protein sequences from RefSeq.
"""

import pandas as pd
import requests
import time
import os
import sys
from Bio import SeqIO
from io import StringIO

def get_protein_accession(transcript_id):
    """Convert NM_ accession to NP_ protein accession via NCBI."""
    # Remove version
    base = transcript_id.split('.')[0]
    # For MANE transcripts, the protein accession is NP_ version
    # We need to query NCBI to get the right one
    return base

def fetch_protein_sequence(protein_id):
    """Fetch protein sequence from NCBI."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'protein',
        'id': protein_id,
        'rettype': 'fasta',
        'retmode': 'text'
    }
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 200 and resp.text.strip():
        records = list(SeqIO.parse(StringIO(resp.text), "fasta"))
        if records:
            return str(records[0].seq)
    return None

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/data/protein_sequences"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading variant data...")
    df = pd.read_csv(os.path.join(data_dir, "clinvar_200_full.csv"))

    # Get unique genes and their transcripts
    gene_transcripts = df.groupby('GeneSymbol')['MANE_transcript_id'].first().reset_index()
    print(f"  Total genes: {len(gene_transcripts)}")

    # For each transcript, we need to find the protein accession
    # MANE transcripts are RefSeq NM_ accessions, protein is NP_
    # We'll use NCBI's elink to find the protein ID

    print("\nFetching protein accessions from NCBI...")
    protein_seqs = {}
    failed = []

    for idx, row in gene_transcripts.iterrows():
        gene = row['GeneSymbol']
        transcript = row['MANE_transcript_id']
        base_accession = transcript.split('.')[0]

        # Use efetch to get the protein product
        # For RefSeq, we can directly construct NP_ from NM_ using eutils
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"
        params = {
            'dbfrom': 'nucleotide',
            'db': 'protein',
            'id': base_accession,
            'linkname': 'nuccore_protein',
            'retmode': 'json'
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                linksets = data.get('linksets', [])
                if linksets and 'linksetdbs' in linksets[0]:
                    links = linksets[0]['linksetdbs'][0].get('links', [])
                    if links:
                        protein_id = links[0]
                        # Fetch the sequence
                        seq = fetch_protein_sequence(str(protein_id))
                        if seq:
                            protein_seqs[gene] = {
                                'transcript': transcript,
                                'protein_id': str(protein_id),
                                'sequence': seq,
                                'length': len(seq)
                            }
                            if (idx + 1) % 10 == 0:
                                print(f"  [{idx+1}/{len(gene_transcripts)}] {gene}: {len(seq)} aa")
                        else:
                            failed.append((gene, transcript, protein_id, "sequence fetch failed"))
                    else:
                        failed.append((gene, transcript, None, "no protein links"))
                else:
                    failed.append((gene, transcript, None, "no linksetdbs"))
            else:
                failed.append((gene, transcript, None, f"HTTP {resp.status_code}"))
        except Exception as e:
            failed.append((gene, transcript, None, str(e)))

        # Rate limit: 3 requests/sec for NCBI
        time.sleep(0.35)

    print(f"\n  Successfully fetched: {len(protein_seqs)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print("\n  Failed genes:")
        for gene, transcript, pid, reason in failed[:10]:
            print(f"    {gene} ({transcript}): {reason}")

    # Save protein sequences
    records = []
    for gene, info in protein_seqs.items():
        records.append({
            'gene': gene,
            'transcript': info['transcript'],
            'protein_id': info['protein_id'],
            'sequence': info['sequence'],
            'length': info['length']
        })

    protein_df = pd.DataFrame(records)
    protein_df.to_csv(os.path.join(data_dir, "protein_sequences.csv"), index=False)
    print(f"\n  Saved protein sequences: {os.path.join(data_dir, 'protein_sequences.csv')}")

    # Also save individual FASTA files for easier use
    fasta_path = os.path.join(output_dir, "proteins.fasta")
    with open(fasta_path, 'w') as f:
        for gene, info in protein_seqs.items():
            f.write(f">{gene}|{info['protein_id']}\n")
            # Wrap at 80 chars
            seq = info['sequence']
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + '\n')
    print(f"  Saved FASTA: {fasta_path}")

    # Summary stats
    lengths = [info['length'] for info in protein_seqs.values()]
    print(f"\n  Protein length stats:")
    print(f"    Min: {min(lengths)}")
    print(f"    Max: {max(lengths)}")
    print(f"    Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"    Median: {sorted(lengths)[len(lengths)//2]}")

    # Check how many are <= 1022 aa (ESM2 limit)
    under_limit = sum(1 for l in lengths if l <= 1022)
    print(f"    <= 1022 aa (ESM2 limit): {under_limit}/{len(lengths)}")

if __name__ == "__main__":
    main()
