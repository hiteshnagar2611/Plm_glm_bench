#!/usr/bin/env python3
"""Download PDB structures for the 200 selected genes.

Uses RCSB PDB search to find structures by gene name.
Downloads the best structure (highest resolution) for each gene.
"""

import pandas as pd
import requests
import json
import os
import time
import sys

def search_pdb_by_gene(gene_name):
    """Search RCSB PDB for structures matching a gene name."""
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": gene_name
            }
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 5
            }
        }
    }

    try:
        resp = requests.post(search_url, json=query, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("result_set", [])
            return results
        else:
            return []
    except Exception as e:
        return []

def get_pdb_id_from_result(result):
    """Extract PDB ID from search result."""
    identifier = result.get("identifier", "")
    # Format: 2CP8_1 or 4OLE_1 - PDB ID is before the underscore
    pdb_id = identifier.split("_")[0]
    return pdb_id

def download_pdb(pdb_id, output_path):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(output_path, 'w') as f:
                f.write(resp.text)
            return True
        return False
    except:
        return False

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/data/pdb_structures"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading protein data...")
    proteins = pd.read_csv(os.path.join(data_dir, "protein_sequences.csv"))

    print(f"  Total genes: {len(proteins)}")

    # Search for PDB structures
    print("\nSearching RCSB PDB for structures...")
    pdb_results = {}
    failed_genes = []

    for idx, row in proteins.iterrows():
        gene = row['gene']

        # Search by gene name
        results = search_pdb_by_gene(gene)

        if results:
            # Get the first result (best score)
            pdb_id = get_pdb_id_from_result(results[0])
            pdb_results[gene] = {
                'pdb_id': pdb_id,
                'score': results[0].get('score', 0)
            }
        else:
            failed_genes.append(gene)

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(proteins)}] Found: {len(pdb_results)}, Not found: {len(failed_genes)}")

        # Rate limit
        time.sleep(0.1)

    print(f"\n  Found PDB structures: {len(pdb_results)}")
    print(f"  Missing structures: {len(failed_genes)}")

    if failed_genes:
        print(f"\n  Genes without PDB structures: {failed_genes[:20]}...")

    # Download PDB files
    print(f"\nDownloading {len(pdb_results)} PDB files...")
    downloaded = 0
    failed_downloads = []

    for gene, info in pdb_results.items():
        pdb_id = info['pdb_id']
        output_path = os.path.join(output_dir, f"{pdb_id}.pdb")

        if os.path.exists(output_path):
            downloaded += 1
            continue

        if download_pdb(pdb_id, output_path):
            downloaded += 1
        else:
            failed_downloads.append((gene, pdb_id))

        if downloaded % 20 == 0:
            print(f"  Downloaded: {downloaded}/{len(pdb_results)}")

        time.sleep(0.05)

    print(f"\n  Downloaded: {downloaded}")
    print(f"  Failed downloads: {len(failed_downloads)}")

    # Save PDB mapping
    pdb_map = pd.DataFrame([
        {'gene': gene, 'pdb_id': info['pdb_id'], 'score': info['score']}
        for gene, info in pdb_results.items()
    ])
    pdb_map.to_csv(os.path.join(data_dir, "pdb_mapping.csv"), index=False)
    print(f"\n  Saved PDB mapping: {os.path.join(data_dir, 'pdb_mapping.csv')}")

    # Run Foldseek to convert PDB to SaProt format
    foldseek_path = "benchmark_200/tools/foldseek/bin/foldseek"
    if os.path.exists(foldseek_path):
        print("\nConverting PDB to SaProt format with Foldseek...")
        saprot_dir = os.path.join(output_dir, "saprot_input")
        os.makedirs(saprot_dir, exist_ok=True)

        converted = 0
        for gene, info in pdb_results.items():
            pdb_id = info['pdb_id']
            pdb_path = os.path.join(output_dir, f"{pdb_id}.pdb")
            output_path = os.path.join(saprot_dir, f"{pdb_id}")

            if os.path.exists(output_path + ".db"):
                converted += 1
                continue

            # Use foldseek to create 3Di sequence
            cmd = f"{foldseek_path} lsaclust {pdb_path} {output_path} --msa-format a3m"
            os.system(cmd)
            converted += 1

            if converted % 20 == 0:
                print(f"  Converted: {converted}/{len(pdb_results)}")

        print(f"  Converted: {converted} PDB files to SaProt format")

if __name__ == "__main__":
    main()
