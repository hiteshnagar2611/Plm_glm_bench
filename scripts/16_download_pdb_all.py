#!/usr/bin/env python3
"""Download PDB structures for all 1,377 genes - fast version."""

import pandas as pd
import os
import json
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data/processed")
PDB_DIR = Path("benchmark_200/data/pdb_structures")
FOLDSEEK = Path("benchmark_200/tools/foldseek/bin/foldseek")

def search_rcsb_gene(gene):
    """Search RCSB for a single gene, return (gene, pdb_id) or (gene, None)."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    payload = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity.pdbx_description",
                "operator": "contains_words",
                "value": gene
            }
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 5},
            "sort": [{"sort_by": "score", "direction": "desc"}]
        }
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if data.get("result_set"):
                pdb_id = data["result_set"][0]["identifier"].split("_")[0]
                return (gene, pdb_id.upper())
    except Exception:
        pass
    return (gene, None)

def download_pdb(pdb_id, output_dir):
    """Download a single PDB file."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = output_dir / f"{pdb_id}.pdb"
    if output_path.exists() and output_path.stat().st_size > 100:
        return True
    try:
        urllib.request.urlretrieve(url, str(output_path))
        return output_path.exists() and output_path.stat().st_size > 100
    except Exception:
        return False

def main():
    # Load full dataset
    full = pd.read_csv(DATA_DIR / "clinvar_benchmark_full.csv")
    genes = sorted(full["GeneSymbol"].unique())
    print(f"Total genes: {len(genes)}")

    # Check existing PDB mapping
    existing_pdb = pd.read_csv("benchmark_200/data/pdb_mapping.csv")
    existing_map = dict(zip(existing_pdb["gene"], existing_pdb["pdb_id"]))
    print(f"Already have PDB for {len(existing_map)} genes")

    # Genes needing PDB
    genes_needed = [g for g in genes if g not in existing_map]
    print(f"Need to search RCSB for {len(genes_needed)} genes")

    # Parallel RCSB search
    print("\nSearching RCSB (16 threads)...")
    new_results = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(search_rcsb_gene, g): g for g in genes_needed}
        done = 0
        for future in as_completed(futures):
            gene, pdb_id = future.result()
            if pdb_id:
                new_results[gene] = pdb_id
            done += 1
            if done % 200 == 0:
                print(f"  Searched {done}/{len(genes_needed)}, found {len(new_results)} PDBs")

    print(f"  Done: searched {len(genes_needed)}, found {len(new_results)} new PDBs")

    # Combine
    all_pdb = dict(existing_map)
    all_pdb.update(new_results)
    print(f"Total genes with PDB: {len(all_pdb)}/{len(genes)} ({len(all_pdb)/len(genes)*100:.1f}%)")

    # Save mapping
    pdb_df = pd.DataFrame([
        {"gene": g, "pdb_id": p, "score": 1.0} for g, p in all_pdb.items()
    ])
    pdb_df.to_csv(DATA_DIR / "pdb_mapping_all.csv", index=False)
    print(f"Saved mapping to pdb_mapping_all.csv")

    # Download PDB files (parallel)
    pdb_download_dir = Path("benchmark_200/data/pdb_structures_all")
    pdb_download_dir.mkdir(parents=True, exist_ok=True)

    unique_pdb = list(set(all_pdb.values()))
    print(f"\nDownloading {len(unique_pdb)} unique PDB files (16 threads)...")

    downloaded = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_pdb, pid, pdb_download_dir): pid for pid in unique_pdb}
        done = 0
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            done += 1
            if done % 200 == 0:
                print(f"  Downloaded {done}/{len(unique_pdb)}")

    print(f"  Downloaded {downloaded}/{len(unique_pdb)} PDB files")

    # Convert to Foldseek (parallel)
    print("\nConverting to Foldseek 3Di databases...")
    fs_output_dir = Path("benchmark_200/data/pdb_structures_all/saprot_input")
    fs_output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = list(pdb_download_dir.glob("*.pdb"))
    converted = 0

    def convert_one(pdb_file):
        db_path = fs_output_dir / pdb_file.stem
        if db_path.exists():
            return True
        try:
            result = subprocess.run(
                [str(FOLDSEEK), "structureto3didescriptor",
                 str(pdb_file), str(db_path)],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(convert_one, pf): pf for pf in pdb_files}
        for future in as_completed(futures):
            if future.result():
                converted += 1

    print(f"Converted {converted}/{len(pdb_files)} PDB files to Foldseek databases")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total genes in ClinVar: {len(genes)}")
    print(f"Genes with PDB structures: {len(all_pdb)}")
    print(f"Unique PDB IDs: {len(unique_pdb)}")
    print(f"PDB files downloaded: {downloaded}")
    print(f"Foldseek databases: {converted}")
    print(f"Coverage: {len(all_pdb)/len(genes)*100:.1f}%")

if __name__ == "__main__":
    main()
