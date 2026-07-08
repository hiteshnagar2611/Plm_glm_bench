#!/usr/bin/env python3
"""
Simple PDB filter: Query PDB for each unique gene symbol.
Uses simple text search in protein descriptions.
"""
import pandas as pd
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

def check_gene_pdb(gene):
    """Check if gene has PDB structure via simple text search."""
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity.pdbx_description",
                "operator": "contains_phrase",
                "value": gene
            }
        },
        "return_type": "polymer_entity",
        "request_options": {
            "paginate": {"start": 0, "rows": 1},
            "results_content_type": ["experimental"]
        }
    }
    
    try:
        resp = requests.post(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            json=query,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code == 200:
            total = resp.json().get("total_count", 0)
            return gene, total > 0
        return gene, False
    except:
        return gene, False

def main():
    print("Loading dataset...")
    df = pd.read_csv(PROCESSED_DIR / "clinvar_missense_protein.csv", low_memory=False)
    print(f"Total variants: {len(df):,}")
    print(f"Class distribution:")
    print(f"  Benign: {len(df[df['ClinVar_label']==0]):,}")
    print(f"  Pathogenic: {len(df[df['ClinVar_label']==1]):,}")
    
    genes = df['GeneSymbol'].dropna().unique()
    print(f"\nUnique genes: {len(genes):,}")
    
    print("Checking PDB availability (parallel)...")
    genes_with_pdb = set()
    
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = {executor.submit(check_gene_pdb, g): g for g in genes}
        done = 0
        for future in as_completed(futures):
            gene, has_pdb = future.result()
            if has_pdb:
                genes_with_pdb.add(gene)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(genes)} checked, {len(genes_with_pdb)} with PDB so far...")
    
    print(f"\nGenes with PDB: {len(genes_with_pdb):,}")
    print(f"Genes without PDB: {len(genes) - len(genes_with_pdb):,}")
    
    # Filter
    df['has_pdb'] = df['GeneSymbol'].isin(genes_with_pdb)
    filtered = df[df['has_pdb']].drop(columns=['has_pdb']).reset_index(drop=True)
    
    print(f"\nFinal dataset:")
    print(f"  Variants with PDB: {len(filtered):,}")
    print(f"  Removed: {len(df) - len(filtered):,}")
    print(f"  Benign: {len(filtered[filtered['ClinVar_label']==0]):,}")
    print(f"  Pathogenic: {len(filtered[filtered['ClinVar_label']==1]):,}")
    
    filtered.to_csv(PROCESSED_DIR / "clinvar_missense_protein.csv", index=False)
    print(f"\n✓ Saved")
    
    sample = list(genes_with_pdb)[:20]
    print(f"\nSample genes with PDB: {', '.join(sample)}")

if __name__ == "__main__":
    main()
