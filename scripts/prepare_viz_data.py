#!/usr/bin/env python3
"""Preprocess V3 benchmark data for 3D structure viewer."""
import os
import json
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V3 = os.path.join(BASE, "benchmark_v3")
DATA = os.path.join(V3, "data")
WEB = os.path.join(V3, "web")

# Load benchmark
df = pd.read_csv(os.path.join(DATA, "benchmark_v3.csv"))
df['label'] = df['ClinVar_label'].map({1.0: 1, 0.0: 0})

# Load gene-to-uniprot mapping
uniprot = pd.read_csv(os.path.join(DATA, "gene_to_uniprot.csv"))
uniprot_map = dict(zip(uniprot['gene'], uniprot['uniprot']))

# Get list of available PDB files
pdb_dir = os.path.join(DATA, "alphafold_structures")
available_pdb = {f.replace('.pdb', '') for f in os.listdir(pdb_dir) if f.endswith('.pdb')}

# Build genes list
genes = []
for gene, grp in df.groupby('GeneSymbol'):
    if gene in available_pdb:
        genes.append({
            'gene': gene,
            'uniprot': uniprot_map.get(gene, ''),
            'n_variants': len(grp),
            'n_pathogenic': int(grp['label'].sum()),
            'n_benign': int((grp['label'] == 0).sum()),
        })

genes.sort(key=lambda x: x['gene'])

# Build variants per gene
variants = {}
for gene, grp in df.groupby('GeneSymbol'):
    if gene not in available_pdb:
        continue

    pos_data = []
    for pos, pos_grp in grp.groupby('aa_position'):
        pos_int = int(pos)
        count = len(pos_grp)
        n_path = int(pos_grp['label'].sum())
        path_rate = n_path / count if count > 0 else 0

        var_list = []
        for _, row in pos_grp.iterrows():
            var_list.append({
                'ref': row['ref_aa'],
                'alt': row['alt_aa'],
                'label': 'Pathogenic' if row['label'] == 1 else 'Benign',
                'variation_id': int(row['VariationID']),
                'phenotype': str(row.get('PhenotypeList', '')),
            })

        pos_data.append({
            'position': pos_int,
            'path_rate': round(path_rate, 4),
            'count': count,
            'n_pathogenic': n_path,
            'n_benign': count - n_path,
            'variants': var_list,
        })

    pos_data.sort(key=lambda x: x['position'])
    variants[gene] = pos_data

# Save
os.makedirs(WEB, exist_ok=True)

with open(os.path.join(WEB, 'genes.json'), 'w') as f:
    json.dump(genes, f, indent=2)

with open(os.path.join(WEB, 'variants.json'), 'w') as f:
    json.dump(variants, f, indent=2)

print(f"Genes: {len(genes)}")
print(f"Genes with variants: {len(variants)}")
print(f"Total positions: {sum(len(v) for v in variants.values())}")
print(f"Output: {WEB}/genes.json, {WEB}/variants.json")
