#!/usr/bin/env python3
"""Score benchmark_v3 with AlphaGenome."""

import pandas as pd
import numpy as np
import time
from alphagenome.data import genome
from alphagenome.models import dna_client

print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

df = pd.read_csv('benchmark_v3/data/benchmark_v3.csv')
dna = pd.read_csv('benchmark_v3/data/dna_sequences.csv')
dna_map = dict(zip(dna['VariationID'], dna['seq']))

API_KEY = os.environ.get('ALPHAGENOME_API_KEY', '')
model = dna_client.create(API_KEY)

requested_outputs = [
    dna_client.OutputType.RNA_SEQ,
    dna_client.OutputType.SPLICE_JUNCTIONS,
    dna_client.OutputType.SPLICE_SITES,
    dna_client.OutputType.SPLICE_SITE_USAGE,
    dna_client.OutputType.DNASE,
]

output_attr_map = {
    dna_client.OutputType.RNA_SEQ: 'rna_seq',
    dna_client.OutputType.SPLICE_JUNCTIONS: 'splice_junctions',
    dna_client.OutputType.SPLICE_SITES: 'splice_sites',
    dna_client.OutputType.SPLICE_SITE_USAGE: 'splice_site_usage',
    dna_client.OutputType.DNASE: 'dnase',
}

results = []
t0 = time.time()
errors = 0

for _, row in df.iterrows():
    vid = row['VariationID']
    chrom = 'chr' + str(row['Chromosome'])
    pos = int(row['PositionVCF'])
    ref = row['ReferenceAlleleVCF']
    alt = row['AlternateAlleleVCF']

    try:
        interval = genome.Interval(
            chromosome=chrom,
            start=max(0, pos - 8192),
            end=pos + 8192
        )
        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=ref,
            alternate_bases=alt,
        )

        outputs = model.predict_variant(
            interval=interval,
            variant=variant,
            ontology_terms=['UBERON:0001157'],
            requested_outputs=requested_outputs,
        )

        deltas = []
        for output_type, attr_name in output_attr_map.items():
            ref_data = getattr(outputs.reference, attr_name, None)
            alt_data = getattr(outputs.alternate, attr_name, None)
            if ref_data is not None and alt_data is not None:
                delta = np.mean(np.abs(np.array(ref_data.values) - np.array(alt_data.values)))
                deltas.append(delta)

        if deltas:
            results.append({
                'VariationID': vid,
                'GeneSymbol': row['GeneSymbol'],
                'ClinVar_label': row['ClinVar_label'],
                'ref_aa': row['ref_aa'],
                'alt_aa': row['alt_aa'],
                'aa_position': row['aa_position'],
                'AlphaGenome_delta': np.mean(deltas)
            })
    except Exception as e:
        errors += 1

    if len(results) % 100 == 0 and len(results) > 0:
        elapsed = time.time() - t0
        print(f"  {len(results)} variants, {elapsed:.0f}s, errors={errors}")

out = pd.DataFrame(results)
out.to_csv('benchmark_v3/results/alphagenome_scores.csv', index=False)
print(f"\nDone: {len(out)} variants, errors={errors} in {time.time()-t0:.0f}s")
print(f"End: {time.strftime('%Y-%m-%d %H:%M:%S')}")
