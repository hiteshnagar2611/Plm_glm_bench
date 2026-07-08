#!/usr/bin/env python3
"""Score variants using AlphaGenome API.

Uses Google's AlphaGenome Python client to score DNA variants.
Computes delta-scores across multiple modalities.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from alphagenome.data import genome
from alphagenome.models import dna_client

API_KEY = os.environ.get('ALPHAGENOME_API_KEY', '')

def main():
    data_dir = "benchmark_200/data"
    output_dir = "benchmark_200/results"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading variant data...", flush=True)
    variants = pd.read_csv(os.path.join(data_dir, "clinvar_200_dna.csv"))
    print(f"  Variants: {len(variants)}", flush=True)

    print("Loading AlphaGenome model...", flush=True)
    model = dna_client.create(API_KEY)
    print("  Model loaded", flush=True)

    # Define requested outputs
    requested_outputs = [
        dna_client.OutputType.RNA_SEQ,
        dna_client.OutputType.SPLICE_JUNCTIONS,
        dna_client.OutputType.SPLICE_SITES,
        dna_client.OutputType.SPLICE_SITE_USAGE,
        dna_client.OutputType.DNASE,
    ]

    # Map output types to attribute names (lowercase, no spaces)
    output_attr_map = {
        dna_client.OutputType.RNA_SEQ: 'rna_seq',
        dna_client.OutputType.SPLICE_JUNCTIONS: 'splice_junctions',
        dna_client.OutputType.SPLICE_SITES: 'splice_sites',
        dna_client.OutputType.SPLICE_SITE_USAGE: 'splice_site_usage',
        dna_client.OutputType.DNASE: 'dnase',
    }

    print(f"\nScoring {len(variants)} variants with AlphaGenome...", flush=True)
    results = []
    errors = 0
    t_start = time.time()

    for idx, row in variants.iterrows():
        chrom = str(row['Chromosome'])
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        pos = int(row['PositionVCF'])
        ref = row['ReferenceAlleleVCF']
        alt = row['AlternateAlleleVCF']
        label = row['ClinVar_label']

        try:
            # AlphaGenome requires specific interval lengths: 16384, 131072, etc.
            # Use minimum supported length (16384)
            half_window = 8192
            interval = genome.Interval(
                chromosome=chrom,
                start=max(0, pos - half_window),
                end=pos + half_window
            )
            variant = genome.Variant(
                chromosome=chrom,
                position=pos,
                reference_bases=ref,
                alternate_bases=alt,
            )

            # Score variant
            outputs = model.predict_variant(
                interval=interval,
                variant=variant,
                ontology_terms=['UBERON:0001157'],
                requested_outputs=requested_outputs,
            )

            # Compute delta score: mean absolute difference across all output types
            deltas = []
            for output_type, attr_name in output_attr_map.items():
                ref_data = getattr(outputs.reference, attr_name, None)
                alt_data = getattr(outputs.alternate, attr_name, None)

                if ref_data is not None and alt_data is not None:
                    ref_vals = np.array(ref_data.values)
                    alt_vals = np.array(alt_data.values)
                    delta = np.mean(np.abs(ref_vals - alt_vals))
                    deltas.append(delta)

            if deltas:
                # Use max delta across modalities
                max_delta = max(deltas)
                results.append({
                    'VariationID': int(row['VariationID']),
                    'GeneSymbol': row['GeneSymbol'],
                    'Chromosome': chrom,
                    'PositionVCF': pos,
                    'ReferenceAlleleVCF': ref,
                    'AlternateAlleleVCF': alt,
                    'ClinVar_label': label,
                    'AlphaGenome_delta': max_delta,
                })
            else:
                errors += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error for {row['GeneSymbol']} {chrom}:{pos}: {str(e)[:200]}", flush=True)

        if (idx + 1) % 25 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{idx+1}/{len(variants)}] Scored: {len(results)}, Errors: {errors} | {rate:.2f} vars/s", flush=True)

        # Small delay to avoid rate limiting
        time.sleep(0.05)

    print(f"\n  Completed: {len(results)} scored, {errors} errors", flush=True)

    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "alphagenome_scores.csv")
    results_df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)

    if len(results_df) > 0:
        print(f"\n  AlphaGenome Score Statistics:", flush=True)
        print(f"    Mean delta: {results_df['AlphaGenome_delta'].mean():.6f}", flush=True)
        print(f"    Std delta: {results_df['AlphaGenome_delta'].std():.6f}", flush=True)
        for label in [0.0, 1.0]:
            subset = results_df[results_df['ClinVar_label'] == label]
            name = "Benign" if label == 0.0 else "Pathogenic"
            print(f"    {name}: mean={subset['AlphaGenome_delta'].mean():.6f}, std={subset['AlphaGenome_delta'].std():.6f}", flush=True)

if __name__ == "__main__":
    main()
