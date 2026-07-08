#!/usr/bin/env python3
"""Step 2a: Predict protein structures using NVIDIA NIM OpenFold 3 API.

Downloads full-length predicted structures for all benchmark genes.
These replace partial PDB structures for SaProt 3Di extraction.

Requires: NVIDIA API key (set NVIDIA_API_KEY env var or edit below).
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'benchmark_v2' / 'data'
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

# API configuration
API_KEY = os.environ.get('NVIDIA_API_KEY', '')
API_URL = 'https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict'

# Output directories
PRED_DIR = DATA_DIR / 'openfold_structures'
PRED_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = DATA_DIR / 'openfold_prediction_log.txt'


def log(msg):
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
        f.flush()


def predict_structure(gene, sequence, max_retries=3):
    """Predict protein structure using OpenFold 3 API."""
    # Minimal MSA: just the query sequence itself
    msa_alignment = f'>query\n{sequence}'

    payload = {
        "inputs": [{
            "input_id": gene,
            "output_format": "pdb",
            "molecules": [{
                "type": "protein",
                "id": "A",
                "sequence": sequence,
                "diffusion_samples": 1,
                "msa": {
                    "query": {
                        "a3m": {
                            "alignment": msa_alignment,
                            "format": "a3m",
                            "rank": -1
                        }
                    }
                }
            }]
        }]
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

            if response.status_code == 429:
                wait = 30 * (attempt + 1)
                log(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                log(f"  Error {response.status_code}: {response.text[:200]}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue

            data = response.json()

            # Extract structure
            outputs = data.get('outputs', [])
            if not outputs:
                log(f"  No outputs in response")
                continue

            structures = outputs[0].get('structures_with_scores', [])
            if not structures:
                log(f"  No structures in output")
                continue

            pdb_content = structures[0].get('structure', '')
            confidence = structures[0].get('confidence_score', 0)
            plddt = structures[0].get('complex_plddt_score', 0)

            if not pdb_content:
                log(f"  Empty structure content")
                continue

            return pdb_content, confidence, plddt

        except requests.exceptions.Timeout:
            log(f"  Timeout on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                time.sleep(10)
        except Exception as e:
            log(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    return None, 0, 0


def main():
    log("=" * 60)
    log("OpenFold 3 Structure Prediction")
    log("=" * 60)

    # Load gene list from benchmark
    benchmark = pd.read_csv(DATA_DIR / 'benchmark_v2.csv')
    genes = sorted(benchmark['GeneSymbol'].unique())
    log(f"Genes to predict: {len(genes)}")

    # Load protein sequences
    prots = pd.read_csv(DATA_DIR / 'protein_sequences.csv')
    gene_seq = dict(zip(prots['gene'], prots['sequence']))

    # Check which genes already have predictions
    existing = set()
    for f in PRED_DIR.glob('*.pdb'):
        existing.add(f.stem)
    log(f"Already predicted: {len(existing)}")

    # Filter to genes needing prediction
    to_predict = [g for g in genes if g not in existing and g in gene_seq]
    log(f"Need to predict: {len(to_predict)}")

    if not to_predict:
        log("All genes already predicted. Done.")
        return 0

    # Predict structures
    t0 = time.time()
    success = 0
    failed = 0
    total = len(to_predict)

    for i, gene in enumerate(to_predict):
        sequence = gene_seq[gene]
        log(f"[{i+1}/{total}] Predicting {gene} ({len(sequence)} aa)...")

        pdb_content, confidence, plddt = predict_structure(gene, sequence)

        if pdb_content:
            pdb_file = PRED_DIR / f'{gene}.pdb'
            with open(pdb_file, 'w') as f:
                f.write(pdb_content)
            log(f"  Saved: {pdb_file.name} (confidence={confidence:.3f}, pLDDT={plddt:.3f})")
            success += 1
        else:
            log(f"  FAILED: {gene}")
            failed += 1

        # Rate limiting: 1.5s between calls
        if i < total - 1:
            time.sleep(1.5)

        # Progress report every 20 genes
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            eta = (total - i - 1) / rate if rate > 0 else 0
            log(f"  Progress: {i+1}/{total} ({success} ok, {failed} failed, {rate:.1f}/min, ETA {eta:.0f}min)")

    elapsed = time.time() - t0
    log(f"\nDone: {success} predicted, {failed} failed, {elapsed:.0f}s")
    log(f"Structures saved to: {PRED_DIR}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
