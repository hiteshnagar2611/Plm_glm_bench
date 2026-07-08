#!/usr/bin/env python3
"""Step 2b: Extract 3Di structural sequences from OpenFold predicted structures.

Runs Foldseek structureto3didescriptor on each predicted PDB file
to generate 3Di codes for SaProt scoring.
"""

import os
import sys
import subprocess
import time
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'benchmark_v2' / 'data'
FOLDSEEK = BASE_DIR / 'benchmark_200' / 'tools' / 'foldseek' / 'bin' / 'foldseek'

# Directories
PRED_DIR = DATA_DIR / 'openfold_structures'
THREE_DI_DIR = DATA_DIR / 'openfold_3di'
THREE_DI_DIR.mkdir(parents=True, exist_ok=True)


def extract_3di(pdb_file, output_dir):
    """Run Foldseek structureto3didescriptor on a PDB file."""
    gene = pdb_file.stem
    db_path = output_dir / gene

    # Skip if already extracted
    if db_path.exists() and (output_dir / f'{gene}.dbtype').exists():
        with open(db_path) as f:
            lines = f.readlines()
            if lines:
                parts = lines[0].strip().split('\t')
                if len(parts) >= 3 and len(parts[2]) > 0:
                    return parts[2]

    try:
        result = subprocess.run(
            [str(FOLDSEEK), 'structureto3didescriptor', str(pdb_file), str(db_path)],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            return None

        # Parse output
        if db_path.exists():
            with open(db_path) as f:
                lines = f.readlines()
                if lines:
                    parts = lines[0].strip().split('\t')
                    if len(parts) >= 3:
                        return parts[2]  # 3Di sequence
        return None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check Foldseek exists
    if not FOLDSEEK.exists():
        print(f"ERROR: Foldseek not found at {FOLDSEEK}")
        print("Run: python download_data.py")
        return 1

    # Find predicted structures
    if not PRED_DIR.exists():
        print(f"ERROR: No predicted structures at {PRED_DIR}")
        print("Run: python scripts/02a_predict_openfold.py")
        return 1

    pdb_files = sorted(PRED_DIR.glob('*.pdb'))
    print(f"Found {len(pdb_files)} predicted structures")

    if not pdb_files:
        print("No PDB files to process.")
        return 0

    t0 = time.time()
    success = 0
    failed = 0

    for i, pdb_file in enumerate(pdb_files):
        gene = pdb_file.stem
        three_di = extract_3di(pdb_file, THREE_DI_DIR)

        if three_di and len(three_di) > 0:
            success += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(pdb_files)}] {gene}: 3Di length={len(three_di)}")
        else:
            failed += 1
            print(f"  FAILED: {gene}")

    elapsed = time.time() - t0
    print(f"\nDone: {success} extracted, {failed} failed, {elapsed:.0f}s")
    print(f"3Di sequences saved to: {THREE_DI_DIR}")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
