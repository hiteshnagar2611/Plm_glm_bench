#!/usr/bin/env python3
"""Run the full benchmark_v3 pipeline.

Steps:
1. Download protein sequences (NCBI elink+efetch)
2. Download AlphaFold structures (AlphaFold DB)
3. Extract 3Di codes (Foldseek)
4. Construct benchmark_v3.csv
5. Extract DNA sequences (hg38)
6-12. Score with 7 models
13. Evaluate and generate figures
"""

import subprocess
import time
from pathlib import Path

SCRIPTS_DIR = Path('scripts')

steps = [
    ('1. Download proteins', 'v3_download_proteins.py'),
    ('2. Download AlphaFold', 'v3_download_alphafold.py'),
    ('3. Extract 3Di', 'v3_extract_3di.py'),
    ('4. Construct benchmark', 'v3_construct_benchmark.py'),
    ('5. Extract DNA', 'v3_extract_dna.py'),
    ('6. Score ESM2', 'v3_score_esm2.py'),
    ('7. Score ESM1b', 'v3_score_esm1b.py'),
    ('8. Score ProtT5', 'v3_score_prott5.py'),
    ('9. Score SaProt', 'v3_score_saprot.py'),
    ('10. Score NT-v2', 'v3_score_ntv2.py'),
    ('11. Score AlphaGenome', 'v3_score_alphagenome.py'),
    ('12. Score HyenaDNA', 'v3_score_hyenadna.py'),
    ('13. Evaluate', 'v3_evaluate.py'),
]

print(f"Benchmark V3 Pipeline")
print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")

for step_name, script in steps:
    print(f"\n{'─'*60}")
    print(f"Step: {step_name}")
    print(f"Script: {script}")
    print(f"{'─'*60}")

    t0 = time.time()
    result = subprocess.run(
        ['python', str(SCRIPTS_DIR / script)],
        capture_output=False
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n*** STEP FAILED: {step_name} (exit code {result.returncode}) ***")
        print("Continue? [y/n]")
        if input().lower() != 'y':
            break
    else:
        print(f"Completed in {elapsed:.0f}s")

print(f"\n{'='*60}")
print(f"Pipeline finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
