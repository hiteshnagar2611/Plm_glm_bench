#!/usr/bin/env python3
"""Run all benchmark_v2 scoring scripts sequentially."""

import subprocess
import time
import os

scripts = [
    'scripts/v2_score_esm2.py',
    'scripts/v2_score_esm1b.py',
    'scripts/v2_score_prott5.py',
    'scripts/v2_score_saprot.py',
    'scripts/v2_score_ntv2.py',
    'scripts/v2_score_alphagenome.py',
    'scripts/v2_score_hyenadna.py',
]

os.makedirs('benchmark_v2/results', exist_ok=True)

for script in scripts:
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    t0 = time.time()
    result = subprocess.run(['python3', script], capture_output=False, timeout=7200)
    elapsed = time.time() - t0
    
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n{script}: {status} in {elapsed:.0f}s")

print(f"\n{'='*60}")
print(f"All done: {time.strftime('%Y-%m-%d %H:%M:%S')}")
