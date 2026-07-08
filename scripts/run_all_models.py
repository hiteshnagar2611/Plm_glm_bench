#!/usr/bin/env python3
"""Run all scoring models sequentially with timestamps."""

import subprocess
import sys
import os
import time
from datetime import datetime

LOG_DIR = "benchmark_200/results"
SCRIPTS = [
    ("08_score_esm2.py", "ESM2-650M"),
    ("09_score_prott5.py", "ProtT5/ESM2-35M"),
    ("10_score_alphagenome.py", "AlphaGenome"),
    ("12_score_ntv2.py", "NT-v2"),
]

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    return line

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    all_logs = []

    for script, name in SCRIPTS:
        log_file = os.path.join(LOG_DIR, f"{os.path.splitext(script)[0]}_log.txt")
        all_logs.append(log_file)

        log(f"Starting {name} ({script})")
        log(f"  Log file: {log_file}")

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, f"scripts/{script}"],
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            log(f"  {name} completed in {elapsed:.0f}s (returncode=0)")
        else:
            log(f"  {name} FAILED in {elapsed:.0f}s (returncode={result.returncode})")

    log("=" * 60)
    log("All scoring scripts completed!")
    log("=" * 60)

if __name__ == "__main__":
    main()
