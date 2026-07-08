#!/usr/bin/env python3
"""End-to-end pipeline runner for DNA PLM Benchmark.

Runs the complete benchmark pipeline:
1. Filter ClinVar data
2. Download PDB structures
3. Download protein sequences
4. Extract DNA sequences
5. Construct benchmark_v2
6. Score with all 7 models
7. Evaluate and generate figures

Usage:
    python run_pipeline.py                    # Run everything
    python run_pipeline.py --skip-download    # Skip data download
    python run_pipeline.py --skip-scoring     # Skip model scoring
    python run_pipeline.py --skip-eval        # Skip evaluation/plotting
    python run_pipeline.py --step 5           # Start from step 5
    python run_pipeline.py --only esm2        # Run only ESM2 scoring
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'

STEPS = [
    (1, '01_filter_clinvar.py',       'Filter ClinVar data',            '~30s'),
    (2, '02_download_pdb.py',         'Download PDB structures + 3Di',  '~10min'),
    (2.1, '02a_predict_openfold.py',  'Predict OpenFold 3 structures',  '~40min'),
    (2.2, '02b_extract_3di.py',       'Extract 3Di from predicted',     '~2min'),
    (3, '03_download_proteins.py',    'Download protein sequences',     '~5min'),
    (4, '04_extract_dna.py',          'Extract DNA sequences from hg38','~2min'),
    (5, '05_construct_benchmark.py',  'Construct benchmark_v2.csv',     '~1min'),
    (6, '06_score_esm2.py',           'Score with ESM2-650M',           '~11min'),
    (7, '07_score_esm1b.py',          'Score with ESM1b-650M',          '~10min'),
    (8, '08_score_prott5.py',         'Score with ProtT5-XL',           '~65min'),
    (9, '09_score_saprot.py',         'Score with SaProt-650M',         '~3min'),
    (10, '10_score_ntv2.py',          'Score with NT-v2-500M',          '~52min'),
    (11, '11_score_alphagenome.py',   'Score with AlphaGenome',         '~15min'),
    (12, '12_score_hyenadna.py',      'Score with HyenaDNA-150M',       '~5min'),
    (13, '13_evaluate.py',            'Evaluate all models',            '~10s'),
    (14, '14a_plot_dataset.py',       'Plot dataset overview',          '~5s'),
    (15, '14b_plot_literature.py',    'Plot literature comparison',     '~5s'),
    (16, '14c_plot_pipeline.py',      'Plot filtering pipeline',        '~5s'),
]

SCORING_STEPS = {6, 7, 8, 9, 10, 11, 12}


def run_script(script_name, description):
    """Run a pipeline script and return success status."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"  ERROR: Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"  Running: {description}")
    print(f"  Script:  {script_name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            timeout=7200  # 2 hour timeout per step
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  DONE: {description} ({elapsed:.0f}s)")
            return True
        else:
            print(f"\n  FAILED: {description} (exit code {result.returncode}, {elapsed:.0f}s)")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT: {description} (>2 hours)")
        return False
    except KeyboardInterrupt:
        print(f"\n  INTERRUPTED by user")
        return False


def main():
    parser = argparse.ArgumentParser(description='DNA PLM Benchmark Pipeline')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip steps 1-2 (data download)')
    parser.add_argument('--skip-scoring', action='store_true',
                        help='Skip model scoring (steps 6-12)')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation and plotting (steps 13-16)')
    parser.add_argument('--step', type=int, default=1,
                        help='Start from step N (1-16)')
    parser.add_argument('--only', type=str, default=None,
                        help='Run only a specific model (esm2, esm1b, prott5, saprot, ntv2, alphagenome, hyenadna)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without executing')
    args = parser.parse_args()

    print("=" * 60)
    print("  DNA PLM Benchmark - End-to-End Pipeline")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python: {sys.executable}")
    print()

    # Determine which steps to run
    steps_to_run = []
    model_map = {
        'esm2': 6, 'esm1b': 7, 'prott5': 8, 'saprot': 9,
        'ntv2': 10, 'alphagenome': 11, 'hyenadna': 12
    }

    for step_num, script, desc, est_time in STEPS:
        # Skip logic
        if step_num < args.step:
            continue
        if args.skip_download and step_num <= 2.2:
            continue
        if args.skip_scoring and step_num in SCORING_STEPS:
            continue
        if args.skip_eval and step_num >= 13:
            continue
        if args.only and step_num in SCORING_STEPS:
            target = model_map.get(args.only)
            if step_num != target:
                continue

        steps_to_run.append((step_num, script, desc, est_time))

    if not steps_to_run:
        print("  No steps to run.")
        return 0

    # Show plan
    print("  Pipeline steps:")
    total_est = 0
    for step_num, script, desc, est_time in steps_to_run:
        print(f"    Step {step_num:5}: {desc} (est. {est_time})")
    print()

    if args.dry_run:
        print("  [DRY RUN] No scripts executed.")
        return 0

    # Run
    t_start = time.time()
    results = {}
    for step_num, script, desc, est_time in steps_to_run:
        success = run_script(script, desc)
        results[step_num] = success
        if not success:
            print(f"\n  Pipeline stopped at step {step_num}. Fix the error and re-run.")
            print(f"  Use --step {step_num} to resume from this step.")
            break

    # Summary
    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    for step_num, success in results.items():
        script = [s[1] for s in STEPS if s[0] == step_num][0]
        status = "OK" if success else "FAILED"
        print(f"  Step {step_num:5}: {status} ({script})")

    n_ok = sum(results.values())
    n_total = len(results)
    print(f"\n  Completed: {n_ok}/{n_total} steps in {total_time:.0f}s")

    if n_ok == n_total:
        print("\n  All steps complete! Results in benchmark_v2/figures/")
        return 0
    else:
        return 1


if __name__ == '__main__':
    sys.exit(main())
