#!/usr/bin/env python3
"""Extract 3Di structural codes from AlphaFold structures via Foldseek.

Converts AlphaFold PDB files to 3Di sequences for SaProt input.
"""

import subprocess
import time
from pathlib import Path

FOLDSEEK = Path('benchmark_200/tools/foldseek/bin/foldseek')
AF_DIR = Path('benchmark_v3/data/alphafold_structures')
THREE_DI_DIR = Path('benchmark_v3/data/alphafold_3di')
THREE_DI_DIR.mkdir(parents=True, exist_ok=True)

# Also check v2 3Di for reuse
v2_3di_dir = Path('benchmark_v2/data/alphafold_3di')

pdb_files = sorted(AF_DIR.glob('*.pdb'))
print(f"AlphaFold structures: {len(pdb_files)}")

# Check which already have 3Di
existing_3di = {f.name for f in THREE_DI_DIR.iterdir() if not f.name.endswith('.dbtype')}
print(f"3Di already extracted: {len(existing_3di)}")

to_extract = [f for f in pdb_files if f.stem not in existing_3di]
print(f"Need to extract: {len(to_extract)}")

extracted = 0
failed = 0
t0 = time.time()

for idx, pdb_file in enumerate(to_extract):
    gene = pdb_file.stem
    out_path = THREE_DI_DIR / gene

    # Also check v2 directory
    v2_file = v2_3di_dir / gene
    if v2_file.exists():
        import shutil
        shutil.copy2(v2_file, out_path)
        extracted += 1
        continue

    try:
        result = subprocess.run(
            [str(FOLDSEEK), 'structureto3didescriptor', str(pdb_file), str(out_path)],
            capture_output=True, timeout=30
        )
        if out_path.exists():
            extracted += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1

    if (idx + 1) % 100 == 0:
        elapsed = time.time() - t0
        print(f"  {idx+1}/{len(to_extract)}: extracted={extracted}, failed={failed}, {elapsed:.0f}s")

print(f"\nDone: extracted={extracted}, failed={failed}")
print(f"Total 3Di files: {len(list(THREE_DI_DIR.iterdir()))}")
print(f"Time: {time.time()-t0:.0f}s")
