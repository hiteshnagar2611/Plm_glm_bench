#!/bin/bash
cd /Users/igib/Desktop/dna_plm_bench
LOG=benchmark_v3/results/scoring.log

echo "=== Scoring Start: $(date) ===" > $LOG

echo "1. SaProt" >> $LOG
python -u scripts/v3_score_saprot.py >> $LOG 2>&1
echo "1. SaProt done: $(date)" >> $LOG

echo "2. ESM2" >> $LOG
python -u scripts/v3_score_esm2.py >> $LOG 2>&1
echo "2. ESM2 done: $(date)" >> $LOG

echo "3. ESM1b" >> $LOG
python -u scripts/v3_score_esm1b.py >> $LOG 2>&1
echo "3. ESM1b done: $(date)" >> $LOG

echo "4. HyenaDNA" >> $LOG
python -u scripts/v3_score_hyenadna.py >> $LOG 2>&1
echo "4. HyenaDNA done: $(date)" >> $LOG

echo "5. ProtT5" >> $LOG
python -u scripts/v3_score_prott5.py >> $LOG 2>&1
echo "5. ProtT5 done: $(date)" >> $LOG

echo "6. NT-v2" >> $LOG
python -u scripts/v3_score_ntv2.py >> $LOG 2>&1
echo "6. NT-v2 done: $(date)" >> $LOG

echo "7. AlphaGenome" >> $LOG
python -u scripts/v3_score_alphagenome.py >> $LOG 2>&1
echo "7. AlphaGenome done: $(date)" >> $LOG

echo "=== All scoring finished: $(date) ===" >> $LOG
