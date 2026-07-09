# ClinVar Benchmark V3: Protein & DNA Language Model Evaluation

Benchmark for evaluating protein language models (PLMs) and DNA/genomic foundation models on variant pathogenicity prediction using ClinVar data.

## Benchmark Summary

| Metric | Value |
|--------|-------|
| **Total variants** | 5,932 |
| **Genes** | 973 |
| **Pathogenic** | 3,625 |
| **Benign** | 2,307 |
| **Chromosomes** | 24 (1-22, X, Y) |

## Results (AUROC)

| Model | Type | AUROC | Variants Scored |
|-------|------|-------|-----------------|
| **ESM1b-650M** | Protein | **0.873** | 5,917 |
| ESM2-650M | Protein | 0.838 | 5,917 |
| SaProt-650M | Protein | 0.833 | 5,052 |
| NT-v2-500M | DNA | 0.638 | 5,932 |
| AlphaGenome | DNA | 0.552 | 5,776 |
| ProtT5-XL | Protein | 0.536 | 5,917 |
| HyenaDNA-150M | DNA | 0.448 | 5,932 |

## Models Evaluated

### Protein Language Models
| Model | Parameters | Scoring Method | Source |
|-------|-----------|----------------|--------|
| ESM1b-650M | 650M | Log-likelihood ratio (LLR) | Facebook AI |
| ESM2-650M | 650M | Log-likelihood ratio (LLR) | Facebook AI |
| SaProt-650M | 650M | LLR + Foldseek 3Di tokens | Westlake University |
| ProtT5-XL | 3B | Cosine similarity (encoder) | Rost Lab |

### DNA/Genomic Models
| Model | Parameters | Scoring Method | Source |
|-------|-----------|----------------|--------|
| NT-v2-500M | 500M | Hidden state delta | InstaDeep |
| AlphaGenome | 1.2B | Embedding delta (cosine) | Google DeepMind |
| HyenaDNA-150M | 150M | Log-likelihood ratio (LLR) | Stanford |

## Repository Structure

```
Plm_glm_bench/
├── benchmark_v3/
│   ├── data/
│   │   ├── benchmark_v3.csv          # Main benchmark file
│   │   ├── protein_sequences.csv     # Protein sequences for 973 genes
│   │   ├── alphafold_structures/     # AlphaFold PDB files (991 structures)
│   │   ├── alphafold_3di/            # Foldseek 3Di codes
│   │   └── gene_to_uniprot.csv       # Gene to UniProt mappings
│   ├── results/
│   │   ├── esm1b_scores.csv          # ESM1b scoring results
│   │   ├── esm2_650m_scores.csv      # ESM2 scoring results
│   │   ├── saprot_scores.csv         # SaProt scoring results
│   │   ├── prott5_scores.csv         # ProtT5 scoring results
│   │   ├── ntv2_scores.csv           # NT-v2 scoring results
│   │   ├── alphagenome_scores.csv    # AlphaGenome scoring results
│   │   ├── hyena_scores.csv          # HyenaDNA scoring results
│   │   └── benchmark_summary.csv     # Summary statistics
│   └── figures/
│       ├── fig1_auroc_comparison.png
│       ├── fig2_roc_curves.png
│       ├── fig3_score_distributions.png
│       ├── fig4_per_gene_auroc.png
│       ├── fig5_summary_table.png
│       ├── fig6_spearman_scatter.png
│       ├── fig_literature_comparison.png
│       └── fig_literature_comparison_spearman.png
├── scripts/
│   ├── v3_download_proteins.py       # Step 1: Download protein sequences
│   ├── v3_download_alphafold.py      # Step 2: Download AlphaFold structures
│   ├── v3_extract_3di.py             # Step 3: Extract 3Di codes
│   ├── v3_construct_benchmark.py     # Step 4: Construct benchmark CSV
│   ├── v3_extract_dna.py             # Step 5: Extract DNA sequences
│   ├── v3_score_esm2.py              # Step 6: Score with ESM2
│   ├── v3_score_esm1b.py             # Step 7: Score with ESM1b
│   ├── v3_score_prott5.py            # Step 8: Score with ProtT5
│   ├── v3_score_saprot.py            # Step 9: Score with SaProt
│   ├── v3_score_ntv2.py              # Step 10: Score with NT-v2
│   ├── v3_score_alphagenome.py       # Step 11: Score with AlphaGenome
│   ├── v3_score_hyenadna.py          # Step 12: Score with HyenaDNA
│   ├── v3_evaluate.py                # Step 13: Evaluate and generate figures
│   ├── v3_literature_comparison.py   # Literature comparison (AUROC)
│   ├── v3_literature_comparison_spearman.py  # Literature comparison (Spearman)
│   ├── v3_run_pipeline.py            # Run full pipeline
│   └── v3_score_all.sh               # Score all models sequentially
├── .gitignore
└── README.md
```

## Quick Start

### Option 1: Run Full Pipeline

```bash
python scripts/v3_run_pipeline.py
```

This runs all 13 steps sequentially:
1. Download protein sequences (NCBI)
2. Download AlphaFold structures
3. Extract 3Di codes (Foldseek)
4. Construct benchmark CSV
5. Extract DNA sequences (hg38)
6-12. Score with 7 models
13. Evaluate and generate figures

### Option 2: Run Individual Steps

```bash
# Step 1: Download protein sequences
python scripts/v3_download_proteins.py

# Step 2: Download AlphaFold structures
python scripts/v3_download_alphafold.py

# Step 3: Extract 3Di codes
python scripts/v3_extract_3di.py

# Step 4: Construct benchmark
python scripts/v3_construct_benchmark.py

# Step 5: Extract DNA sequences
python scripts/v3_extract_dna.py

# Step 6-12: Score with models (run in parallel if GPU available)
python scripts/v3_score_esm2.py
python scripts/v3_score_esm1b.py
python scripts/v3_score_prott5.py
python scripts/v3_score_saprot.py
python scripts/v3_score_ntv2.py
python scripts/v3_score_alphagenome.py
python scripts/v3_score_hyenadna.py

# Step 13: Evaluate
python scripts/v3_evaluate.py
```

### Option 3: Score All Models Sequentially

```bash
bash scripts/v3_score_all.sh
```

## Dependencies

### Required (pip install)
```bash
pip install torch torchvision torchaudio
pip install pandas numpy scipy scikit-learn matplotlib
pip install transformers esm foldseek
pip install pysam biopython requests tqdm
```

### For Specific Models

**ESM2/ESM1b:**
```bash
pip install fair-esm  # Facebook's ESM package
```

**SaProt:**
```bash
pip install saprot
# Requires Foldseek binary: benchmark_200/tools/foldseek/bin/foldseek
```

**AlphaGenome:**
```bash
export ALPHAGENOME_API_KEY="your_api_key_here"
# Or set in environment: export ALPHAGENOME_API_KEY="..."
```

**NT-v2:**
```bash
# Uses HuggingFace transformers
pip install transformers
```

**HyenaDNA:**
```bash
pip install hyena-dna
```

**ProtT5:**
```bash
pip install transformers sentencepiece
```

## Hardware Requirements

| Model | Minimum GPU | Recommended GPU | Time (5,932 variants) |
|-------|-------------|-----------------|----------------------|
| ESM2-650M | 8GB VRAM | 16GB+ VRAM | ~30 min |
| ESM1b-650M | 8GB VRAM | 16GB+ VRAM | ~30 min |
| SaProt-650M | 8GB VRAM | 16GB+ VRAM | ~1 hr (includes 3Di) |
| ProtT5-XL | CPU (slow) | GPU | ~16 hrs (CPU) |
| NT-v2-500M | CPU (slow) | GPU | ~16 hrs (CPU) |
| AlphaGenome | API | API | ~2 hrs (API calls) |
| HyenaDNA-150M | 8GB VRAM | 16GB+ VRAM | ~1 hr |

**Note:** ProtT5 and NT-v2 run on CPU by default. GPU acceleration significantly reduces runtime.

## Benchmark Construction

### Filtering Criteria
1. **Variant type**: Missense only
2. **Clinical significance**: Pathogenic or Benign only
3. **Transcript**: MANE Select (single-transcript genes only)
4. **Protein length**: < 1,001 amino acids
5. **Structure availability**: AlphaFold DB coverage

### Data Sources
- **ClinVar**: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/
- **MANE**: https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.5/
- **AlphaFold DB**: https://alphafold.ebi.ac.uk/
- **Reference genome**: hg38 (GRCh38)

## Scoring Methods

### Protein Models (ESM1b, ESM2, SaProt)
Log-likelihood ratio (LLR):
```
LLR = -log P(mut | context) + log P(wt | context)
```
Higher LLR = more pathogenic

### ProtT5
Cosine similarity between wild-type and mutant encoder representations:
```
score = cosine_similarity(enc_wt, enc_mut)
```
Lower similarity = more pathogenic (negated for evaluation)

### DNA Models (NT-v2, HyenaDNA)
- **NT-v2**: Hidden state delta (mean absolute difference)
- **HyenaDNA**: Log-likelihood ratio (LLR)

### AlphaGenome
Embedding delta:
```
score = cosine_similarity(ref_embedding, alt_embedding)
```

## Figures

### 1. AUROC Comparison
`fig1_auroc_comparison.png` - Bar chart comparing AUROC across all models

### 2. ROC Curves
`fig2_roc_curves.png` - ROC curves for all models

### 3. Score Distributions
`fig3_score_distributions.png` - Score distributions for pathogenic vs benign

### 4. Per-Gene AUROC
`fig4_per_gene_auroc.png` - Heatmap of per-gene AUROC

### 5. Summary Table
`fig5_summary_table.png` - Table with all metrics

### 6. Spearman Scatter
`fig6_spearman_scatter.png` - Spearman correlation scatter plots

### 7. Literature Comparison (AUROC)
`fig_literature_comparison.png` - Comparison with published AUROC values

### 8. Literature Comparison (Spearman)
`fig_literature_comparison_spearman.png` - Comparison with published Spearman values

## Literature Comparison

| Model | Our V3 | Literature | Delta | Literature Source |
|-------|--------|------------|-------|-------------------|
| ESM1b-650M | 0.873 | 0.905 | -0.032 | Brandes et al., Nature Genetics 2023 |
| ESM2-650M | 0.838 | 0.862 | -0.024 | SaProt paper, ICLR 2024 |
| SaProt-650M | 0.833 | 0.909 | -0.076 | SaProt paper, ICLR 2024 |
| NT-v2-500M | 0.638 | 0.780 | -0.142 | Dalla-Torre et al., Nature Methods 2025 |
| AlphaGenome | 0.552 | N/A | — | Avsec et al., Nature 2026 |
| ProtT5-XL | 0.536 | 0.610 | -0.074 | Elnaggar et al. 2022 |
| HyenaDNA-150M | 0.448 | 0.550 | -0.102 | Nguyen et al. 2023 |

**Note:** Literature values are from larger ClinVar datasets (30k-100k variants). Our V3 benchmark uses 5,932 variants across 973 genes with strict filtering.

## Environment Variables

Some models require API keys:

```bash
# AlphaGenome (Google DeepMind)
export ALPHAGENOME_API_KEY="your_key"

# HuggingFace (for model downloads)
export HF_TOKEN="your_token"

# NVIDIA (for some DNA models)
export NVIDIA_API_KEY="your_key"
```

## Citation

If you use this benchmark, please cite:

```bibtex
@article{plm_benchmark_v3,
  title={ClinVar Benchmark V3: Protein and DNA Language Model Evaluation},
  author={Hitesh Nagar},
  year={2025},
  url={https://github.com/hiteshnagar2611/Plm_glm_bench}
}
```

## License

This project is open source. See the repository for license details.

## Contact

For questions or issues, please open a GitHub issue at:
https://github.com/hiteshnagar2611/Plm_glm_bench/issues
