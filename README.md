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

## Reference Genome Setup

The DNA sequence extraction requires the hg38 reference genome. Download and set up:

```bash
# Download hg38 reference genome
wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
gunzip hg38.fa.gz

# Create reference directory
mkdir -p reference
mv hg38.fa reference/

# Update script path (if running DNA extraction)
# Edit scripts/v3_extract_dna.py: REF_PATH = 'reference/hg38.fa'
```

**Note:** The `benchmark_v3/data/dna_sequences.csv` file contains pre-extracted 6001bp sequences for all 5,932 variants. You only need the reference genome if re-running DNA extraction.

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
- **Reference genome**: hg38 (GRCh38.p14) - UCSC Genome Browser
  - Download: https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
  - Or GDC version: https://gdc.cancer.gov/files/public/file/GRCh38.d1.vd1.fa

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

### Models Used

If you use the results from this benchmark, please cite the original model papers:

**ESM1b-650M:**
```bibtex
@article{rives2021biological,
  title={Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences},
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Mike and Zitnick, C Lawrence and Ma, Jerry and others},
  journal={Proceedings of the National Academy of Sciences},
  volume={118},
  number={15},
  year={2021},
  publisher={National Academy of Sciences}
}
```

**ESM2-650M:**
```bibtex
@article{lin2023evolutionary,
  title={Evolutionary-scale prediction of atomic-level protein structure with a language model},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Sercu, Tom and Rives, Alexander},
  journal={Science},
  volume={379},
  number={6637},
  pages={1123--1130},
  year={2023},
  publisher={American Association for the Advancement of Science}
}
```

**SaProt-650M:**
```bibtex
@inproceedings{su2024saprot,
  title={SaProt: Protein Language Modeling with Structure-aware Vocabulary},
  author={Su, Zhiqiang and Chen, Yifeng and Ma, Yuxiang and Jiang, Wei and Li, Zhanhui and Gao, Junzhou},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

**ProtT5-XL:**
```bibtex
@article{elnaggar2022prottrans,
  title={ProtTrans: Toward Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing},
  author={Elnaggar, Ahmed and Heinzinger, Martin and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={10},
  pages={7112--7123},
  year={2022},
  publisher={IEEE}
}
```

**NT-v2-500M:**
```bibtex
@article{dallatorre2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and Carranza, Nicolas L{\'o}pez and Grzywaczewski, Adam and Oteri, Francesco and others},
  journal={Nature Methods},
  year={2025},
  publisher={Nature Publishing Group}
}
```

**AlphaGenome:**
```bibtex
@article{avsec2026alphagenome,
  title={Advancing regulatory variant effect prediction with AlphaGenome},
  author={Avsec, {\v Z}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and others},
  journal={Nature},
  volume={649},
  number={8099},
  pages={1206--1218},
  year={2026},
  publisher={Nature Publishing Group}
}
```

**HyenaDNA-150M:**
```bibtex
@article{nguyen2023hyenadna,
  title={HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author={Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Andrew W. and Massaroli, Stefano and Birber, Christos and Shafi, R. and Ermon, Stefano and Ré, Christopher and Hsu, Patrick},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

### Tools Used

**ClinVar:**
```bibtex
@article{landrum2018clinvar,
  title={ClinVar: improvements to accessing data},
  author={Landrum, Melissa J and Lee, Jennifer M and Benson, Lisa and Brown, Jennifer and Chao, Chiao and Chitipiralla, Shan and Gu, Bin and Hart, Jolen and Hoffman, Douglas and Jang, Wonhee and others},
  journal={Nucleic acids research},
  volume={46},
  number={D1},
  pages={D636--D641},
  year={2018},
  publisher={Oxford University Press}
}
```

**Foldseek:**
```bibtex
@article{van Kempen2024foldseek,
  title={Fast and accurate protein structure search with Foldseek},
  author={van Kempen, Michel and Kim, Stephanie S and Tumescheit, Charlotte and Mirdita, Milot and Lee, Jooyoung and Gilchrist, Cameron L M and Söding, Johannes and Steinegger, Martin},
  journal={Nature Biotechnology},
  volume={42},
  pages={586--589},
  year={2024},
  publisher={Nature Publishing Group}
}
```

**AlphaFold DB:**
```bibtex
@article{varadi2024alphafold,
  title={AlphaFold Protein Structure Database: significantly expanding the structural coverage of protein-sequence annotations with high-accuracy models},
  author={Varadi, Mihaly and Anyango, Stephen and Deshpande, Mandar and Nair, Sreenath and Natassia, Cindy and Yordanova, Galabina and Taylor, David and Yankova, Kexin and Zaretski, Gal and Marth, Sean and others},
  journal={Nucleic acids research},
  volume={50},
  number={D1},
  pages={D439--D444},
  year={2024},
  publisher={Oxford University Press}
}
```

## License

This project is open source. See the repository for license details.

## Contact

For questions or issues, please open a GitHub issue at:
https://github.com/hiteshnagar2611/Plm_glm_bench/issues
