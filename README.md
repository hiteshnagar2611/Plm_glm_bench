# ClinVar Benchmark Dataset for PLM and Genome Model Evaluation

This dataset provides filtered ClinVar variants for benchmarking protein language models (PLMs) and genome models on pathogenic prediction tasks.

## Dataset Overview

| Metric | Count |
|--------|-------|
| **Total variants** | 1,426,842 |
| **Pathogenic** | 72,027 |
| **Likely pathogenic** | 97,747 |
| **Benign** | 173,062 |
| **Likely benign** | 1,084,006 |
| **Missense** | 265,487 |
| **Synonymous** | 668,285 |

## Output Files

### 1. `clinvar_benchmark_full.csv` (215MB)
Complete dataset with all filtered variants and annotations.

**Columns:**
- `VariationID` - ClinVar variation ID
- `GeneSymbol` - Gene symbol
- `Chromosome` - Chromosome (1-22, X, Y)
- `PositionVCF` - Genomic position (GRCh38)
- `ReferenceAlleleVCF` / `AlternateAlleleVCF` - Alleles
- `ClinVar_label` - 0=Benign, 0.1=Likely benign, 1=Pathogenic, 1.1=Likely pathogenic
- `gold_stars` - Review status (1-4 stars)
- `MANE_transcript_id` - MANE Select transcript ID
- `MANE_gene` - Gene from MANE annotation
- `MANE_strand` - Strand (+/-)
- `raw_protein_change` - Original HGVS protein change
- `amino_acid_change` - Parsed AA change (e.g., R123H)
- `ref_aa` / `alt_aa` - Reference/alternate amino acids
- `aa_position` - Amino acid position
- `variant_type` - missense/synonymous/other
- `RCVaccession` - RCV accessions
- `PhenotypeList` - Associated phenotypes
- `ClinicalSignificance` - Original clinical significance text

### 2. `clinvar_missense_protein.csv` (153MB)
Missense and synonymous variants for **protein language models**:
- ESM2, ESM3, SaProt, ProtT5

Use `amino_acid_change`, `ref_aa`, `alt_aa`, `aa_position` for protein-level analysis.

### 3. `clinvar_benchmark_dna.csv` (159MB)
All variants formatted for **genome models**:
- AlphaGenome, EVO2, NT-v2, HyenaDNA

Use `Chromosome`, `PositionVCF`, `ReferenceAlleleVCF`, `AlternateAlleleVCF` for DNA-level analysis.

### 4. `clinvar_benchmark.vcf` (117MB)
Standard VCF format for tools that require VCF input.

## Filtering Criteria

1. **Assembly**: GRCh38 only
2. **Variant type**: Single nucleotide variants (SNVs) only
3. **Origin**: Germline only (somatic excluded)
4. **Chromosomes**: 1-22, X, Y only
5. **Clinical significance**: Pathogenic, Likely pathogenic, Benign, Likely benign
6. **Review status**: Gold stars >= 1 (at least one submitter with criteria)
7. **Alleles**: REF and ALT must be single nucleotides (A/T/C/G)

## Usage Examples

### For Protein Models (ESM2, ESM3, SaProt, ProtT5)

```python
import pandas as pd

# Load missense variants
df = pd.read_csv('clinvar_missense_protein.csv')

# Filter for missense only
missense = df[df['variant_type'] == 'missense']

# Get protein change info
for _, row in missense.head(10).iterrows():
    print(f"{row['GeneSymbol']}: {row['amino_acid_change']}")
    # e.g., "HFE: Q283P"
```

### For Genome Models (EVO2, NT-v2, HyenaDNA)

```python
import pandas as pd

# Load DNA dataset
df = pd.read_csv('clinvar_benchmark_dna.csv')

# Get variants for a specific region
region = df[(df['Chromosome'] == '17') & 
            (df['PositionVCF'].between(43044295, 43125483))]

# Format for model input
for _, row in region.iterrows():
    chrom = row['Chromosome']
    pos = row['PositionVCF']
    ref = row['ReferenceAlleleVCF']
    alt = row['AlternateAlleleVCF']
    label = row['ClinVar_label']
    print(f"chr{chrom}:{pos} {ref}>{alt} (label={label})")
```

### Computing AUROC

```python
from sklearn.metrics import roc_auc_score

# Assuming you have model scores
# model_scores = your_model.predict(variants)
# true_labels = (df['ClinVar_label'] >= 1).astype(int)  # Pathogenic=1, Benign=0

# auroc = roc_auc_score(true_labels, model_scores)
```

## Data Source

- **ClinVar**: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/
- **MANE**: https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.5/

## Citation

If you use this dataset, please cite:
- ClinVar: https://www.ncbi.nlm.nih.gov/clinvar/
- MANE: https://www.ncbi.nlm.nih.gov/refseq/MANE/

## Scripts

- `scripts/step1_basic_filter.py` - Basic ClinVar filtering
- `scripts/step2_mane_annotate.py` - MANE annotation and output generation
