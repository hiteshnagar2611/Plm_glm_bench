#!/usr/bin/env python3
"""Figure 9b: Protein family and functional diversity of the benchmark."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

FIG_DIR = Path(__file__).parent.parent / 'benchmark_v2' / 'figures'

# ── Data (from UniProt annotations) ──
mol_functions = {
    'Hydrolase': 46, 'Transferase': 39, 'DNA-binding': 32,
    'Receptor': 16, 'Activator': 16, 'Developmental protein': 13,
    'Kinase': 12, 'Repressor': 11, 'Serine/threonine-protein kinase': 8,
    'Oxidoreductase': 8, 'RNA-binding': 8, 'Protease': 8,
    'Helicase': 7, 'Chromatin regulator': 6, 'Dioxygenase': 5,
    'Methyltransferase': 5, 'Serine protease': 5, 'Protein phosphatase': 4,
    'Ion channel': 4, 'Nuclease': 4, 'Chaperone': 4,
}

cell_components = {
    'Cytoplasm': 96, 'Nucleus': 95, 'Membrane': 88,
    'Cell membrane': 52, 'Secreted': 25, 'ER': 24,
    'Cell projection': 20, 'Golgi': 19, 'Cytoskeleton': 17,
    'Mitochondrion': 17, 'Chromosome': 17, 'Lysosome': 15,
    'Endosome': 15, 'Cell junction': 8, 'Synapse': 7,
}

ligands = {
    'Metal-binding': 58, 'Nucleotide-binding': 42, 'ATP-binding': 30,
    'Zinc': 26, 'Calcium': 16, 'Magnesium': 11,
    'GTP-binding': 10, 'Iron': 9, 'SAM': 5,
    'Manganese': 4, 'Lipid-binding': 4, 'Potassium': 4,
}

diseases = {
    'Disease variant': 137, 'Intellectual disability': 30,
    'Neurodegeneration': 14, 'Proto-oncogene': 12,
    'Deafness': 8, 'Dwarfism': 8, 'Epilepsy': 8,
    'Hereditary hemolytic anemia': 6, 'Neuropathy': 6,
    'Fanconi anemia': 5, 'Cardiomyopathy': 5,
    'Ectodermal dysplasia': 4, 'Tumor suppressor': 4,
    'Leukodystrophy': 4,
}

# ── Figure: 4-panel functional diversity ──
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ── Panel A: Molecular Functions ──
ax = axes[0, 0]
items = sorted(mol_functions.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in items]
counts = [x[1] for x in items]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Proteins', fontsize=11, fontweight='bold')
ax.set_title('A. Molecular Functions', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=8, color='#333')

# ── Panel B: Cellular Components ──
ax = axes[0, 1]
items = sorted(cell_components.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in items]
counts = [x[1] for x in items]
colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(names)))
bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Proteins', fontsize=11, fontweight='bold')
ax.set_title('B. Subcellular Localization', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=8, color='#333')

# ── Panel C: Ligand Binding ──
ax = axes[1, 0]
items = sorted(ligands.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in items]
counts = [x[1] for x in items]
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(names)))
bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Proteins', fontsize=11, fontweight='bold')
ax.set_title('C. Ligand & Cofactor Binding', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=8, color='#333')

# ── Panel D: Disease Categories ──
ax = axes[1, 1]
items = sorted(diseases.items(), key=lambda x: x[1], reverse=True)
names = [x[0] for x in items]
counts = [x[1] for x in items]
colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(names)))
bars = ax.barh(range(len(names)), counts, color=colors, edgecolor='white', linewidth=0.5, zorder=3)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Number of Proteins', fontsize=11, fontweight='bold')
ax.set_title('D. Disease Associations', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            str(count), va='center', fontsize=8, color='#333')

fig.suptitle('Benchmark V2: Protein Family & Functional Diversity (207 Genes)',
             fontsize=14, fontweight='bold', y=1.01)

fig.savefig(FIG_DIR / 'fig9b_protein_families.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved: {FIG_DIR / 'fig9b_protein_families.png'}")
