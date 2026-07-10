#!/usr/bin/env python3
"""
Training Data Comparison for PLMs and GLMs used in V3 Benchmark.
Generates a multi-panel figure showing training data characteristics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data ──
models = {
    'ESM-1b':      {'type': 'PLM',  'params_M': 650,   'data_source': 'UniRef50', 'sequences_M': 30,    'tokens_B': 86,  'year': 2020},
    'ESM-2':       {'type': 'PLM',  'params_M': 650,   'data_source': 'UniRef50', 'sequences_M': 45,    'tokens_B': 86,  'year': 2022},
    'SaProt':      {'type': 'PLM',  'params_M': 650,   'data_source': 'UniRef50\n+ AlphaFoldDB', 'sequences_M': 40, 'tokens_B': 86, 'year': 2024},
    'ProtT5':      {'type': 'PLM',  'params_M': 3000,  'data_source': 'UniRef50', 'sequences_M': 45,    'tokens_B': 150, 'year': 2021},
    'NT-v2':       {'type': 'GLM',  'params_M': 500,   'data_source': '3,202 human\n+ 850 multispecies', 'sequences_M': 4052, 'tokens_B': 174, 'year': 2024},
    'HyenaDNA':    {'type': 'GLM',  'params_M': 1.6,   'data_source': 'Human genome\n(hg38)', 'sequences_M': 0.003, 'tokens_B': 3.1, 'year': 2023},
    'AlphaGenome': {'type': 'GLM',  'params_M': 250,   'data_source': 'ENCODE\nGTEx\n4D Nucleome\nFANTOM5', 'sequences_M': None, 'tokens_B': None, 'year': 2025},
}

# ── Colors ──
plm_color = '#3b82f6'   # blue
glm_color = '#10b981'   # green
plm_light = '#93c5fd'
glm_light = '#6ee7b7'

model_names = list(models.keys())
colors = [plm_color if models[m]['type'] == 'PLM' else glm_color for m in model_names]
light_colors = [plm_light if models[m]['type'] == 'PLM' else glm_light for m in model_names]

# ── Figure ──
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Training Data Comparison: Protein vs Genomic Language Models', fontsize=15, fontweight='bold', y=0.98)

# ═══ Panel A: Training Data Scale (log scale) ═══
ax = axes[0, 0]
# Use tokens (billions) - AlphaGenome estimated ~50B based on ENCODE scale
tokens = []
for m in model_names:
    t = models[m]['tokens_B']
    if t is None:
        t = 50  # AlphaGenome estimate based on ENCODE/GTEx/4D/FANTOM5 scale
    tokens.append(t)

bars = ax.bar(range(len(model_names)), tokens, color=colors, edgecolor='white', linewidth=1.2)
ax.set_yscale('log')
ax.set_ylabel('Training Tokens (Billions)', fontsize=10)
ax.set_title('A. Training Data Scale', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.set_ylim(1, 500)
ax.grid(axis='y', alpha=0.3)

# Annotate values
for i, (bar, val) in enumerate(zip(bars, tokens)):
    label = f'{val:.0f}B' if val >= 1 else f'{val*1000:.0f}M'
    if model_names[i] == 'AlphaGenome':
        label += '*'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.15, label,
            ha='center', va='bottom', fontsize=8, fontweight='bold')

# Legend
ax.legend(handles=[
    mpatches.Patch(color=plm_color, label='Protein LM'),
    mpatches.Patch(color=glm_color, label='Genomic LM'),
], fontsize=8, loc='upper left')

# ═══ Panel B: Training Data Source Comparison ═══
ax = axes[0, 1]
categories = ['UniRef50\n(Protein)', 'AlphaFoldDB\n(Structure)', 'Human Genomes\n(DNA)', 'Multi-species\nGenomes', 'ENCODE/\nRegulatory']
# Each model's contribution to each category
data_matrix = np.array([
    # UniRef50, AlphaFoldDB, Human Genomes, Multi-species, ENCODE
    [30,  0,   0,   0,   0],   # ESM-1b
    [45,  0,   0,   0,   0],   # ESM-2
    [40,  40,  0,   0,   0],   # SaProt (40M seqs + 40M AF2 structures)
    [45,  0,   0,   0,   0],   # ProtT5
    [0,   0,   3202, 850, 0],   # NT-v2 (3202 human + 850 multi)
    [0,   0,   1,    0,   0],   # HyenaDNA (1 human genome)
    [0,   0,   0,    0,   50],  # AlphaGenome (ENCODE + GTEx + 4DN + FANTOM5)
])

x = np.arange(len(categories))
width = 0.11
for i, (name, row) in enumerate(zip(model_names, data_matrix)):
    offset = (i - 3) * width
    ax.bar(x + offset, row, width, color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)

ax.set_yscale('log')
ax.set_ylabel('Scale (Millions of sequences/genomes)', fontsize=9)
ax.set_title('B. Training Data Sources', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=8)
ax.set_ylim(0.5, 10000)
ax.grid(axis='y', alpha=0.3)
ax.legend(model_names, fontsize=6.5, ncol=4, loc='upper right', framealpha=0.8)

# ═══ Panel C: Parameters vs Training Tokens ═══
ax = axes[1, 0]
params = [models[m]['params_M'] for m in model_names]
# Tokens for scatter, AlphaGenome = 50B estimate
tok_plot = [models[m]['tokens_B'] if models[m]['tokens_B'] else 50 for m in model_names]

scatter = ax.scatter(tok_plot, params, c=colors, s=200, edgecolors='white', linewidth=1.5, zorder=5)
for i, name in enumerate(model_names):
    ax.annotate(name, (tok_plot[i], params[i]), textcoords="offset points",
                xytext=(8, 5), fontsize=8, fontweight='bold')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Training Tokens (Billions)', fontsize=10)
ax.set_ylabel('Model Parameters (Millions)', fontsize=10)
ax.set_title('C. Model Size vs Training Data', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color=plm_color, label='Protein LM'),
    mpatches.Patch(color=glm_color, label='Genomic LM'),
], fontsize=8, loc='upper left')

# ═══ Panel D: Training Modality & Year ═══
ax = axes[1, 1]
years = [models[m]['year'] for m in model_names]
params_log = [np.log10(models[m]['params_M']) for m in model_names]

# Size proportional to tokens
tok_sizes = [models[m]['tokens_B'] if models[m]['tokens_B'] else 50 for m in model_names]
sizes = [max(80, t * 3) for t in tok_sizes]

scatter = ax.scatter(years, params_log, c=colors, s=sizes, alpha=0.8,
                     edgecolors='white', linewidth=1.5, zorder=5)
for i, name in enumerate(model_names):
    ax.annotate(name, (years[i], params_log[i]), textcoords="offset points",
                xytext=(10, 0), fontsize=8, fontweight='bold')

ax.set_xlabel('Publication Year', fontsize=10)
ax.set_ylabel('log₁₀(Parameters, M)', fontsize=10)
ax.set_title('D. Model Evolution Timeline', fontsize=12, fontweight='bold')
ax.set_xticks([2020, 2021, 2022, 2023, 2024, 2025])
ax.grid(alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color=plm_color, label='Protein LM'),
    mpatches.Patch(color=glm_color, label='Genomic LM'),
], fontsize=8, loc='upper left')

# Bubble size legend
for t_val, label in [(10, '10B'), (100, '100B')]:
    ax.scatter([], [], s=max(80, t_val*3), c='gray', alpha=0.4, label=f'{label} tokens')
ax.legend(fontsize=7, loc='lower right', framealpha=0.8)

# ── Footer ──
fig.text(0.5, 0.01,
         '*AlphaGenome tokens estimated from ENCODE/GTEx/4D Nucleome/FANTOM5 scale  |  AlphaGenome params from public disclosure',
         ha='center', fontsize=8, style='italic', color='#64748b')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmark_v3', 'figures')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'fig_training_data_comparison.png')
fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {out_path}")
