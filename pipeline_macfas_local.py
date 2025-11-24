import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr
from sklearn.decomposition import PCA


def read_fasta(path):
    sequences = {}
    with open(path, "r") as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]  # ID limpio
                sequences[current_id] = ""
            else:
                sequences[current_id] += line
    return sequences


fasta_file = "Macaca_fascicularis.Macaca_fascicularis_6.0.cdna.all.fa"    
seqs = read_fasta(fasta_file)
gene_ids = list(seqs.keys())

print(f"Se cargaron {len(gene_ids)} secuencias del FASTA")

# ========================================================
# 2. METODOLOGÍA ORIGINAL 
# ========================================================

np.random.seed(0)

# Elegimos los primeros 3 genes del FASTA
genes_original = gene_ids[:3]

n = 20  # muestras

# Ct simulados
Ct_control = {g: np.random.normal(25, 1, n) for g in genes_original}
Ct_case = {g: np.random.normal(24, 1, n) for g in genes_original}

HK_control = np.random.normal(20, 0.5, n)
HK_case = np.random.normal(20, 0.5, n)

deltaCt_control = {g: Ct_control[g] - HK_control for g in genes_original}
deltaCt_case = {g: Ct_case[g] - HK_case for g in genes_original}

results_qpcr = {
    g: ttest_ind(deltaCt_control[g], deltaCt_case[g]).pvalue
    for g in genes_original
}

# ========================================================
# 3. METODOLOGÍA PROPUESTA (RNA-seq simulada usando genes del FASTA)
# ========================================================

G = len(gene_ids)  # número real de genes del FASTA

# Simulación de counts (los genes reales definen columnas)
counts_control = np.random.poisson(lam=50, size=(n, G))
counts_case = np.random.poisson(lam=55, size=(n, G))

counts = pd.DataFrame(
    np.vstack([counts_control, counts_case]),
    columns=gene_ids
)
groups = np.array([0]*n + [1]*n)

# --- VOOM ---
logCPM = np.log2((counts + 0.5) / counts.sum(axis=1).values[:, None] * 1e6)
variances = logCPM.var(axis=0).values
weights = 1 / (variances + 1e-6)

# --- LIMMA simplificado ---
betas = []
pvals = []

for g in gene_ids:
    gene_exp = logCPM[g]
    beta = np.mean(gene_exp[groups == 1]) - np.mean(gene_exp[groups == 0])
    pv = ttest_ind(gene_exp[groups == 0], gene_exp[groups == 1]).pvalue
    betas.append(beta)
    pvals.append(pv)

# --- GSEA (vía artificial = primeros 20 genes del FASTA) ---
pathway = gene_ids[:20]
ranking = np.argsort(np.abs(betas))[::-1]
ES = sum([1 if gene_ids[i] in pathway else -1 for i in ranking[:50]])

# --- WGCNA simplificado ---
subset = logCPM.iloc[:, :min(50, G)]   # máximo 50 genes
pca = PCA(n_components=1)
module_eigengene = pca.fit_transform(subset)
corr_mod, _ = pearsonr(module_eigengene[:, 0], groups)

# ========================================================
# RESULTADOS FINALES
# ========================================================

results = {
    "QPCR_ttests": results_qpcr,
    "RNAseq_significant_genes_p<0.05": int(np.sum(np.array(pvals) < 0.05)),
    "GSEA_enrichment_score": ES,
    "Module_group_correlation": corr_mod
}

print("\nRESULTADOS:")
print(results)
