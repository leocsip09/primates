import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, pearsonr, shapiro
from sklearn.decomposition import PCA

def read_fasta(path):
    """
    Lee un archivo FASTA y devuelve un diccionario:
    {ID : secuencia}
    """
    sequences = {}
    with open(path, "r") as f:
        current_id = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:].split()[0]
                sequences[current_id] = ""
            else:
                sequences[current_id] += line
    return sequences

fasta_file = "Macaca_fascicularis.Macaca_fascicularis_6.0.cdna.all.fa"
seqs = read_fasta(fasta_file)
gene_ids = list(seqs.keys())

print(f"\n[INFO] Se cargaron {len(gene_ids)} secuencias del FASTA.\n")

np.random.seed(0)
n = 20  # muestras por condición
genes_original = gene_ids[:3]

print("[INFO] Genes evaluados originalmente (qPCR):")
print(genes_original)

# Simulación de Cts
Ct_control = {g: np.random.normal(25, 1, n) for g in genes_original}
Ct_case    = {g: np.random.normal(24, 1, n) for g in genes_original}

HK_control = np.random.normal(20, 0.5, n)
HK_case    = np.random.normal(20, 0.5, n)

deltaCt_control = {g: Ct_control[g] - HK_control for g in genes_original}
deltaCt_case    = {g: Ct_case[g] - HK_case for g in genes_original}

results_qpcr = {}
for g in genes_original:
    stat, p = ttest_ind(deltaCt_control[g], deltaCt_case[g])
    W, p_sh = shapiro(np.concatenate([deltaCt_control[g], deltaCt_case[g]]))
    results_qpcr[g] = {"pvalue_ttest": p, "pvalue_shapiro": p_sh}

# voom + limma

G = len(gene_ids)
counts_control = np.random.poisson(50, size=(n, G))
counts_case    = np.random.poisson(55, size=(n, G))

counts = pd.DataFrame(
    np.vstack([counts_control, counts_case]),
    columns=gene_ids
)
groups = np.array([0]*n + [1]*n)

# ------------------ VOOM ------------------
logCPM = np.log2((counts + 0.5) / counts.sum(axis=1).values[:, None] * 1e6)
variances = logCPM.var(axis=0)
weights = 1 / (variances + 1e-6)

# ------------------ LIMMA ---------------
betas = []
pvals = []
for g in gene_ids:
    gene_exp = logCPM[g]
    beta = np.mean(gene_exp[groups == 1]) - np.mean(gene_exp[groups == 0])
    pv = ttest_ind(gene_exp[groups == 0], gene_exp[groups == 1]).pvalue
    betas.append(beta)
    pvals.append(pv)

betas = np.array(betas)
pvals = np.array(pvals)

significant = np.sum(pvals < 0.05)

# GSEA

pathway = gene_ids[:20]
ranking = np.argsort(np.abs(betas))[::-1]  # ranking por |beta|
ES = sum([1 if gene_ids[i] in pathway else -1 for i in ranking[:50]])

# WGCNA

subset = logCPM.iloc[:, :min(50, G)]
pca = PCA(n_components=1)
eigengene = pca.fit_transform(subset)
corr_mod, p_corr = pearsonr(eigengene[:, 0], groups)

interpretation = {}

# --- qPCR ---
interp_qpcr = {}
for g, vals in results_qpcr.items():
    if vals["pvalue_ttest"] < 0.05:
        interp_qpcr[g] = f"{g} muestra diferencia significativa entre grupos."
    else:
        interp_qpcr[g] = f"{g} no presenta diferencia significativa."
interpretation["qPCR"] = interp_qpcr

# --- RNA-seq ---
interpretation["RNAseq"] = (
    f"Se encontraron {significant} genes diferencialmente expresados (p<0.05) con Limma. "
    + ("Indica impacto global claro." if significant > 200 else "Cambios moderados.")
)

# --- GSEA ---
if ES > 0:
    interpretation["GSEA"] = "El conjunto génico evaluado está enriquecido en los genes más alterados."
else:
    interpretation["GSEA"] = "No hay enriquecimiento detectable en el conjunto génico evaluado."

# --- WGCNA ---
if abs(corr_mod) > 0.4:
    interpretation["WGCNA"] = (
        f"Existe un módulo fuertemente correlado con el fenotipo (r={corr_mod:.2f})."
    )
else:
    interpretation["WGCNA"] = (
        f"No se detecta correlación marcada entre módulos y fenotipo (r={corr_mod:.2f})."
    )

# ========================================================
# 7. RESULTADOS FINALES
# ========================================================

results = {
    "QPCR_ttests": results_qpcr,
    "RNAseq_significant_genes_p<0.05": int(significant),
    "GSEA_enrichment_score": ES,
    "Module_group_correlation": corr_mod,
    "Interpretation": interpretation
}

print("\n" + "="*60)
print("                RESULTADOS FINALES")
print("="*60 + "\n")

# -------- qPCR --------
print("RESULTADOS RT-qPCR\n")
for g, vals in results_qpcr.items():
    print(f"  • {g}")
    print(f"      - p-value (t-test):     {vals['pvalue_ttest']:.3e}")
    print(f"      - p-value (Shapiro):    {vals['pvalue_shapiro']:.3e}")
    print()

# -------- RNA-seq --------
print("RNA-seq (voom + limma)\n")
print(f"  • Genes significativamente expresados (p < 0.05): {significant}")
print()

# -------- GSEA --------
print("GSEA\n")
print(f"  • Enrichment Score (ES): {ES}")
print()

# -------- WGCNA --------
print("WGCNA\n")
print(f"  • Correlación módulo–grupo: r = {corr_mod:.3f}")
print()

# -------- Interpretación automática --------
print("INTERPRETACIÓN AUTOMÁTICA\n")

print("  → qPCR:")
for g, desc in interpretation["qPCR"].items():
    print(f"      - {desc}")
print()

print(f"  → RNA-seq:\n      - {interpretation['RNAseq']}\n")

print(f"  → GSEA:\n      - {interpretation['GSEA']}\n")

print(f"  → WGCNA:\n      - {interpretation['WGCNA']}\n")

print("="*60 + "\n")

