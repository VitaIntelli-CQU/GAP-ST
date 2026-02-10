import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr

def metric_func(preds_all, y_test, genes):
    per_gene = []
    n_nan = 0

    for i, gene in enumerate(genes):
        y_true = y_test[:, i]
        y_pred = preds_all[:, i]
        mse = np.mean((y_true - y_pred) ** 2)
        r2 = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-8)
        pearson, _ = pearsonr(y_true, y_pred)

        if np.isnan(pearson):
            n_nan += 1
            pearson = 0

        per_gene.append({
            "gene": gene,
            "mse": float(mse),
            "r2": float(r2),
            "pearson": float(pearson)
        })

    # all
    pearsons = np.array([g["pearson"] for g in per_gene])
    r2s = np.array([g["r2"] for g in per_gene])
    mses = np.array([g["mse"] for g in per_gene])

    fold_metrics = {
        "n_test": len(y_test),
        "metrics": {
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons)),
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s)),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses))
        },
        "quantiles": {
            "pearson_q1": float(np.percentile(pearsons, 25)),
            "pearson_q2": float(np.median(pearsons)),
            "pearson_q3": float(np.percentile(pearsons, 75)),
            "r2_q1": float(np.percentile(r2s, 25)),
            "r2_q2": float(np.median(r2s)),
            "r2_q3": float(np.percentile(r2s, 75)),
            "mse_q1": float(np.percentile(mses, 25)),
            "mse_q2": float(np.median(mses)),
            "mse_q3": float(np.percentile(mses, 75)),
        },
        "per_gene": per_gene
    }

    if n_nan > 0:
        print(f"Warning: {n_nan} genes had NaN Pearson correlations")
    return fold_metrics


def merge_fold_results(fold_results):
    gene_dict = {} 
    for fold in fold_results:
        for gene_data in fold["per_gene"]:
            gname = gene_data["gene"]
            gene_dict.setdefault(gname, []).append(gene_data["pearson"])

    per_gene_summary = []
    all_pearsons = []
    for gname, vals in gene_dict.items():
        vals = np.array(vals, dtype=float)
        mean = np.nanmean(vals)
        std = np.nanstd(vals)
        per_gene_summary.append({
            "gene": gname,
            "pearsons": vals.tolist(),
            "mean": float(mean),
            "std": float(std)
        })
        all_pearsons.extend(vals[np.isfinite(vals)])

    fold_means = [fold["metrics"]["pearson_mean"] for fold in fold_results]
    fold_stds = [fold["metrics"]["pearson_std"] for fold in fold_results]

    merged = {
        "kfold_summary": {
            "n_folds": len(fold_results),
            "pearson_mean_over_folds": float(np.mean(fold_means)),
            "pearson_std_over_folds": float(np.std(fold_means)),
            "fold_means": fold_means,
            "fold_stds": fold_stds,
        },
        "per_gene": per_gene_summary,
    }
    return merged



def merge_dataset_results(dataset_results):
    dataset_summaries = []
    pearson_means = []

    for res in dataset_results:
        name = res.get("dataset_name", "unknown")
        ksum = res["kfold_summary"]
        mean = ksum["pearson_mean_over_folds"]
        std = ksum["pearson_std_over_folds"]

        dataset_summaries.append({
            "dataset": name,
            "pearson_mean_over_folds": mean,
            "pearson_std_over_folds": std
        })
        pearson_means.append(mean)

    merged = {
        "n_datasets": len(dataset_results),
        "pearson_mean_over_datasets": float(np.mean(pearson_means)),
        "pearson_std_over_datasets": float(np.std(pearson_means)),
        "datasets": dataset_summaries
    }
    return merged

