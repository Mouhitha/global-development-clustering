"""Cluster country profiles built from the supplied development workbook."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SHEET_NAME = "world_development"
FEATURE_COLUMNS = (
    "Birth Rate",
    "Business Tax Rate",
    "CO2 Emissions",
    "Days to Start Business",
    "Ease of Business",
    "Energy Usage",
    "GDP",
    "Health Exp % GDP",
    "Health Exp/Capita",
    "Hours to do Tax",
    "Infant Mortality Rate",
    "Internet Usage",
    "Lending Interest",
    "Life Expectancy Female",
    "Life Expectancy Male",
    "Mobile Phone Usage",
    "Population 0-14",
    "Population 15-64",
    "Population 65+",
    "Population Total",
    "Population Urban",
    "Tourism Inbound",
    "Tourism Outbound",
)
N_CLUSTERS = 3
RANDOM_SEED = 42
KMEANS_STARTS = 20
PCA_VARIANCE = 0.90

NUMBER = (
    r"(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)
FORMATTED_NUMBER = rf"(?:[+-]?{NUMBER}%?|(?:[+-]?\$|\$[+-]){NUMBER})"


def read_workbook(path: Path) -> pd.DataFrame:
    """Read the named sheet, ignoring completely empty rows."""
    rows = pd.read_excel(
        path, sheet_name=SHEET_NAME, engine="openpyxl", keep_default_na=False, na_values=[""]
    )
    return rows.dropna(how="all")


def normalize_numeric(values: pd.Series) -> pd.Series:
    """Use fractions for explicit percent strings and keep bare numeric values as stored."""
    text = values.astype("string").str.strip()
    valid = text.str.fullmatch(FORMATTED_NUMBER, na=False)
    numeric_text = text.where(valid).str.replace(r"[$,%]", "", regex=True)
    numeric = pd.to_numeric(numeric_text, errors="coerce").astype(float)
    numeric.loc[text.str.endswith("%", na=False)] /= 100.0
    return numeric.where(np.isfinite(numeric))


def cluster_silhouette(matrix: np.ndarray, labels: np.ndarray) -> float | None:
    """Score assigned samples only, using the DBSCAN convention of -1 for noise."""
    assigned = labels != -1
    cluster_count = np.unique(labels[assigned]).size
    if 2 <= cluster_count < assigned.sum():
        return float(silhouette_score(matrix[assigned], labels[assigned]))
    return None


def analyze(rows: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    required = ("Country", *FEATURE_COLUMNS)
    missing_columns = [name for name in required if name not in rows]
    if missing_columns:
        raise ValueError(f"Required columns missing from {SHEET_NAME}: {', '.join(missing_columns)}")

    country = rows["Country"]
    if not country.map(lambda label: isinstance(label, str) and bool(label.strip())).all():
        raise ValueError("Every observation must have a nonblank text Country label.")
    country = country.str.strip()

    raw = rows.loc[:, list(FEATURE_COLUMNS)]
    blanks = raw.apply(lambda values: values.astype("string").str.fullmatch(r"\s*", na=False))
    raw = raw.mask(blanks)
    numeric = raw.apply(normalize_numeric)
    countries = numeric.groupby(country, sort=True).median()
    if len(countries) <= N_CLUSTERS:
        raise ValueError(f"At least {N_CLUSTERS + 1} distinct Country labels are required.")

    distinct = countries.nunique()
    excluded = {
        name: "entirely missing" if count == 0 else "constant among country medians"
        for name, count in distinct.items()
        if count < 2
    }
    retained = countries.drop(columns=list(excluded))
    if retained.shape[1] == 0:
        raise ValueError("No numeric features vary across the observed country medians.")

    fill_values = retained.median()
    scaled = StandardScaler().fit_transform(retained.fillna(fill_values))
    pca = {"applied": False, "reason": "Only one nonconstant feature remains."}
    matrix = scaled
    if scaled.shape[1] >= 2:
        model = PCA(n_components=PCA_VARIANCE, svd_solver="full")
        matrix = model.fit_transform(scaled)
        pca = {
            "applied": True,
            "variance_target": PCA_VARIANCE,
            "components_retained": int(model.n_components_),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "total_explained_variance_ratio": float(model.explained_variance_ratio_.sum()),
        }

    if np.unique(matrix, axis=0).shape[0] < N_CLUSTERS:
        raise ValueError(f"At least {N_CLUSTERS} distinct transformed profiles are required.")

    models = {
        "kmeans": KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=KMEANS_STARTS),
        "agglomerative": AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward"),
    }
    assignments = pd.DataFrame(index=countries.index)
    clustering = {}
    for name, model in models.items():
        labels = model.fit_predict(matrix)
        assignments[name] = labels
        cluster_ids, sizes = np.unique(labels, return_counts=True)
        clustering[name] = {
            "silhouette": cluster_silhouette(matrix, labels),
            "cluster_sizes": {str(label): int(size) for label, size in zip(cluster_ids, sizes)},
        }

    observations_per_country = country.value_counts()
    summary = {
        "input_rows": len(rows),
        "input_columns": len(rows.columns),
        "unique_country_labels": int(country.nunique()),
        "country_profiles": len(countries),
        "observations_per_country": {
            "min": int(observations_per_country.min()),
            "max": int(observations_per_country.max()),
        },
        "aggregation": "Median of available values per Country",
        "missingness": {
            "source_blanks_by_feature": raw.isna().sum().to_dict(),
            "invalid_nonblank_values_by_feature": (numeric.isna() & raw.notna()).sum().to_dict(),
            "missing_after_numeric_conversion_by_feature": numeric.isna().sum().to_dict(),
            "missing_country_medians_by_feature": countries.isna().sum().to_dict(),
        },
        "preprocessing": {
            "non_feature_columns": [name for name in rows if name not in FEATURE_COLUMNS],
            "excluded_features": excluded,
            "retained_features": retained.columns.tolist(),
            "imputation": "Median of available country medians for each retained feature",
            "imputed_values_by_feature": retained.isna().sum().to_dict(),
            "fully_imputed_country_profiles": int(retained.isna().all(axis=1).sum()),
            "scaling": "StandardScaler",
        },
        "pca": pca,
        "clustering_settings": {
            "n_clusters": N_CLUSTERS,
            "kmeans_random_seed": RANDOM_SEED,
            "kmeans_n_init": KMEANS_STARTS,
            "agglomerative_linkage": "ward",
            "silhouette_metric": "euclidean",
        },
        "clustering": clustering,
    }
    return summary, assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Path to the .xlsx workbook")
    parser.add_argument("--out", required=True, type=Path, help="Directory for the results")
    args = parser.parse_args()

    try:
        summary, assignments = analyze(read_workbook(args.data))
    except (OSError, ValueError) as error:
        parser.error(str(error))

    summary = {
        "workbook": args.data.name,
        "sheet": SHEET_NAME,
        "package_versions": {
            name: version(name) for name in ("numpy", "openpyxl", "pandas", "scikit-learn")
        },
        **summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    assignments.to_csv(args.out / "clusters.csv", lineterminator="\n")
    print(
        f"{summary['input_rows']} observations → {summary['country_profiles']} profiles "
        f"from {summary['unique_country_labels']} distinct Country labels."
    )
    print(f"Excluded numeric features: {summary['preprocessing']['excluded_features'] or 'none'}")
    print(f"Saved summary.json and clusters.csv in {args.out}")


if __name__ == "__main__":
    main()
