"""Reproducible, read-only clustering analysis of the checked-in workbook.

Run from the repository root:

    python analysis/run_clustering.py --input World_development_measurement.xlsx --output-dir outputs

All comparisons use the same median-imputed, standardized numeric representation.
No dimensionality reduction, external labels, or self-referential "accuracy" metrics
are used. See docs/METHODOLOGY.md for the reasons.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, SpectralClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Confirmed exact header in World_development_measurement.xlsx. The aliases are
# checked by exact, case-insensitive name only; no fuzzy guessing of identifiers.
COUNTRY_COLUMN = "Country"
COUNTRY_COLUMN_ALIASES = ("Country Name", "CountryName", "Country Code")
YEAR_COLUMN_CANDIDATES = ("Year", "Date")
IDENTIFIER_COLUMNS = frozenset(
    {"country", "country name", "countryname", "country code", "year", "date"}
)

RANDOM_STATE = 42
K_VALUES = (2, 3, 4, 5, 6)
N_INIT = 10  # explicit, avoids version-dependent KMeans defaults
MIN_SAMPLES = 5
# Fixed a priori for standardized Euclidean distances in roughly 18 dimensions;
# it is not tuned to maximize a reported score.
EPS_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
SPECTRAL_NEIGHBORS = 10
IMPUTE_STRATEGY = "median"

RESULT_COLUMNS = (
    "method",
    "parameters",
    "n_clusters",
    "n_noise",
    "noise_fraction",
    "cluster_size_distribution",
    "silhouette_score",
    "silhouette_status",
    "fit_status",
)


def discover_country_column(columns: pd.Index | list[str]) -> str | None:
    """Return the exact country-identifier header, or None if not identifiable."""
    by_lower = {str(col).strip().lower(): col for col in columns}
    for candidate in (COUNTRY_COLUMN, *COUNTRY_COLUMN_ALIASES):
        match = by_lower.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def discover_year_column(columns: pd.Index | list[str]) -> str | None:
    by_lower = {str(col).strip().lower(): col for col in columns}
    for candidate in YEAR_COLUMN_CANDIDATES:
        match = by_lower.get(candidate.lower())
        if match is not None:
            return str(match)
    return None


def load_data(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the workbook and log its raw shape and country coverage.

    Duplicate country identifiers are retained unless an explicit country-year
    definition is present and violated. This workbook has no year column, so its
    13 rows per country are treated as repeated observations rather than guessed
    as distinct time points.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Input workbook not found: {source}")

    df = pd.read_excel(source)
    if df.empty:
        raise ValueError(f"Input workbook has no rows: {source}")

    country_column = discover_country_column(df.columns)
    year_column = discover_year_column(df.columns)
    metadata: dict[str, Any] = {
        "source_file": source.name,
        "raw_shape": list(df.shape),
        "country_column": country_column,
        "country_count": None,
        "year_column": year_column,
        "country_label_status": "OK",
        "duplicate_policy": (
            "Rows retained: no explicit year column, so one row per country "
            "is not assumed."
        ),
    }

    if country_column is None:
        metadata["country_label_status"] = (
            "BLOCKED: no identifiable country-identifier column; "
            "country labels omitted rather than guessed."
        )
        logger.warning(metadata["country_label_status"])
    else:
        country_count = int(df[country_column].nunique(dropna=True))
        metadata["country_count"] = country_count
        duplicate_rows = int(df.duplicated(subset=[country_column]).sum())
        metadata["duplicate_country_rows"] = duplicate_rows
        if year_column is not None:
            duplicate_country_year = int(
                df.duplicated(subset=[country_column, year_column]).sum()
            )
            if duplicate_country_year:
                raise ValueError(
                    "Data definition requires one row per country-year, but "
                    f"{duplicate_country_year} duplicate country-year rows were found."
                )
            metadata["duplicate_policy"] = (
                "Rows rejected on duplicate country-year identifiers only."
            )
        else:
            metadata["rows_per_country"] = {
                str(int(rows)): int(count)
                for rows, count in df[country_column]
                .value_counts(dropna=False)
                .value_counts()
                .sort_index()
                .items()
            }
            if duplicate_rows:
                logger.warning(
                    "Duplicate country identifiers retained (%s of %s rows); "
                    "rows are not independent country observations.",
                    duplicate_rows,
                    len(df),
                )

    logger.info("Loaded %s with raw shape %s", source.name, df.shape)
    logger.info("Country count: %s", metadata["country_count"])
    return df, metadata


def prepare_features(
    df: pd.DataFrame, country_column: str | None = None
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Select numeric indicators, drop degenerate columns, impute, and scale.

    Returns the transformed matrix, the retained feature names, and a report.
    Currency- and percentage-formatted strings are not silently parsed; they are
    reported as excluded so any future inclusion needs a documented conversion.
    """
    country = country_column if country_column is not None else discover_country_column(df.columns)
    excluded: dict[str, str] = {}
    candidates: list[str] = []

    for column in df.columns:
        name = str(column)
        normalized = name.strip().lower()
        if country is not None and name == country:
            excluded[name] = "country identifier, not a numeric indicator"
        elif normalized in IDENTIFIER_COLUMNS:
            excluded[name] = "identifier/time column, not a numeric indicator"
        elif not pd.api.types.is_numeric_dtype(df[name]):
            excluded[name] = f"non-numeric dtype ({df[name].dtype}); not parsed"
        else:
            candidates.append(name)

    kept: list[str] = []
    dropped: dict[str, str] = {}
    for column in candidates:
        present = int(df[column].notna().sum())
        if present == 0:
            dropped[column] = "all values missing"
        elif int(df[column].nunique(dropna=True)) <= 1:
            dropped[column] = "zero variance (single observed value)"
        else:
            kept.append(column)

    if not kept:
        raise ValueError("No usable numeric indicator columns remain after preprocessing.")

    raw = df[kept]
    imputer = SimpleImputer(strategy=IMPUTE_STRATEGY)
    imputed = imputer.fit_transform(raw)
    if not np.isfinite(imputed).all():
        raise ValueError("Non-finite values remain after imputation; cannot scale safely.")

    scaled = StandardScaler().fit_transform(imputed)

    report: dict[str, Any] = {
        "strategy": IMPUTE_STRATEGY,
        "scale": "StandardScaler (zero mean, unit variance)",
        "numeric_candidate_columns": candidates,
        "included_features": kept,
        "excluded_columns": excluded,
        "dropped_columns": dropped,
        "missing_counts": {col: int(df[col].isna().sum()) for col in kept},
        "imputed_value_counts": {
            col: int(df[col].isna().sum()) for col in kept if int(df[col].isna().sum())
        },
        "imputation_statistics": {
            col: float(value) for col, value in zip(kept, imputer.statistics_)
        },
        "scaled_matrix_shape": list(scaled.shape),
    }
    return scaled, kept, report


def silhouette_for_labels(
    X: np.ndarray, labels: np.ndarray
) -> tuple[float | None, str]:
    """Compute silhouette only when at least two non-noise clusters exist."""
    labels = np.asarray(labels)
    non_noise = labels != -1
    non_noise_labels = labels[non_noise]
    n_clusters = int(len(np.unique(non_noise_labels)))
    n_non_noise = int(non_noise.sum())

    if n_clusters < 2:
        return None, "skipped: fewer than two non-noise clusters"
    if n_clusters >= n_non_noise:
        return None, "skipped: non-noise cluster count >= non-noise sample count"
    return (
        float(silhouette_score(np.asarray(X)[non_noise], non_noise_labels)),
        "computed on non-noise rows only",
    )


def _result_row(
    X: np.ndarray,
    method: str,
    parameters: str,
    labels: np.ndarray,
    fit_status: str = "ok",
) -> dict[str, Any]:
    labels = np.asarray(labels)
    non_noise = labels[labels != -1]
    sizes = pd.Series(non_noise).value_counts().sort_index()
    score, status = silhouette_for_labels(X, labels)
    return {
        "method": method,
        "parameters": parameters,
        "n_clusters": int(len(sizes)),
        "n_noise": int((labels == -1).sum()),
        "noise_fraction": float((labels == -1).mean()),
        "cluster_size_distribution": json.dumps(
            {str(int(label)): int(count) for label, count in sizes.items()},
            separators=(",", ":"),
        ),
        "silhouette_score": score,
        "silhouette_status": status,
        "fit_status": fit_status,
    }


def run_models(
    X: np.ndarray,
    *,
    k_values: tuple[int, ...] = K_VALUES,
    eps_grid: tuple[float, ...] = EPS_GRID,
    include_spectral: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Fit all models on the same scaled X and return summary rows and labels."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must be a non-empty 2-D numeric matrix.")

    results: list[dict[str, Any]] = []
    labels_by_method: dict[str, np.ndarray] = {}
    n_samples = X.shape[0]

    for k in k_values:
        if k >= n_samples:
            results.append(
                {
                    "method": "unavailable",
                    "parameters": f"k={k} (n_samples={n_samples})",
                    "n_clusters": None,
                    "n_noise": None,
                    "noise_fraction": None,
                    "cluster_size_distribution": "",
                    "silhouette_score": None,
                    "silhouette_status": "skipped: k >= n_samples",
                    "fit_status": "skipped",
                }
            )
            continue

        kmeans_labels = KMeans(
            n_clusters=k, init="k-means++", n_init=N_INIT, random_state=RANDOM_STATE
        ).fit_predict(X)
        results.append(
            _result_row(
                X,
                f"KMeans_k{k}",
                f"n_clusters={k}, init=k-means++, n_init={N_INIT}, random_state={RANDOM_STATE}",
                kmeans_labels,
            )
        )
        labels_by_method[f"KMeans_k{k}"] = kmeans_labels

        agg_labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
        results.append(
            _result_row(
                X,
                f"Agglomerative_k{k}",
                f"n_clusters={k}, linkage=ward (default), metric=euclidean (default)",
                agg_labels,
            )
        )
        labels_by_method[f"Agglomerative_k{k}"] = agg_labels

        if include_spectral:
            n_neighbors = min(SPECTRAL_NEIGHBORS, n_samples - 1)
            if n_neighbors >= n_samples:
                results.append(
                    {
                        "method": f"Spectral_k{k}",
                        "parameters": f"n_neighbors={n_neighbors}, k={k}",
                        "n_clusters": None,
                        "n_noise": None,
                        "noise_fraction": None,
                        "cluster_size_distribution": "",
                        "silhouette_score": None,
                        "silhouette_status": "skipped: n_neighbors >= n_samples",
                        "fit_status": "skipped",
                    }
                )
                continue
            try:
                spectral_labels = SpectralClustering(
                    n_clusters=k,
                    affinity="nearest_neighbors",
                    n_neighbors=n_neighbors,
                    assign_labels="kmeans",
                    random_state=RANDOM_STATE,
                ).fit_predict(X)
                results.append(
                    _result_row(
                        X,
                        f"Spectral_k{k}",
                        (
                            f"n_clusters={k}, affinity=nearest_neighbors, "
                            f"n_neighbors={n_neighbors}, assign_labels=kmeans, "
                            f"random_state={RANDOM_STATE}"
                        ),
                        spectral_labels,
                    )
                )
                labels_by_method[f"Spectral_k{k}"] = spectral_labels
            except Exception as exc:  # reported explicitly, never silently omitted
                results.append(
                    {
                        "method": f"Spectral_k{k}",
                        "parameters": (
                            f"n_clusters={k}, affinity=nearest_neighbors, "
                            f"n_neighbors={n_neighbors}"
                        ),
                        "n_clusters": None,
                        "n_noise": None,
                        "noise_fraction": None,
                        "cluster_size_distribution": "",
                        "silhouette_score": None,
                        "silhouette_status": "skipped: fit failed",
                        "fit_status": f"failed: {type(exc).__name__}: {exc}",
                    }
                )

    for eps in eps_grid:
        dbscan_labels = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(X)
        results.append(
            _result_row(
                X,
                f"DBSCAN_eps{eps:g}",
                f"eps={eps:g}, min_samples={MIN_SAMPLES}",
                dbscan_labels,
            )
        )
        labels_by_method[f"DBSCAN_eps{eps:g}"] = dbscan_labels

    return results, labels_by_method


def write_outputs(
    output_dir: str | Path,
    *,
    results: list[dict[str, Any]],
    labels: dict[str, np.ndarray],
    preprocessing_summary: dict[str, Any],
    load_metadata: dict[str, Any],
    country_values: pd.Series | None,
) -> list[Path]:
    """Write comparison, preprocessing, and per-row cluster assignment files."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    comparison = pd.DataFrame(results, columns=RESULT_COLUMNS)
    comparison_path = target / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    summary_path = target / "preprocessing_summary.json"
    summary_payload = {
        "data_load": load_metadata,
        "preprocessing": preprocessing_summary,
        "models": {
            "k_values": list(K_VALUES),
            "kmeans_n_init": N_INIT,
            "random_state": RANDOM_STATE,
            "dbscan_eps_grid": list(EPS_GRID),
            "dbscan_min_samples": MIN_SAMPLES,
            "spectral": (
                "affinity=nearest_neighbors, n_neighbors=min(10, n_samples-1), "
                "assign_labels=kmeans, random_state=42"
            ),
            "representation": (
                "All scores are computed on the same median-imputed, "
                "standardized numeric matrix; no PCA/t-SNE score comparisons."
            ),
            "metrics": (
                "Silhouette only when 2 <= non-noise clusters < non-noise samples. "
                "No accuracy column: cluster labels have no external ground truth."
            ),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")

    n_rows = len(country_values) if country_values is not None else len(next(iter(labels.values())))
    assignments: dict[str, Any] = {"row_index": np.arange(n_rows)}
    if country_values is not None:
        assignments["country"] = list(country_values)
    else:
        assignments["country"] = ""
    assignments["country_label_status"] = load_metadata["country_label_status"]
    for method, label_values in labels.items():
        assignments[method] = np.asarray(label_values)

    clusters_path = target / "country_clusters.csv"
    pd.DataFrame(assignments).to_csv(clusters_path, index=False)
    return [comparison_path, summary_path, clusters_path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare clustering models on standardized development indicators."
    )
    default_input = Path(__file__).resolve().parents[1] / "World_development_measurement.xlsx"
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        df, load_metadata = load_data(args.input)
        X, feature_names, preprocessing_report = prepare_features(
            df, load_metadata["country_column"]
        )
        results, labels = run_models(X)
        country_values = (
            df[load_metadata["country_column"]]
            if load_metadata["country_column"] is not None
            else None
        )
        paths = write_outputs(
            args.output_dir,
            results=results,
            labels=labels,
            preprocessing_summary=preprocessing_report,
            load_metadata=load_metadata,
            country_values=country_values,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("Run failed: %s", exc)
        return 1

    computed = [row for row in results if row["silhouette_score"] is not None]
    print(f"Rows: {load_metadata['raw_shape'][0]} | countries: {load_metadata['country_count']}")
    print(f"Numeric features used: {len(feature_names)}")
    print(f"Model rows: {len(results)} | silhouette scores computed: {len(computed)}")
    print(f"Outputs: {', '.join(str(path) for path in paths)}")
    if any(row["fit_status"].startswith("failed") for row in results):
        print("Some model fits failed; see fit_status in model_comparison.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
