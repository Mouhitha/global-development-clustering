"""Synthetic tests for the canonical clustering pipeline (no workbook needed)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.run_clustering import (
    N_INIT,
    prepare_features,
    run_models,
    silhouette_for_labels,
    write_outputs,
)


def _synthetic_X() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
            [10.1, 10.0],
        ]
    )


def test_numeric_only_selection_and_degenerate_drops() -> None:
    df = pd.DataFrame(
        {
            "Country": ["A", "B", "C", "D", "E"],
            "income": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rate": [5.0, 4.0, 3.0, 2.0, 1.0],
            "constant": [1.0] * 5,
            "all_missing": [np.nan] * 5,
            "GDP": ["$1", "$2", "$3", "$4", "$5"],  # string formatting, not parsed
        }
    )
    X, names, report = prepare_features(df)

    assert names == ["income", "rate"]
    assert X.shape == (5, 2)
    assert report["excluded_columns"]["Country"].startswith("country identifier")
    assert report["excluded_columns"]["GDP"].startswith("non-numeric dtype")
    assert report["dropped_columns"]["constant"].startswith("zero variance")
    assert report["dropped_columns"]["all_missing"].startswith("all values missing")


def test_missing_value_imputation_uses_median() -> None:
    df = pd.DataFrame(
        {
            "income": [1.0, 3.0, np.nan, 7.0, 9.0],  # median = 5
            "other": [2.0, 4.0, 6.0, 8.0, 10.0],
        }
    )
    X, names, report = prepare_features(df)

    assert names == ["income", "other"]
    assert np.isfinite(X).all()
    assert report["imputed_value_counts"] == {"income": 1}
    assert report["imputation_statistics"]["income"] == 5.0


def test_silhouette_skips_invalid_cluster_configurations() -> None:
    X = _synthetic_X()
    all_noise = np.array([-1, -1, -1, -1, -1, -1])
    one_cluster = np.array([0, 0, 0, 0, 0, 0])
    two_clusters = np.array([0, 0, 0, 1, 1, 1])
    singleton_clusters = np.array([0, 1, 2, 3, 4, 5])

    assert silhouette_for_labels(X, all_noise) == (
        None,
        "skipped: fewer than two non-noise clusters",
    )
    assert silhouette_for_labels(X, one_cluster) == (
        None,
        "skipped: fewer than two non-noise clusters",
    )
    score, status = silhouette_for_labels(X, two_clusters)
    assert score is not None and status == "computed on non-noise rows only"
    assert silhouette_for_labels(X, singleton_clusters)[0] is None


def test_non_noise_cluster_counts_match_labels() -> None:
    X = _synthetic_X()
    results, labels = run_models(
        X, k_values=(2, 3), eps_grid=(1.0, 20.0), include_spectral=False
    )

    assert results
    for row in results:
        label_values = labels[row["method"]]
        assert row["n_clusters"] == len(set(label_values.tolist()) - {-1})
        assert row["noise_fraction"] == float((label_values == -1).mean())
        if row["silhouette_score"] is None:
            assert row["silhouette_status"].startswith("skipped")


def test_kmeans_is_deterministic_and_uses_explicit_n_init() -> None:
    X = _synthetic_X()
    first, first_labels = run_models(
        X, k_values=(3,), eps_grid=(5.0,), include_spectral=False
    )
    second, second_labels = run_models(
        X, k_values=(3,), eps_grid=(5.0,), include_spectral=False
    )

    np.testing.assert_array_equal(first_labels["KMeans_k3"], second_labels["KMeans_k3"])
    assert f"n_init={N_INIT}" in first[0]["parameters"]


def test_csv_and_json_outputs(tmp_path) -> None:
    X = _synthetic_X()
    results, labels = run_models(
        X, k_values=(2,), eps_grid=(5.0,), include_spectral=False
    )
    summary = {"included_features": ["x", "y"]}
    load_metadata = {
        "raw_shape": [6, 3],
        "country_count": 6,
        "country_label_status": "OK",
    }
    paths = write_outputs(
        tmp_path,
        results=results,
        labels=labels,
        preprocessing_summary=summary,
        load_metadata=load_metadata,
        country_values=pd.Series(["A", "B", "C", "D", "E", "F"]),
    )

    comparison = pd.read_csv(paths[0])
    assert len(comparison) == len(results)
    assert "accuracy" not in comparison.columns
    assert set(comparison["method"]) == set(labels)

    saved_summary = json.loads(paths[1].read_text(encoding="utf-8"))
    assert saved_summary["preprocessing"] == summary

    assignments = pd.read_csv(paths[2])
    assert assignments["country"].tolist() == ["A", "B", "C", "D", "E", "F"]
    assert "KMeans_k2" in assignments.columns
    assert len(assignments) == 6
