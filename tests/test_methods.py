import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import silhouette_score

from src.analysis import FEATURE_COLUMNS, analyze, cluster_silhouette


@pytest.fixture
def repeated_country_rows():
    rows = pd.DataFrame(1.0, index=range(7), columns=list(FEATURE_COLUMNS))
    rows["Country"] = ["A", "A", "A", "B", "C", "D", "E"]
    rows["GDP"] = [1, 1000, 3, 4, 21, 22, 40]
    rows["Population Urban"] = [0.1, 0.9, 0.2, 0.21, 0.5, 0.52, np.nan]
    rows["Business Tax Rate"] = np.nan
    return rows


@pytest.mark.parametrize("labels", [[-1, -1, -1], [0, 0, -1], [0, 1, -1]])
def test_dbscan_silhouette_guard(labels):
    matrix = np.array([[0.0], [1.0], [100.0]])
    assert cluster_silhouette(matrix, np.array(labels)) is None


def test_dbscan_silhouette_ignores_noise():
    matrix = np.array([[0.0], [1.0], [10.0], [11.0], [100.0]])
    labels = np.array([0, 0, 1, 1, -1])
    assert cluster_silhouette(matrix, labels) == pytest.approx(
        silhouette_score(matrix[:4], labels[:4])
    )


def test_kmeans_reproducibility(repeated_country_rows):
    first, first_assignments = analyze(repeated_country_rows)
    second, second_assignments = analyze(repeated_country_rows.sample(frac=1, random_state=3))
    assert first["clustering_settings"]["kmeans_random_seed"] == 42
    assert first["clustering"]["kmeans"] == second["clustering"]["kmeans"]
    pd.testing.assert_series_equal(first_assignments["kmeans"], second_assignments["kmeans"])


def test_medians_are_formed_before_counting_and_clustering_countries(repeated_country_rows):
    summary, assignments = analyze(repeated_country_rows)
    country_medians = repeated_country_rows.iloc[[0, 3, 4, 5, 6]].copy()
    country_medians.loc[0, ["GDP", "Population Urban"]] = [3, 0.2]
    _, expected_assignments = analyze(country_medians)

    assert summary["input_rows"] == 7
    assert summary["country_profiles"] == summary["unique_country_labels"] == 5
    assert summary["observations_per_country"] == {"min": 1, "max": 3}
    assert assignments.index.tolist() == ["A", "B", "C", "D", "E"]
    excluded = summary["preprocessing"]["excluded_features"]
    assert excluded["Business Tax Rate"] == "entirely missing"
    assert excluded["Birth Rate"] == "constant among country medians"
    assert summary["preprocessing"]["imputed_values_by_feature"] == {"GDP": 0, "Population Urban": 1}
    pd.testing.assert_frame_equal(assignments, expected_assignments)


def test_numeric_country_surrogate_cannot_change_clusters(repeated_country_rows):
    _, expected = analyze(repeated_country_rows)
    with_identifiers = repeated_country_rows.assign(
        Country_encoded=[10, 10, 10, 100, 2, 35, 75],
        **{"Number of Records": [3, 9, 1, 4, 8, 7, 2]},
    )
    summary, actual = analyze(with_identifiers)
    assert set(summary["preprocessing"]["non_feature_columns"]) == {
        "Country", "Country_encoded", "Number of Records",
    }
    pd.testing.assert_frame_equal(actual, expected)
