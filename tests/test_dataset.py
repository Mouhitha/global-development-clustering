from pathlib import Path

import pandas as pd
import pytest

from src.analysis import FEATURE_COLUMNS, analyze, normalize_numeric, read_workbook

WORKBOOK = Path(__file__).resolve().parents[1] / "World_development_measurement.xlsx"


@pytest.fixture(scope="module")
def workbook_rows():
    return read_workbook(WORKBOOK)


def test_workbook_shape_and_observation_unit(workbook_rows):
    assert workbook_rows.shape == (2704, 25)
    assert workbook_rows["Country"].nunique() == 208
    assert workbook_rows["Country"].value_counts().eq(13).all()
    assert not any("year" in column.casefold() for column in workbook_rows.columns)


def test_workbook_missingness(workbook_rows):
    expected = {
        "GDP": 210,
        "Business Tax Rate": 1281,
        "Health Exp % GDP": 309,
        "Population Urban": 26,
    }
    for feature, missing in expected.items():
        assert workbook_rows[feature].isna().sum() == missing
        assert normalize_numeric(workbook_rows[feature]).isna().sum() == missing


def test_numeric_normalization():
    parsed = normalize_numeric(
        pd.Series([" $9,129,594,819 ", "76.9%", 0.769, "0.06", "-$1,200.50", "1.2E3"])
    )
    assert parsed.tolist() == pytest.approx([9129594819, 0.769, 0.769, 0.06, -1200.50, 1200])
    invalid = normalize_numeric(pd.Series([None, " ", "unknown", "1,00", "$5%", "1.000,25", "inf"]))
    assert invalid.isna().all()


def test_country_identifiers_and_record_counts_are_outside_workbook_features(workbook_rows):
    assert set(FEATURE_COLUMNS) == set(workbook_rows.columns) - {"Country", "Number of Records"}
    assert "Country_encoded" not in FEATURE_COLUMNS


@pytest.mark.parametrize("column", ["Country", "GDP"])
def test_missing_required_columns_are_reported(workbook_rows, column):
    with pytest.raises(ValueError, match=f"Required columns missing.*{column}"):
        analyze(workbook_rows.drop(columns=column))
