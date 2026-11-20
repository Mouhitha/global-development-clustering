# Global Development Disparities: Statistical Analysis and Clustering

An exploratory analysis of socioeconomic development indicators across countries. The project examines associations among development measures and compares ways of grouping countries with similar indicator profiles.

**Publication:** [Global Development Disparities, IEEE I4C 2024](https://ieeexplore.ieee.org/document/10748509)

## Research questions

1. Which development indicators move together in this dataset?
2. How do K-Means, agglomerative clustering, spectral clustering, and DBSCAN differ in their groupings?
3. How sensitive are those groupings to scaling and dimensionality reduction?

## Data and methods

- Source workbook: `World_development_measurement.xlsx`.
- Methods investigated in the original notebook: hypothesis tests, standardized numeric indicators, PCA, t-SNE visualization, and several clustering algorithms.
- The original notebook is preserved in `legacy/` as a record of the coursework and research process.
- The new reproducible analysis is implemented in `analysis/run_clustering.py`.

See [data provenance](docs/DATA.md), [methodology](docs/METHODOLOGY.md), and [results and limitations](docs/RESULTS.md).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python analysis/run_clustering.py --input World_development_measurement.xlsx --output-dir outputs
```

Full-run status: PASSED on the checked-in workbook (2026-11-21); see [results and limitations](docs/RESULTS.md).

## Interpretation

These are exploratory, descriptive groupings, not rankings of countries and not causal evidence. Clustering labels are arbitrary. The repository does not use an unexplained `0.99 accuracy` as proof of predictive performance.

## Credit

This is academic project work; see the linked paper for the full author list and citation.
