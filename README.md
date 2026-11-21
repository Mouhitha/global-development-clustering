# Global Development Disparities: Statistical Analysis and Clustering

This project explores cross-country socioeconomic differences using a workbook with 2,704 observed rows, 208 unique `Country` labels, and no year field. The historical notebook records hypothesis tests, tests of association, and clustering experiments associated with the published study. The country-level PCA, K-means, and Ward clustering below form a separate reanalysis. [IEEE I4C 2024 paper](https://ieeexplore.ieee.org/document/10748509)

## Run the country-level reanalysis

Run these commands from the repository root with Python 3.12.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.analysis --data World_development_measurement.xlsx --out results/
```

The script takes the median of each available indicator per country label, fills missing country medians with the feature median across countries, and standardizes the features. Both methods use three clusters on the same PCA scores.

## Results

K-means and Ward produced nearly equal silhouette scores of **0.3646** and **0.3631**, respectively. Silhouette measures cohesion and separation among the fitted country profiles.

- The smallest K-means cluster (ID `0`) contains China and the United States.
- The smallest Ward cluster (ID `2`) contains China, India, and the United States.

The IDs are specific to each method. [Country assignments](results/clusters.csv) · [Summary metrics](results/summary.json)

## Tests

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Limitations

- There are 13 observations per label. Their measurement dates and the meaning of row order are unknown. The country medians discard within-label variation.
- The original data source and applicable license remain unverified.
- Some indicators have substantial missingness. Feature-median imputation shrinks incomplete profiles toward the center, and correlated indicators and absolute totals can influence distances.
- The partitions are exploratory. Their stability under other model settings and their relationship to external socioeconomic groupings remain unverified.

See the [dataset inventory](docs/DATASET.md), [methodology](docs/METHODOLOGY.md), and [historical notebook review](docs/ORIGINAL_NOTEBOOK_ISSUES.md).
