# Methodology and corrections

## Scope of the canonical analysis

The canonical pipeline is `analysis/run_clustering.py`. It reads the checked-in workbook through `argparse` and `pathlib.Path`; it contains no absolute user paths and makes no network calls. The legacy notebook is unchanged and archived in `legacy/`.

The workbook contains 2704 rows but only 208 countries, with exactly 13 rows per country and no year column. Country identifiers are retained for **labels only**: they are never clustering inputs. Because there is no explicit country-year key, duplicate country rows are retained and the observations are **not treated as independent**. See [DATA.md](DATA.md).

### Preprocessing

1. Confirm the country column by exact header name (`Country`; aliases checked by exact case-insensitive name only). If no identifier can be identified, country labels are marked `BLOCKED` rather than guessed.
2. Select pandas numeric-dtype columns only. Currency- and percent-formatted strings are reported as excluded, not silently parsed.
3. Drop all-empty columns and zero-variance columns. `Number of Records` is confirmed constant at 1.0 and is dropped.
4. Impute missing numeric values with `SimpleImputer(strategy='median')`.
5. Scale all retained indicators with `StandardScaler`.

This yields an 18-feature, 2704-row standardized matrix. All reported model comparisons use this **same representation**. PCA and t-SNE are not part of the canonical run, so scores from differently transformed spaces are never ranked as if interchangeable.

## Hypothesis-test limitations (not rerun as validated findings)

The legacy notebook contains hypothesis tests that are not reproduced in the canonical results. The issues below are documented rather than silently "fixed" with a new impressive-looking number.

### One-sample test on health spending

`Health Exp % GDP` is stored as a fraction (observed 0.008–0.225, median 0.061). The notebook's `pop_mean = 0.06` is therefore numerically consistent with a 6 percent hypothesis **only in fraction units**; the scale check is documented in [DATA.md](DATA.md). A two-sided `ttest_1samp(x, 0.06)` tests only "mean differs from 0.06". A directional claim requires choosing `alternative='greater'` or `alternative='less'` *a priori* and reporting it. In addition, the 13 repeated rows per country violate independence, so no validated one-sample result is claimed here.

### Urban/rural comparison

The notebook computed `rural_population = 1 - urban_population` and then ran `ttest_ind(urban_population, rural_population)`. These are deterministic complements of the **same** observation, not two independent samples, so an independent-samples t-test is invalid. In this workbook `Population Urban` is a 0–1 fraction, so subtracting from 1 is scale-consistent here; if a dataset stored the column as 0–100 percentages, subtracting from 1 would also be a scale error. This test is excluded from the new script for the dependence reason.

### Chi-square tests

The legacy cells used `pd.cut(..., bins=2)` followed by `chi2_contingency`. Bin boundaries derived from the sample make the contingency table data-dependent. Any retained chi-square test must report the exact bin boundaries, the contingency cells, the minimum expected count, and the p-value, and must be described as an association test — not evidence of causation. None are retained in the canonical results.

### Other legacy metrics

- Several legacy cells compute a "cluster accuracy" by comparing predictions with the clustering's own labels. That is self-agreement, not external classification accuracy. It is not reproduced, and no `accuracy` column exists in the new outputs.
- A DBSCAN cell computed centers by iterating over every sample label, repeating centers and potentially including noise. The canonical implementation computes no DBSCAN centers.
- The legacy notebook states that "DBSCAN is not most effective on this dataset". The canonical results do not override that with a more impressive-sounding claim; DBSCAN's high silhouettes occur only when most rows are labeled noise.
- Legacy `TSNE()` was called without a fixed seed. Any new visualization would set `random_state=42` and be labeled exploratory; t-SNE geometry is not used as cluster-validity evidence.

## Models and comparison protocol

- `KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)` for `k = 2, 3, 4, 5, 6`, with an explicit `n_init` for stable behavior.
- `AgglomerativeClustering(n_clusters=k)` (scikit-learn defaults: Ward linkage, Euclidean metric) for the same `k`.
- `SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=min(10, n_samples - 1), assign_labels='kmeans', random_state=42)`. The neighbor count is explicitly less than the sample count. On this workbook scikit-learn warns that the nearest-neighbor graph is not fully connected; those spectral results are therefore exploratory and unstable.
- `DBSCAN(min_samples=5)` on the same scaled matrix over a fixed a priori eps grid `(0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)`. The grid is not tuned to maximize a reported score. Noise is tracked explicitly (`-1` label, excluded from cluster-size counts).
- Silhouette is computed only when `2 <= number of non-noise clusters < number of non-noise samples`. For DBSCAN it is computed on non-noise rows only, so it is **not directly comparable** to the partition methods, which score all rows. A high silhouette from a configuration that labels most rows as noise is not evidence of a good clustering.

## Leakage, overfitting, and interpretation limits

- Imputation parameters and scaling are fit on all rows because this is a descriptive, unsupervised analysis. There is no train/test split and no generalization claim.
- Repeated rows per country mean observations are not independent; any supervised extension would risk leakage if rows from the same country were split across train and test sets.
- Silhouette favors compact, convex groups and can be misleading for density-based or highly imbalanced clusters. Cluster sizes are reported alongside it.
- Cluster labels are arbitrary; they are not country rankings, risk scores, or causal groupings. Small clusters (for example 3–13 rows) should not be over-interpreted.
- No external ground truth is available, so no predictive accuracy or external-validity claim is made.
- The workbook's source and license remain unknown; see [DATA.md](DATA.md).

## Reproduction

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python analysis/run_clustering.py --input World_development_measurement.xlsx --output-dir outputs
```

Outputs: `outputs/model_comparison.csv`, `outputs/preprocessing_summary.json`, `outputs/country_clusters.csv`. Environment and observed results are recorded in [RESULTS.md](RESULTS.md).
