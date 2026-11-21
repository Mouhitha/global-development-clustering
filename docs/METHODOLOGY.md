# Methodology

## Observation units

- **Historical row-level exploration.** The original notebook works with 2,704 observations, including 13 rows for each `Country` label. Row-level summaries describe the workbook, and repeated observations within a country can be dependent.
- **Country-level reanalysis.** The new pipeline summarizes each indicator using the median of its available observations within a country label. This gives 208 profiles and gives each label one row in the clustering. Medians collapse within-country variation and may summarize different subsets of records for different indicators. With measurement years unavailable, the profiles support no time trend or chronological comparison.

## Preprocessing and clustering

1. Read `world_development` and require `Country` and the 23 source indicator columns. Trim surrounding whitespace from country labels and use them as group keys and output labels. A fixed feature list excludes `Country`, any `Country_encoded` column or other surrogate, and `Number of Records`.
2. Parse dollar signs and commas in numeric strings using standard thousands grouping. An explicit percent string such as `"76.9%"` becomes `0.769`, and a bare number such as `0.06` keeps its stored value. The parser applies no currency conversion, percentage bounds, or guessed corrections. Invalid values become missing. The summary counts blanks and nonblank invalid values separately.
3. Compute country medians before filling any missing data. Exclude indicators that are entirely missing or constant among observed country medians. For each remaining indicator, fill missing country values with the median across the available country medians. Report how many cells were filled. This retains country labels while reducing the variation captured by incomplete profiles and leaving uncertainty about the missing values.
4. Apply `StandardScaler`. When two or more varying indicators remain, use full-SVD PCA to retain at least 90% of the variance in the standardized matrix. Report the retained variance by component and in total. This threshold limits variance lost in the projection. It does not establish which directions are substantively important, and correlated indicators can still exert combined influence.
5. Fit K-means with seed 42 and 20 initializations and agglomerative clustering with Ward linkage. Both use the same PCA scores and a fixed three-cluster setting for comparison. Three is an analysis setting, and cluster IDs are arbitrary within each method. Compare Euclidean silhouette and cluster sizes in that shared space. Silhouette measures within-sample cohesion and separation.

The reanalysis retains all 208 country profiles without an outlier filter. The historical IQR filter and t-SNE experiments are described in the [notebook review](ORIGINAL_NOTEBOOK_ISSUES.md). Results describe the included labels under the stated feature, imputation, scaling, and clustering choices. Changes to those choices, large differences in absolute totals, or different data coverage can alter the partitions.

## Statistical inference

The reanalysis omits inferential hypothesis tests. The workbook supplies neither a sampling design nor measurement dates, and country-level aggregation alone cannot establish independent sampling across country labels. For the urban/rural question, the two values would be deterministic complements if the definition is exhaustive. The country-level paired difference would then be `2 × median(Population Urban) − 1`, and a paired t-test would test whether the mean country-level urban share equals one-half. That test would require defensible sampling and independence assumptions across country-level differences. The original notebook review describes the assumptions for the other tests.
