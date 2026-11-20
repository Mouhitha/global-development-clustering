# Results and limitations

## Reproduction status: PASSED

Full run completed on 2026-11-21 from the repository root with the checked-in workbook.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
python analysis/run_clustering.py --input World_development_measurement.xlsx --output-dir outputs
python -m json.tool legacy/original_clustering_analysis.ipynb >/dev/null
```

- Tests: **6 passed, 0 failed**.
- Python 3.11.5; numpy 2.4.6, pandas 3.1.1, openpyxl 3.1.5, scipy 1.17.1, scikit-learn 1.9.1, pytest 9.2.0.
- Input: `World_development_measurement.xlsx`; raw shape 2704 × 25; 208 countries; 13 rows per country.
- Preprocessing: 18 numeric features retained; `Number of Records` dropped for zero variance; `Country` and five string-formatted indicators excluded. Median imputation and standardization applied.
- Model rows written: 24; silhouette computed for all 24 under the stated validity rule.
- Result files: `outputs/model_comparison.csv`, `outputs/preprocessing_summary.json`, `outputs/country_clusters.csv` (generated locally; `outputs/` is git-ignored). Values below are rounded to four decimals from those files.

## Computed comparison

Silhouette is computed on all rows for K-Means, agglomerative, and spectral clustering, and on non-noise rows only for DBSCAN. The columns are therefore not interchangeable across model families.

| Method | k | Noise | Noise fraction | Silhouette | Smallest / largest non-noise cluster |
| --- | ---: | ---: | ---: | ---: | ---: |
| KMeans_k2 | 2 | 0 | 0.0000 | 0.3313 | 1028 / 1676 |
| Agglomerative_k2 | 2 | 0 | 0.0000 | 0.3134 | 663 / 2041 |
| Spectral_k2 | 2 | 0 | 0.0000 | 0.1831 | 13 / 2691 |
| KMeans_k3 | 3 | 0 | 0.0000 | 0.3483 | 38 / 1653 |
| Agglomerative_k3 | 3 | 0 | 0.0000 | 0.3359 | 39 / 2002 |
| Spectral_k3 | 3 | 0 | 0.0000 | 0.2079 | 13 / 2654 |
| KMeans_k4 | 4 | 0 | 0.0000 | 0.2530 | 38 / 1094 |
| Agglomerative_k4 | 4 | 0 | 0.0000 | 0.2093 | 39 / 1169 |
| Spectral_k4 | 4 | 0 | 0.0000 | 0.3055 | 13 / 1604 |
| KMeans_k5 | 5 | 0 | 0.0000 | 0.2557 | 3 / 1104 |
| Agglomerative_k5 | 5 | 0 | 0.0000 | 0.1997 | 22 / 1169 |
| Spectral_k5 | 5 | 0 | 0.0000 | 0.3134 | 11 / 1600 |
| KMeans_k6 | 6 | 0 | 0.0000 | 0.1915 | 8 / 839 |
| Agglomerative_k6 | 6 | 0 | 0.0000 | 0.2014 | 3 / 1169 |
| Spectral_k6 | 6 | 0 | 0.0000 | 0.2283 | 11 / 1026 |
| DBSCAN_eps0.25 | 10 | 2610 | 0.9652 | 0.8373 | 5 / 21 |
| DBSCAN_eps0.5 | 59 | 2264 | 0.8373 | 0.6415 | 5 / 38 |
| DBSCAN_eps1 | 80 | 681 | 0.2518 | -0.2167 | 4 / 1353 |
| DBSCAN_eps1.5 | 26 | 224 | 0.0828 | -0.2309 | 5 / 2265 |
| DBSCAN_eps2 | 15 | 104 | 0.0385 | 0.0219 | 6 / 2463 |
| DBSCAN_eps2.5 | 7 | 51 | 0.0189 | 0.1758 | 7 / 2593 |
| DBSCAN_eps3 | 8 | 22 | 0.0081 | 0.2506 | 5 / 2624 |
| DBSCAN_eps4 | 5 | 13 | 0.0048 | 0.6275 | 7 / 2642 |
| DBSCAN_eps5 | 5 | 7 | 0.0026 | 0.6119 | 8 / 2643 |

## Observed caveats

- The DBSCAN configurations with the highest silhouettes (`eps=0.25`, `eps=0.5`) label 96.52% and 83.73% of rows as noise and contain only tiny non-noise clusters. Their silhouettes are computed on those small subsets and are **not** evidence of a strong clustering.
- Larger eps values produce one very large cluster (for example 2642 of 2704 rows at `eps=4`) with a handful of small clusters. This does not support an "impressive" DBSCAN conclusion; it is consistent with the legacy notebook's caution that DBSCAN is not most effective here.
- Spectral clustering emitted the scikit-learn warning `Graph is not fully connected, spectral embedding may not work as expected.` for five fits. Spectral cluster sizes are highly imbalanced (for example 13 / 2691 at k=2); treat them as exploratory.
- K-Means at k=3 has the highest silhouette among the partition methods on this representation (0.3483), but one cluster has only 38 rows. This is a descriptive geometry measure, not an external validation.
- No accuracy column was computed. Legacy self-agreement "accuracy" values are not reproduced.
- Repeated country rows violate independence; see [METHODOLOGY.md](METHODOLOGY.md).

## Result provenance

All numbers above come from the successful run recorded at the start of this file. No figure was carried over from the legacy notebook. The archived notebook remains unchanged and is not labeled reproducible.

## Explicitly unresolved research questions

1. **Which indicators move together?** The canonical script performs clustering only. Legacy hypothesis-test and chi-square results were excluded for scale, dependence, and reporting problems; no replacement association analysis has been validated.
2. **What do the 13 rows per country represent?** The workbook has no year/date column, so the time structure is unknown and duplicate country rows cannot be safely collapsed.
3. **How sensitive are groupings to dimensionality reduction?** The canonical run deliberately uses one representation. PCA/t-SNE comparisons are not validated here, and legacy t-SNE results used an unseeded transform.
4. **Do string-formatted indicators change the groupings?** `GDP`, `Business Tax Rate`, `Health Exp/Capita`, and tourism columns were excluded pending a documented parsing rule.
5. **Are spectral results stable?** The nearest-neighbor graph is not fully connected; neighbor-count sensitivity has not been investigated.
6. **External validation:** no ground-truth labels are available; cluster quality cannot be established beyond internal geometry.
7. **Data source and license:** not established from the workbook; see [DATA.md](DATA.md).
