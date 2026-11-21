# Dataset

`World_development_measurement.xlsx` contains the `world_development` sheet with **2,704 nonempty data rows and 25 columns**, excluding the header. The `Country` column contains 208 distinct labels, each appearing 13 times. These are repeated observations within each label. The workbook has no year field, and the measurement dates and meaning of row order are unknown.

## Original columns

1. `Birth Rate`
2. `Business Tax Rate`
3. `CO2 Emissions`
4. `Country`
5. `Days to Start Business`
6. `Ease of Business`
7. `Energy Usage`
8. `GDP`
9. `Health Exp % GDP`
10. `Health Exp/Capita`
11. `Hours to do Tax`
12. `Infant Mortality Rate`
13. `Internet Usage`
14. `Lending Interest`
15. `Life Expectancy Female`
16. `Life Expectancy Male`
17. `Mobile Phone Usage`
18. `Number of Records`
19. `Population 0-14`
20. `Population 15-64`
21. `Population 65+`
22. `Population Total`
23. `Population Urban`
24. `Tourism Inbound`
25. `Tourism Outbound`

## Missing observations

| Column | Blank source cells | Missing after numeric conversion |
| --- | ---: | ---: |
| `GDP` | 210 | 210 |
| `Business Tax Rate` | 1,281 | 1,281 |
| `Health Exp % GDP` | 309 | 309 |
| `Population Urban` | 26 | 26 |

`GDP` mixes dollar-formatted strings with a stored numeric value. The nonblank `Business Tax Rate` entries are percent strings. The stored numeric values in `Health Exp % GDP` and `Population Urban` use fractional units. The reanalysis converts explicit percent strings to fractions and retains bare numeric values as stored.

## Provenance

The workbook is the input supplied with the repository. Its original publisher, source URL, measurement years, variable definitions, and data license have not been verified. The repository owner should supply the original source URL and the applicable data license, along with any available data dictionary and measurement years.
