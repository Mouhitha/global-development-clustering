# Data provenance and scope

## Audit method

Read-only audit of the checked-in binary `World_development_measurement.xlsx` on 2026-11-21 using pandas/openpyxl. No rows were modified, no dataset was exported, and no row-level records are reproduced here. The binary workbook's contents and license were not independently established beyond this audit.

## Source and license status

- **Original source URL/attribution:** not established. The workbook has one visible sheet, no cell hyperlinks, no defined names, and no custom properties containing a source URL or citation. The notebook links to an IEEE paper, but the workbook itself does not state that it is the paper's source data.
- **Workbook metadata:** creator `openpyxl`; `lastModifiedBy` is `Mayur Ramtekkar`. This is file metadata, not a verified source attribution or license grant.
- **License:** `unknown`. Do not assume a permissive license from the file's presence in the repository.

## Confirmed structure

- Sheets: one, `world_development` (visible).
- Rows × columns: **2704 × 25**.
- Country identifier: exact header `Country`; **208** distinct non-null values.
- Each of the 208 countries appears exactly **13** times. There is **no year/date column**, so these are repeated observations whose time interpretation is not established. There are no fully duplicated rows; 2496 rows repeat a country identifier.
- Numeric-dtype columns: **19** (one of them constant). Non-numeric columns: 6.

## Columns

`Birth Rate`, `Business Tax Rate`, `CO2 Emissions`, `Country`, `Days to Start Business`, `Ease of Business`, `Energy Usage`, `GDP`, `Health Exp % GDP`, `Health Exp/Capita`, `Hours to do Tax`, `Infant Mortality Rate`, `Internet Usage`, `Lending Interest`, `Life Expectancy Female`, `Life Expectancy Male`, `Mobile Phone Usage`, `Number of Records`, `Population 0-14`, `Population 15-64`, `Population 65+`, `Population Total`, `Population Urban`, `Tourism Inbound`, `Tourism Outbound`.

## Missingness (null counts per column)

| Column | Nulls |
| --- | ---: |
| Ease of Business | 2519 |
| Hours to do Tax | 1288 |
| Business Tax Rate | 1281 |
| Days to Start Business | 986 |
| Energy Usage | 919 |
| Lending Interest | 824 |
| CO2 Emissions | 579 |
| Tourism Outbound | 471 |
| Tourism Inbound | 368 |
| Health Exp % GDP | 309 |
| Health Exp/Capita | 309 |
| Infant Mortality Rate | 260 |
| Population 0-14 | 220 |
| Population 15-64 | 220 |
| Population 65+ | 220 |
| GDP | 210 |
| Internet Usage | 173 |
| Mobile Phone Usage | 167 |
| Life Expectancy Female | 136 |
| Life Expectancy Male | 136 |
| Birth Rate | 119 |
| Population Urban | 26 |
| Number of Records | 0 |
| Population Total | 0 |
| Country | 0 |

## Scale and typing notes

- `Number of Records` has a single observed value, `1.0` (zero variance), and is dropped by the canonical preprocessing.
- `Health Exp % GDP` is stored as a **fraction**: observed range 0.008–0.225, median 0.061. A value of 0.06 therefore corresponds to 6 percent. See [METHODOLOGY.md](METHODOLOGY.md).
- `Population Urban` is stored as a **fraction** in this workbook: observed range 0.082–1.0, median 0.56.
- Five indicators are stored as formatted strings, not numbers: `Business Tax Rate` (for example `76.9%`), `GDP` (for example `$9,129,594,819`), `Health Exp/Capita` (for example `$60`), `Tourism Inbound` and `Tourism Outbound` (for example `$102,000,000`). They are excluded from numeric clustering until a documented conversion is added. Only formatting examples are quoted; no row-level records are reproduced.
- No additional dataset was added to the repository.
