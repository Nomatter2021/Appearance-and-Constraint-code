# Appearance and Constraint: Source Code

Source code for:

> **Appearance and Constraint: Behavioural Homomorphism, Environmental Stratification, and the Flow of Uninitiated Obligation in Speculative Markets**
> Nguyen, H. P. (2026)

Nine empirical tests on RSSI, Phi-gated amplification, and reflexive loop validation across 354 firms and 3,239 speculative company-quarter observations (Technology, Healthcare, Services, 2017--2026).

Companion paper: [What Price Presupposes](https://doi.org/10.5281/zenodo.19480971) (Nguyen 2026a).

---

## Repository Structure

```
Appearance-and-Constraint-code/
├── Main_pipeline/          # Data collection and variable construction (Steps 00-13)
├── Staticits/              # Statistical tests T1-T9
└── README.md
```

---

## Main Pipeline

Run scripts in numerical order. Each script reads from and writes to `data/`.

```
data/
├── classified/         # Firm-quarter classified data (output of Step 11)
│   ├── Healthcare/
│   ├── Technology/
│   └── Services/
└── processed/          # Intermediate and final processed data
```

### Step-by-Step

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `00_Create_SEC_Ticker_config.py` | Build SEC ticker list from Excel to YAML config | Excel ticker file | `config/sec_tickers.yaml` |
| `01_Create_benchmark_ticker_config.py` | Build benchmark ticker config | Benchmark list | `config/benchmark_tickers.yaml` |
| `02_Create_survey_config.py` | Build full survey config for crawling | Ticker configs | `config/survey_config.yaml` |
| `03_Crawl_benchmark.py` | Crawl sector benchmark margin data from SEC EDGAR | Survey config | `data/raw/benchmark/` |
| `04_Crawl_all_sample.py` | Crawl all firm financial data from SEC EDGAR via XBRL API | Survey config | `data/raw/firms/` |
| `05_Benchmark_calculate.py` | Compute rolling 12-quarter sector median benchmark margins | Benchmark data | `data/processed/benchmark_margins.csv` |
| `07_Crawl_margin.py` | Crawl and clean operating margin data | Firm data | `data/processed/margins/` |
| `08_Clean_market_cap.py` | Compute market capitalisation from shares outstanding and Yahoo Finance prices | Firm data | `data/processed/market_cap/` |
| `09_KBrand_calculate.py` | Compute brand-anchored claim K_Brand using sector multipliers | Firm data | `data/processed/kbrand/` |
| `10_Framework_calculate.py` | Compute full price decomposition: V_Prod_base, s_baseline, S_surplus, K_Brand, K_Pi', E3, A, B, R_t, PDI | All processed data | `data/processed/framework/` |
| `11_Classify_configurations.py` | Classify each firm-quarter into C1--C6 or Normal based on R_t and delta K_Pi' | Framework data | `data/classified/{Sector}/` |
| `12_RSSI_calculate.py` | Compute sector-level RSSI using 8-quarter historical rolling baseline (shift=1, min_periods=4) | Classified data | `data/processed/{Sector}_RSSI_historical.csv` |
| `13_adding_variables.py` | Merge RSSI, dRSSI/dt, Phi, MCF, MRF into classified files | Classified + RSSI | `data/classified/{Sector}/` (updated) |

> **Note on Step 06:** Step 06 is not included in this release. Steps 07 onwards do not depend on it.

---

## Statistical Tests

Located in `Staticits/`. Each script is self-contained, reads from `data/classified/`, and writes outputs to `results/tables/`, `results/figures/`, and `results/reports/`.

| Script | Test | Description |
|--------|------|-------------|
| `00_Plot_RSSI_Timeseries.py` | -- | Plot sector-level RSSI time series with peak detection |
| `01_Univariate_signal.py` | T1 | Univariate signal analysis: D_t, B, RSSI concurrent and lagged associations, peak-to-collapse lag distribution, two-cycle structure |
| `02_Instability_analysis.py` | T2 | Instability region analysis: collapse rates by B x RSSI joint matrix, Markov transitions by RSSI level, 4Q and 6Q window rates |
| `03_Conditional_transition.py` | T3 | Conditional transition analysis: C3/C4 direction, Normal-to-C2 entry, self-loop persistence, full transition matrix comparison, 8-quarter escape rates |
| `04_Joint_collpase_condition.py` | T4 | Joint collapse condition: 2x3 and 5x3 collapse matrices, logistic regression with RSSI dummies, Kaplan-Meier and Cox PH survival analysis, absence pattern test |
| `05_Cluster_separation.py` | T5 | Cluster separation: silhouette and Davies-Bouldin in 2D and 3D spaces, permutation test (1,000 perms, seed=42), feature importance (Random Forest) |
| `06_RSSI_Parapola.py` | T6 | RSSI parabola validation: aligned trajectory, shape test (linear vs quadratic), peak-to-collapse lag distribution, two-cycle detection, FC2 post-collapse discriminator |
| `07_Incremental_AUC.py` | T7 | Phase space risk stratification: conditional collapse rates by B quartile and RSSI phase in C2, CMH test controlling for B |
| `08_Reflexive_loop_validation.py` | T8 | Reflexive loop validation: MCF/MRF predictive performance, MRF leading correlation, MCF autocorrelation, Granger causality under Phi=1 |
| `09_Phi_Gated_Auxiliary.py` | T9 | Phi-gated and auxiliary tests: CMH by Phi state (T9), Mann-Whitney by configuration (T10), dRSSI/dt trajectories (T11), RSSI at Phi drop (T12), post-collapse restart (T13), placebo test (T14, 1,000 perms, seed=42) |

Reports are saved to `results/reports/T{N}_*.txt`. Figures are saved to `results/figures/`.

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
statsmodels
lifelines
requests
pyyaml
```

Install:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels lifelines requests pyyaml
```

---

## Data Sources

- **Financial data:** SEC EDGAR via XBRL company facts API (`https://data.sec.gov/api/xbrl/companyfacts/`)
- **Market prices:** Yahoo Finance (quarterly closing prices)
- **Sectors covered:** Technology, Healthcare and Pharmaceuticals, Services
- **Observation window:** 2017 Q1 -- 2026 Q1
- **Sample:** 354 companies, 3,239 speculative company-quarter observations after applying four data quality filters (minimum 6 consecutive quarters, non-zero variance, non-constant shares outstanding, available XBRL data)

---

## Key Variables

| Variable | Description |
|----------|-------------|
| `K_Pi_prime` | Uninitiated obligation stock: market cap minus verified productive base, surplus, and brand components |
| `E3` | Obligation ratio: K_Pi' / V_Prod_base |
| `B` | Surplus extraction component: E3 - (1 + PGR_t) |
| `R_t` | Absorption ratio: s_t / K_Pi'_{t-1} |
| `PDI_t` | Productive Discharge Index: s_t / (|delta K_Pi'_t| + s_t) |
| `Phi_t` | Flow indicator: 1 if delta K_Pi' > 0, else 0 |
| `RSSI` | Relative Sector Skew Index: 8-quarter rolling z-score of sector mean K_Pi' |
| `MCF_add` | Market Cognitive Function (additive): D_t + RSSI_t |
| `MRF` | Market Reflexive Function: lambda x MCF x V_Price_{t-1} x Psi_t |
| `Configuration` | C1--C6 or Normal: lifecycle stage classification based on R_t and delta K_Pi' |

---

## Reproducing Results

```bash
# 1. Build configs
python Main_pipeline/00_Create_SEC_Ticker_config.py
python Main_pipeline/01_Create_benchmark_ticker_config.py
python Main_pipeline/02_Create_survey_config.py

# 2. Crawl data
python Main_pipeline/03_Crawl_benchmark.py
python Main_pipeline/04_Crawl_all_sample.py

# 3. Compute variables
python Main_pipeline/05_Benchmark_calculate.py
python Main_pipeline/07_Crawl_margin.py
python Main_pipeline/08_Clean_market_cap.py
python Main_pipeline/09_KBrand_calculate.py
python Main_pipeline/10_Framework_calculate.py
python Main_pipeline/11_Classify_configurations.py
python Main_pipeline/12_RSSI_calculate.py
python Main_pipeline/13_adding_variables.py

# 4. Run statistical tests
cd Staticits/
python 00_Plot_RSSI_Timeseries.py
python 01_Univariate_signal.py
python 02_Instability_analysis.py
python 03_Conditional_transition.py
python 04_Joint_collpase_condition.py
python 05_Cluster_separation.py
python 06_RSSI_Parapola.py
python 07_Incremental_AUC.py
python 08_Reflexive_loop_validation.py
python 09_Phi_Gated_Auxiliary.py
```

Pre-computed results are available in `Staticits/results.zip`.

---

## Citation

Nguyen Hong, P. (2026). Appearance and Constraint: Behavioural Homomorphism, Environmental Stratification, and the Flow of Uninitiated Obligation in Speculative Markets. Zenodo. https://doi.org/10.5281/zenodo.19589022

```bibtex
@misc{nguyenhong2026appearance,
  author    = {Nguy{\~{\^e}}n H{\`{\^o}}ng, P.},
  title     = {Appearance and Constraint: Behavioural Homomorphism, Environmental
               Stratification, and the Flow of Uninitiated Obligation in
               Speculative Markets},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19589022},
  url       = {https://doi.org/10.5281/zenodo.19589022}
}
```

---

## License

Apache 2.0. See `LICENSE`.
