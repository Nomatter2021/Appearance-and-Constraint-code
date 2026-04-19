# Appearance and Constraint: Source Code

Source code for:

> **Appearance and Constraint: Behavioural Homomorphism, Environmental Stratification, and the Flow of Uninitiated Obligation in Speculative Markets**
> Nguyen, H. P. (2026b). Zenodo. https://doi.org/10.5281/zenodo.19589022

Thirteen empirical tests on RSSI, Phi-gated amplification, boundary condition validation, and reflexive loop causal chain across 354 firms and 3,239 speculative company-quarter observations (Technology, Healthcare, Services, 2017–2026).

Companion paper: [What Price Presupposes](https://doi.org/10.5281/zenodo.19480971) (Nguyen 2026a) — establishes K_Pi' as uninitiated labour obligation and the six structural configurations.

---

## Repository Structure

```
Appearance-and-Constraint-code/
├── Main_pipeline/          # Data collection and variable construction (Steps 00–13)
├── Staticits/              # Statistical tests T1–T13
│   └── results.zip         # Pre-computed outputs (tables, figures, reports)
└── README.md
```

> **Note on folder name:** The statistics folder is named `Staticits/` (not `Statistics/`). This is the canonical name used throughout the repository.

> **Note on file naming:** Scripts 00–09 use the format `NN_name.py`. Scripts added later (T10–T13) use the format `TNN_name.py`. Both conventions point to the same test numbering system described in the paper.

---

## Main Pipeline

Run scripts in numerical order. Each script reads from and writes to `data/`.

```
data/
├── classified/             # Firm-quarter classified data (output of Step 11)
│   ├── Healthcare/
│   ├── Technology/
│   └── Services/
└── processed/              # Intermediate and final processed data
    ├── benchmark_margins.csv
    ├── margins/
    ├── market_cap/
    ├── kbrand/
    ├── framework/
    └── {Sector}_RSSI_historical.csv
```

### Step-by-Step

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `00_Create_SEC_Ticker_config.py` | Build SEC ticker list from Excel to YAML config | Excel ticker file | `config/sec_tickers.yaml` |
| `01_Create_benchmark_ticker_config.py` | Build benchmark ticker config | Benchmark list | `config/benchmark_tickers.yaml` |
| `02_Create_survey_config.py` | Build full survey config for data collection | Ticker configs | `config/survey_config.yaml` |
| `03_Crawl_benchmark.py` | Crawl sector benchmark margin data from SEC EDGAR | Survey config | `data/raw/benchmark/` |
| `04_Crawl_all_sample.py` | Crawl all firm financial data from SEC EDGAR via XBRL API | Survey config | `data/raw/firms/` |
| `05_Benchmark_calculate.py` | Compute rolling 12-quarter sector-median benchmark margins (s_baseline) | Benchmark data | `data/processed/benchmark_margins.csv` |
| `07_Crawl_margin.py` | Crawl and clean operating margin data | Firm data | `data/processed/margins/` |
| `08_Clean_market_cap.py` | Compute market capitalisation from shares outstanding × quarterly closing prices (Yahoo Finance) | Firm data | `data/processed/market_cap/` |
| `09_KBrand_calculate.py` | Compute brand-anchored claim K_Brand using sector multipliers and Brand_Score composite | Firm data | `data/processed/kbrand/` |
| `10_Framework_calculate.py` | Compute full price decomposition: V_Prod_base, s_baseline, S_surplus, K_Brand, K_Pi', E3, A, B, R_t, PDI | All processed data | `data/processed/framework/` |
| `11_Classify_configurations.py` | Classify each firm-quarter into C1–C6 or Normal based on R_t and delta K_Pi' | Framework data | `data/classified/{Sector}/` |
| `12_RSSI_calculate.py` | Compute sector-level RSSI: 8-quarter rolling z-score of sector-mean K_Pi' (shift=1, min_periods=4, no look-ahead bias) | Classified data | `data/processed/{Sector}_RSSI_historical.csv` |
| `13_adding_variables.py` | Merge RSSI, dRSSI/dt, Phi, MCF (additive and multiplicative), MRF into classified files | Classified + RSSI | `data/classified/{Sector}/` (updated in place) |

> **Note on Step 06:** Step 06 is not included in this release. Steps 07 onwards do not depend on it.

---

## Statistical Tests

Located in `Staticits/`. Each script is self-contained: reads from `data/classified/` and writes outputs to `results/tables/`, `results/figures/`, and `results/reports/`.

### Full Test Table

| Script | Test | Purpose | Key output |
|--------|------|---------|------------|
| `00_Plot_RSSI_Timeseries.py` | — | Plot sector-level RSSI time series with peak detection (Technology, Services, Healthcare) | Sector cycle figures |
| `01_Univariate_signal.py` | T1 | Establish baseline predictive content of D_t, B, and RSSI operating independently. Lag structure analysis (lags 1–12) confirms RSSI has no univariate predictive power at any horizon. | `T1_Summary_Report.txt` |
| `02_Instability_analysis.py` | T2 | Test whether RSSI creates non-monotonic risk stratification (Mid > Extreme). Joint B×RSSI collapse rate matrix. Quadratic regression for inverted-U shape test. | `T2_Instability_Report.txt` |
| `03_Conditional_transition.py` | T3 | Full Markov transition matrices by RSSI level. Confirms Mid RSSI as bifurcation zone (simultaneously highest collapse and highest evolve rates in C2). | `T3_Transition_Analysis_Report.txt` |
| `04_Joint_collpase_condition.py` | T4 | Joint B × RSSI collapse condition. Cox proportional hazard model. Kaplan-Meier survival curves. Confirms independent RSSI stratification effect after controlling for B. | `T4_Joint_Collapse_Report.txt` |
| `05_Cluster_separation.py` | T5 | Cluster separation in 2D and 3D spaces. Permutation test (1,000 resamples, seed=42). Random Forest feature importance. Confirms C2 trajectory separation is not achievable from firm-level data — theoretically predicted. | `T5_Cluster_Separation_Report.txt` |
| `06_RSSI_Parapola.py` | T6 | Sector trajectory validation (cycle structure, peak detection). Peak-to-collapse lag distribution. FC2 post-collapse discriminator (Terminus vs. Cycle Bottom). Direction analysis: ascending vs. descending Mid RSSI (Block 6G). Mirror test: RSSI vs. B trajectory around collapse (Block 6E). | `T6_RSSI_Parabola_Report.txt` |
| `07_Incremental_AUC.py` | T7 | Phase space risk in C2 gestation. Conditional collapse rates by B quartile and RSSI phase. CMH test controlling for B quartile (p = 0.0232). | `T7_Phase_Space_Risk_Report.txt` |
| `08_Reflexive_loop_validation.py` | T8 | Validate reflexive loop components: MCF predictive performance across all variants, MRF leading vs. concurrent price correlation, MCF autocorrelation under Phi=1 vs. Phi=0, Granger causality. | `T8_Reflexive_Loop_Report.txt` |
| `09_Phi_Gated_Auxiliary.py` | T9 | Phi-gated CMH test: RSSI Mid effect under Phi=1 vs. Phi=0. Placebo validation (cross-firm shuffle and global shuffle). Post-collapse restart analysis. | `Test9_Comprehensive_Report.txt` |
| `T10_RSSI_Validity.py` | T10 | RSSI validity and independence tests. Mann-Whitney by configuration (C2, C3, C4). Stratified matching (B quartiles 2–3, Phi=1). Time-split validation (train 2017–2021, test 2022–2026). | `T10_RSSI_Validity_Report.txt` |
| `T11_Structural_Bifurcation.py` | T11 | Rate-of-change analysis: dRSSI/dt across C2 trajectories (Collapse, Sustain, Evolve). Entropy test. Direction-aware C2 transition matrices (Mid_asc vs. Mid_desc). | `T11_Structural_Bifurcation_Report.txt` |
| `T12_Placebo_Validation_of_RSSI_Dual_Properties.py` | T12 | Boundary condition validation of RSSI dual properties. Four permutation sub-tests isolating each boundary condition: (1) moderator without B threshold, (2) direction at firm level, (3) full joint condition (all three boundary conditions satisfied), (4) cycle autocorrelation. | `T12_Placebo_Validation_of_RSSI_dual_properties.txt` |
| `T13_Causality_Tests_For_Reflexifity_Theory.py` | T13 | Causal chain validation of the three-phase reflexive loop. Sub-test 1: MRF Granger-causes MCF at firm level (initiation). Sub-test 2: MCF_sector ↔ RSSI bidirectional Granger at sector level (self-reinforcement). Sub-test 3: B predicts collapse within full boundary conditions via permutation test (termination). | `T13_causality_report.txt` |

> **Note on script name typos:** `04_Joint_collpase_condition.py` ('collpase'), `06_RSSI_Parapola.py` ('Parapola'), and `T13_Causality_Tests_For_Reflexifity_Theory.py` ('Reflexifity') contain spelling errors in the filenames. These are the canonical names on the repository and should be used as-is when running the pipeline.

---

## Key Variables

| Variable | Formula | Description |
|----------|---------|-------------|
| `K_Pi_prime` | V_Price − V_Prod_base − s_baseline − S_surplus − K_Brand | Uninitiated obligation stock: market cap residual after subtracting all verified and brand-anchored components |
| `E3` | K_Pi' / V_Prod_base | Obligation ratio: uninitiated obligations relative to verified productive base |
| `A` | 1 + PGR_t | Productive growth component; approximately 1 in all speculative configurations |
| `B` | E3 − (1 + PGR_t) | Surplus extraction component: obligation in excess of what productive growth could justify |
| `R_t` | s_t / K_Pi'_(t−1) | Absorption ratio: productive discharge capacity per period |
| `PDI_t` | s_t / (\|ΔK_Pi'_t\| + s_t) | Productive Discharge Index: proportion of K_Pi' change driven by surplus vs. price movement |
| `Phi_t` | 1 if ΔK_Pi' > 0, else 0 | Flow indicator: 1 when the reflexive loop is actively imposing new obligation on firms |
| `RSSI` | (SectorMean_KPi'_t − RollingMean_8q_(t−1)) / RollingStd_8q_(t−1) | Relative Sector Skew Index: sector obligation density relative to its own 8-quarter historical norm |
| `MCF_add` | D_t + RSSI_t | Market Cognitive Function (additive, operational): log price-production displacement plus RSSI |
| `MCF_mult` | D_t × RSSI_t × Phi_t | Market Cognitive Function (multiplicative, theoretical): degenerates under Phi=0 |
| `MRF` | λ × MCF_t × V_Price_(t−1) × Psi_t | Market Reflexive Function: material pressure on future price; Psi_t = dK_Pi'_pct × (1 − PDI_t) |
| `Configuration` | Based on R_t and ΔK_Pi' | C1–C6 or Normal: six lifecycle stages — see companion paper for full definitions |

### Configuration Reference

| Config | R_t | ΔK_Pi' | Phi | Lifecycle stage |
|--------|-----|--------|-----|----------------|
| C1 | = 0 | < 0 | 0 | Termination (public): price collapse |
| C2 | = 0 | > 0 | 1 | Gestation: pure obligation accumulation |
| C3 | > 0 | > 0 | 1 | Maturity: market overwhelms productive discharge |
| C4 | 0 < R_t < 1 | < 0 | 0 | Partial discharge: productive resistance |
| C5 | ≥ 1 | < 0 | n/a | Full discharge: **absent from all observed periods** |
| C6 | = 0 | < 0 | 0 | Termination (silent): no price signal |

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
lifelines
requests
pyyaml
```

Install:
```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn lifelines requests pyyaml
```

> **Note on `statsmodels`:** The original pipeline used `statsmodels` for logistic regression p-values in T2 Block 2E. The updated `02_Instability_analysis.py` computes Wald test p-values via Fisher Information Matrix using only `numpy` and `scipy` — no `statsmodels` required. If `statsmodels` is installed, the script will use it automatically; otherwise it falls back to the Wald test implementation. Results are equivalent.

> **Note on `lifelines`:** Required for Kaplan-Meier and Cox PH survival analysis in T4 (`04_Joint_collpase_condition.py`). If not available on your platform, T4 will skip the survival analysis blocks and report only the matrix results.

---

## Data Sources

- **Financial data:** SEC EDGAR via XBRL company facts API (`https://data.sec.gov/api/xbrl/companyfacts/`)
- **Market prices:** Yahoo Finance (quarterly closing prices, via `yfinance` or manual download)
- **Sectors covered:** Technology, Healthcare and Pharmaceuticals, Services
- **Excluded:** Financials and Real Estate (revenue structure violates M-C-M' proxy assumption)
- **Observation window:** 2017 Q1 – 2026 Q1
- **Sample construction:** 4,326 candidate companies → 524 passed continuity filters (≥6 consecutive quarters, non-zero variance in financial metrics, non-constant shares outstanding, available XBRL data) → 354 retained after sector exclusions

---

## Reproducing Results

### Full pipeline from scratch

```bash
# Step 1 — Build configuration files
python Main_pipeline/00_Create_SEC_Ticker_config.py
python Main_pipeline/01_Create_benchmark_ticker_config.py
python Main_pipeline/02_Create_survey_config.py

# Step 2 — Crawl data from SEC EDGAR and Yahoo Finance
python Main_pipeline/03_Crawl_benchmark.py
python Main_pipeline/04_Crawl_all_sample.py

# Step 3 — Compute variables
python Main_pipeline/05_Benchmark_calculate.py
python Main_pipeline/07_Crawl_margin.py
python Main_pipeline/08_Clean_market_cap.py
python Main_pipeline/09_KBrand_calculate.py
python Main_pipeline/10_Framework_calculate.py
python Main_pipeline/11_Classify_configurations.py
python Main_pipeline/12_RSSI_calculate.py
python Main_pipeline/13_adding_variables.py

# Step 4 — Run all statistical tests
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
python T10_RSSI_Validity.py
python T11_Structural_Bifurcation.py
python T12_Placebo_Validation_of_RSSI_Dual_Properties.py
python T13_Causality_Tests_For_Reflexifity_Theory.py
```

### Using pre-computed results

If you only want to inspect outputs without re-running the pipeline:

```bash
cd Staticits/
unzip results.zip
# Reports: results/reports/
# Tables:  results/tables/
# Figures: results/figures/
```

### Running a single test

Each test script is self-contained. Set `CLASSIFIED_DIR` at the top of each script to point to your `data/classified/` directory, then run directly:

```bash
cd Staticits/
python T12_Placebo_Validation_of_RSSI_Dual_Properties.py
```

Output is written to `results/reports/`, `results/tables/`, and `results/figures/` relative to the script's working directory.

---

## Computational Notes

- **Random seeds:** All permutation tests use `seed=42` with `N_PERM=1000` unless otherwise noted in the script header.
- **T2 Block 2E runtime:** The Wald test implementation runs in under one second. The original bootstrap implementation (1,000 permutations via sklearn) runs approximately 1–3 minutes depending on hardware.
- **T12 runtime:** Four permutation sub-tests, each with 1,000 permutations. Total runtime approximately 3–8 minutes on a standard laptop.
- **T13 runtime:** Pooled Granger causality tests. Runtime under one minute.
- **Data size:** `data/classified/` contains approximately 354 CSV files (~50MB total). `results.zip` contains all pre-computed outputs (~15MB).

---

## Known Issues

| Issue | Affected script | Description | Workaround |
|-------|----------------|-------------|------------|
| `next_config = 0` in Markov block | `02_Instability_analysis.py` | `next_config` computed within filtered sub-sample (C2/C3/C4 only), so transitions to C1/C6 show as 0. | Use `03_Conditional_transition.py` for all transition matrix results. T2 Markov block is not cited in the paper. |
| `statsmodels` not available | `02_Instability_analysis.py` | If `statsmodels` is not installed, p-values computed via Wald test (Fisher Information Matrix). Results are equivalent; pseudo R² requires intercept to be included in the matrix calculation (handled automatically). | No action needed — the script detects availability and chooses the correct method. |
| Wilcoxon p = nan | `08_Reflexive_loop_validation.py` | Loop break test (Block 8E) has insufficient Phi drop events with complete pre/post price windows (N_collapse = 14). | This result is reported as inconclusive in the paper. All other T8 results are unaffected. |
| Small cell sizes in test split | `T11_Structural_Bifurcation.py` | Mid_asc N=7 in 2022–2026 test split due to rarity of Mid_asc conditions under post-2022 macro regime. | Reported as descriptive only in the paper. Not used for inference. |
| Flat Kaplan-Meier after t=1 | `04_Joint_collpase_condition.py` | All events drop at t=1 by construction of the 4-quarter collapse window. | Shape is correct — reflects censoring structure, not a bug. Survival curves are valid for comparing relative fragility across groups. |
| Script name typos | Multiple | `04_Joint_collpase_condition.py`, `06_RSSI_Parapola.py`, `T13_Causality_Tests_For_Reflexifity_Theory.py` contain spelling errors. | Use filenames exactly as shown — do not correct spelling when running scripts or referencing paths. |

---

## Citation

**Paper:**
```bibtex
@misc{nguyenhong2026appearance,
  author    = {Nguy{\~{\^e}}n, H. P.},
  title     = {Appearance and Constraint: Behavioural Homomorphism, Environmental
               Stratification, and the Flow of Uninitiated Obligation in
               Speculative Markets},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19589022},
  url       = {https://doi.org/10.5281/zenodo.19589022}
}
```

**Companion paper:**
```bibtex
@misc{nguyenhong2026whatprice,
  author    = {Nguy{\~{\^e}}n, H. P.},
  title     = {What Price Presupposes: Quantifying Labour Obligations in
               Speculative Market Regimes},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19480971},
  url       = {https://doi.org/10.5281/zenodo.19480971}
}
```

---

## License

Apache 2.0. See `LICENSE`.
