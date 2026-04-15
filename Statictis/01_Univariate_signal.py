"""
Univariate Signal Analysis – Full Pipeline (No Re‑calculation)
---------------------------------------------------------------
This module performs a comprehensive univariate analysis of speculative firm
observations and sector‑level RSSI signals to assess predictive power for
subsequent collapse events (collapse_next = 1).

The analysis is structured into five blocks:
    1. Firm‑level standalone signals (D_t, B) across configurations.
    2. Sector‑level RSSI signal: concurrent, lagged, and peak‑to‑collapse.
    3. Two‑cycle market phase identification and collapse rates.
    4. Sector‑context variance decomposition and cross‑sectional correlations.
    5. Configuration‑specific lagged RSSI effects.

All results are exported as CSV tables to `results/tables/` and as publication‑
ready figures to `results/figures/`. A summary report is generated in
`results/reports/T1_Summary_Report.txt`.

Input data:
    - Classified firm‑quarter observations from `data/classified/<Sector>/`.
    - Historical RSSI series from `data/processed/<Sector>_RSSI_historical.csv`.

Author: [Phuc Nguyen Hong]
Date:   2026-04-14
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
CLASSIFIED_DIR = Path("../data/classified")
PROCESSED_DIR = Path("../data/processed")
OUTPUT_DIR = Path("results")
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

# Ensure output directories exist
for d in (TABLE_DIR, FIGURE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Analysis parameters
SECTORS: List[str] = ["Healthcare", "Technology", "Services"]
FIRM_VARIABLES: List[str] = ["D_t", "B"]
MAX_LAG: int = 12  # quarters
RSSI_THRESHOLD: float = 1.0  # threshold for high‑RSSI restriction in Block 2b
PEAK_HEIGHT: float = 0.5  # minimum peak height for cycle detection
PEAK_DISTANCE: int = 6  # minimum quarters between peaks


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_full_panel() -> pd.DataFrame:
    """
    Load all speculative firm‑quarter observations from classified CSV files.

    The function ensures that each observation contains the required columns:
        - Ticker, period_end, Configuration
        - D_t, B, RSSI, dRSSI_dt
        - Sector (inferred from directory name if missing)
        - collapse_next (derived from Configuration if missing)

    Returns
    -------
    pd.DataFrame
        Panel data sorted by Ticker and period_end.
    """
    required_columns = [
        "Ticker", "period_end", "Configuration",
        "D_t", "B", "RSSI", "dRSSI_dt"
    ]
    all_dataframes: List[pd.DataFrame] = []

    for sector in SECTORS:
        sector_path = CLASSIFIED_DIR / sector
        if not sector_path.exists():
            print(f"WARNING: Sector directory not found: {sector_path}")
            continue

        for file_path in sector_path.glob("*_classified.csv"):
            try:
                df = pd.read_csv(file_path, parse_dates=["period_end"])

                # 1. Add Sector column if missing
                if "Sector" not in df.columns:
                    df["Sector"] = sector

                # 2. Derive collapse_next from Configuration if missing
                if "collapse_next" not in df.columns:
                    if "Configuration" not in df.columns:
                        print(f"WARNING: {file_path.name} missing Configuration; cannot create collapse_next.")
                        continue
                    df = df.sort_values(["Ticker", "period_end"])
                    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                    df["collapse_next"] = df["next_config"].isin(["C1", "C6"]).astype(int)
                    df.drop(columns=["next_config"], inplace=True)

                # 3. Validate required columns
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    print(f"WARNING: {file_path.name} missing columns: {missing_cols}")
                    continue

                # Keep only necessary columns
                keep_cols = required_columns + ["Sector", "collapse_next"]
                df = df[keep_cols].copy()
                all_dataframes.append(df)

            except Exception as e:
                print(f"ERROR reading {file_path}: {e}")

    if not all_dataframes:
        raise FileNotFoundError("No valid classified files were loaded.")

    panel = pd.concat(all_dataframes, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel


def load_rssi_series() -> Dict[str, pd.Series]:
    """
    Load sector‑level historical RSSI time series for peak detection.

    Returns
    -------
    Dict[str, pd.Series]
        Dictionary mapping sector name to a datetime‑indexed RSSI series.
        Prefers 'RSSI_hist_winsor' if available, otherwise 'RSSI_hist'.
    """
    rssi_series_dict: Dict[str, pd.Series] = {}
    for sector in SECTORS:
        file_path = PROCESSED_DIR / f"{sector}_RSSI_historical.csv"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, parse_dates=["period_end"])
        column = "RSSI_hist_winsor" if "RSSI_hist_winsor" in df.columns else "RSSI_hist"
        if column not in df.columns:
            continue
        rssi_series_dict[sector] = df.set_index("period_end")[column].sort_index()
    return rssi_series_dict


# =============================================================================
# STATISTICAL HELPER FUNCTIONS
# =============================================================================
def logistic_auc_and_coef(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit a univariate logistic regression and compute AUC and coefficient.

    Parameters
    ----------
    x : np.ndarray
        Predictor variable (1D).
    y : np.ndarray
        Binary outcome (0/1).

    Returns
    -------
    Tuple[float, float]
        (AUC, coefficient). Returns (np.nan, np.nan) if y is constant.
    """
    if y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(x.reshape(-1, 1), y)
    y_prob = model.predict_proba(x.reshape(-1, 1))[:, 1]
    auc = roc_auc_score(y, y_prob)
    coef = model.coef_[0][0]
    return auc, coef


# =============================================================================
# BLOCK 1 – FIRM‑LEVEL STANDALONE SIGNALS
# =============================================================================
def block1_firm_standalone(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate univariate predictive power of firm‑level variables (D_t, B)
    across different configurations (All, C2, C3, C4).

    Metrics computed:
        - Spearman rank correlation (r, p‑value)
        - Logistic regression AUC and coefficient
        - Mann‑Whitney U test p‑value
        - Median values for collapse vs. non‑collapse groups

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel data (must contain 'Configuration', 'collapse_next' and the
        firm variables).

    Returns
    -------
    pd.DataFrame
        Summary table with one row per variable–configuration combination.
    """
    results: List[Dict] = []
    for var in FIRM_VARIABLES:
        for config in ["All", "C2", "C3", "C4"]:
            if config == "All":
                data = panel.dropna(subset=[var, "collapse_next"])
            else:
                data = panel[panel["Configuration"] == config].dropna(
                    subset=[var, "collapse_next"]
                )
            if len(data) == 0:
                continue

            # Spearman correlation
            r, p_spearman = stats.spearmanr(data[var], data["collapse_next"])
            # Logistic regression
            auc, coef = logistic_auc_and_coef(data[var].values, data["collapse_next"].values)
            # Mann‑Whitney U test
            g0 = data[data["collapse_next"] == 0][var]
            g1 = data[data["collapse_next"] == 1][var]
            if len(g0) > 0 and len(g1) > 0:
                _, p_mw = stats.mannwhitneyu(g0, g1, alternative="two-sided")
            else:
                p_mw = np.nan

            results.append({
                "Variable": var,
                "Configuration": config,
                "N": len(data),
                "Collapse_N": int(data["collapse_next"].sum()),
                "Collapse_Rate": data["collapse_next"].mean(),
                "Spearman_r": r,
                "Spearman_p": p_spearman,
                "AUC": auc,
                "Logit_coef": coef,
                "MW_p": p_mw,
                "Median_NonColl": g0.median(),
                "Median_Coll": g1.median(),
            })
    return pd.DataFrame(results)


# =============================================================================
# BLOCK 2a – RSSI CONCURRENT (t)
# =============================================================================
def block2a_rssi_concurrent(panel: pd.DataFrame) -> Dict[str, Union[int, float]]:
    """
    Compute concurrent (same‑quarter) association between RSSI and collapse_next.

    Returns
    -------
    Dict
        Contains N, Spearman r, p‑value, and AUC.
        Returns empty dict if no valid observations.
    """
    data = panel.dropna(subset=["RSSI", "collapse_next"])
    if len(data) == 0:
        return {}
    r, p = stats.spearmanr(data["RSSI"], data["collapse_next"])
    auc, _ = logistic_auc_and_coef(data["RSSI"].values, data["collapse_next"].values)
    return {"N": len(data), "Spearman_r": r, "p_value": p, "AUC": auc}


# =============================================================================
# BLOCK 2b – RSSI LAG STRUCTURE (DETRENDED & RESTRICTED)
# =============================================================================
def block2b_rssi_lag_structure(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation between detrended RSSI at various lags and
    collapse_next. Analysis is restricted to firms that ever exceed the
    RSSI_THRESHOLD.

    Detrending is performed using a rolling mean of 8 quarters within each sector.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel data.

    Returns
    -------
    pd.DataFrame
        Columns: Lag_quarters, Spearman_r, p_value, N.
    """
    df = panel.copy()
    df = df.sort_values(["Sector", "period_end"])

    # Sector‑specific detrending: 8‑quarter rolling mean
    for sector in SECTORS:
        mask = df["Sector"] == sector
        if mask.sum() > 2:
            roll_mean = df.loc[mask, "RSSI"].rolling(8, min_periods=4).mean()
            df.loc[mask, "RSSI_det"] = df.loc[mask, "RSSI"] - roll_mean

    # Restrict to firms that ever had RSSI above threshold
    high_tickers = df[df["RSSI"] > RSSI_THRESHOLD]["Ticker"].unique()
    df = df[df["Ticker"].isin(high_tickers)].copy()

    results = []
    for lag in range(1, MAX_LAG + 1):
        df[f"RSSI_lag{lag}"] = df.groupby("Sector")["RSSI_det"].shift(lag)
        valid = df.dropna(subset=[f"RSSI_lag{lag}", "collapse_next"])
        if len(valid) == 0:
            r, p = np.nan, np.nan
        else:
            r, p = stats.spearmanr(valid[f"RSSI_lag{lag}"], valid["collapse_next"])
        results.append({
            "Lag_quarters": lag,
            "Spearman_r": r,
            "p_value": p,
            "N": len(valid)
        })
    return pd.DataFrame(results)


# =============================================================================
# BLOCK 2c – PEAK‑TO‑COLLAPSE LAG DISTRIBUTION
# =============================================================================
def block2c_peak_to_collapse(
    panel: pd.DataFrame, rssi_dict: Dict[str, pd.Series]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    For each collapse event, find the nearest preceding RSSI peak (within
    MAX_LAG quarters) and compute the lag in quarters.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data with collapse_next and Sector.
    rssi_dict : Dict[str, pd.Series]
        Sector‑level RSSI series.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        - DataFrame with column 'lag_quarters' for each collapse.
        - Dictionary of summary statistics (mean, median, std, quantiles).
    """
    collapses = panel[panel["collapse_next"] == 1].copy()
    lags: List[int] = []

    for ticker, group in collapses.groupby("Ticker"):
        collapse_date = group["period_end"].min()
        sector = group["Sector"].iloc[0]
        if sector not in rssi_dict:
            continue

        rssi_series = rssi_dict[sector].dropna()
        # Define window ending at collapse date
        start_date = collapse_date - pd.DateOffset(months=3 * MAX_LAG)
        window = rssi_series.loc[start_date:collapse_date].dropna()
        if len(window) < 2:
            continue

        peak_date = window.idxmax()
        lag = (collapse_date.to_period("Q") - peak_date.to_period("Q")).n
        lags.append(lag)

    if not lags:
        return pd.DataFrame(), {}

    df_lag = pd.DataFrame({"lag_quarters": lags})
    stats_dict = {
        "N": len(lags),
        "Mean": np.mean(lags),
        "Median": np.median(lags),
        "Std": np.std(lags),
        "Min": np.min(lags),
        "Max": np.max(lags),
        "Q25": np.percentile(lags, 25),
        "Q75": np.percentile(lags, 75),
    }
    return df_lag, stats_dict


# =============================================================================
# BLOCK 3 – TWO‑CYCLE MARKET PHASE ANALYSIS
# =============================================================================
def block3_cycle_analysis(
    panel: pd.DataFrame, rssi_dict: Dict[str, pd.Series]
) -> Tuple[Dict, pd.DataFrame]:
    """
    Identify RSSI peaks in each sector to define two market cycles.
    Classify each observation into phases: pre_cycle1, cycle_1, cycle_2, quiet.
    Compute collapse rates by phase and configuration.

    Returns
    -------
    Tuple[Dict, pd.DataFrame]
        - cycle_info: dictionary with peak details per sector.
        - rates: DataFrame with mean collapse rate by phase and configuration.
    """
    cycle_info: Dict = {}
    phase_dataframes: List[pd.DataFrame] = []

    for sector in SECTORS:
        if sector not in rssi_dict:
            continue
        rssi_series = rssi_dict[sector].dropna()
        vals = rssi_series.values
        peaks, props = find_peaks(vals, height=PEAK_HEIGHT, distance=PEAK_DISTANCE)
        peak_dates = rssi_series.iloc[peaks].index
        cycle_info[sector] = {
            "n_peaks": len(peaks),
            "peak_dates": peak_dates,
            "peak_heights": props["peak_heights"]
        }

        if len(peak_dates) >= 2:
            df_sector = panel[panel["Sector"] == sector].copy()
            df_sector["phase"] = "quiet"
            df_sector.loc[df_sector["period_end"] < peak_dates[0], "phase"] = "pre_cycle1"
            df_sector.loc[
                (df_sector["period_end"] >= peak_dates[0]) &
                (df_sector["period_end"] < peak_dates[1]),
                "phase"
            ] = "cycle_1"
            df_sector.loc[df_sector["period_end"] >= peak_dates[1], "phase"] = "cycle_2"
            phase_dataframes.append(df_sector[["Ticker", "period_end", "phase"]])

    if phase_dataframes:
        phase_all = pd.concat(phase_dataframes)
        panel_phase = panel.merge(phase_all, on=["Ticker", "period_end"], how="left")
    else:
        panel_phase = panel.copy()
        panel_phase["phase"] = "unknown"
    panel_phase["phase"] = panel_phase["phase"].fillna("unknown")

    # Collapse rates by phase and configuration (excluding 'Normal')
    rates = (
        panel_phase[panel_phase["Configuration"] != "Normal"]
        .groupby(["phase", "Configuration"])["collapse_next"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return cycle_info, rates


# =============================================================================
# BLOCK 4 – SECTOR CONTEXT (VARIANCE DECOMPOSITION)
# =============================================================================
def block4_sector_context(
    panel: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Decompose RSSI variance into within‑ and between‑sector components,
    and compute cross‑sectional correlations of RSSI with D_t and B for each quarter.

    Returns
    -------
    Tuple[Dict[str, float], pd.DataFrame]
        - stats: 'within_sector_var', 'between_sector_var'
        - corr_df: quarterly cross‑sectional correlations.
    """
    df = panel.dropna(subset=["RSSI"])
    if len(df) == 0:
        return {"within_sector_var": np.nan, "between_sector_var": np.nan}, pd.DataFrame()

    within_var = df.groupby("Sector")["RSSI"].var().mean()
    between_var = df.groupby("period_end")["RSSI"].mean().var()

    corr_records = []
    for quarter, group in df.groupby("period_end"):
        if len(group) > 5:
            r_d, _ = stats.spearmanr(group["RSSI"], group["D_t"])
            r_b, _ = stats.spearmanr(group["RSSI"], group["B"])
            corr_records.append({
                "period_end": quarter,
                "corr_RSSI_Dt": r_d,
                "corr_RSSI_B": r_b
            })
    corr_df = pd.DataFrame(corr_records)
    stats_dict = {"within_sector_var": within_var, "between_sector_var": between_var}
    return stats_dict, corr_df


# =============================================================================
# BLOCK 5 – SIGNAL BY CONFIGURATION (SELECTED LAGS)
# =============================================================================
def block5_by_configuration(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate lagged RSSI predictive power within each speculative configuration
    (C2, C3, C4) for lags 1, 4, and 8 quarters.

    Returns
    -------
    pd.DataFrame
        Columns: Config, Lag, Spearman_r, p, N.
    """
    results = []
    for config in ["C2", "C3", "C4"]:
        df_conf = panel[panel["Configuration"] == config].copy()
        if len(df_conf) < 50:
            continue
        for lag in [1, 4, 8]:
            df_conf[f"RSSI_lag{lag}"] = df_conf.groupby("Sector")["RSSI"].shift(lag)
            valid = df_conf.dropna(subset=[f"RSSI_lag{lag}", "collapse_next"])
            if len(valid) > 10:
                r, p = stats.spearmanr(valid[f"RSSI_lag{lag}"], valid["collapse_next"])
                results.append({
                    "Config": config,
                    "Lag": lag,
                    "Spearman_r": r,
                    "p": p,
                    "N": len(valid)
                })
    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_lag_correlation_curve(lag_df: pd.DataFrame) -> None:
    """Plot Spearman correlation across RSSI lags (Block 2b)."""
    plt.figure(figsize=(10, 5))
    plt.plot(lag_df["Lag_quarters"], lag_df["Spearman_r"], marker="o", linewidth=2)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Lag (quarters)", fontsize=12)
    plt.ylabel("Spearman correlation", fontsize=12)
    plt.title("RSSI Lag Correlation with Collapse", fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(range(1, MAX_LAG + 1))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T1_lag_curve.png", dpi=150)
    plt.close()


def plot_peak_lag_histogram(lag_series: pd.Series) -> None:
    """Histogram of peak‑to‑collapse lags (Block 2c)."""
    plt.figure(figsize=(8, 5))
    bins = range(0, int(lag_series.max()) + 2)
    plt.hist(lag_series, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Quarters from RSSI peak to collapse", fontsize=12)
    plt.ylabel("Number of firms", fontsize=12)
    plt.title("Peak‑to‑Collapse Lag Distribution", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T1_peak_lag_histogram.png", dpi=150)
    plt.close()


def plot_phase_collapse_rates(rates_df: pd.DataFrame) -> None:
    """Bar plot of average collapse rate by market phase (Block 3)."""
    phase_rates = rates_df.groupby("phase")["mean"].mean().reset_index()
    phase_rates.columns = ["Phase", "Collapse rate"]
    order = ["pre_cycle1", "cycle_1", "quiet", "cycle_2", "unknown"]
    phase_rates["Phase"] = pd.Categorical(phase_rates["Phase"], categories=order, ordered=True)
    phase_rates = phase_rates.sort_values("Phase")

    plt.figure(figsize=(8, 5))
    bars = plt.bar(phase_rates["Phase"], phase_rates["Collapse rate"],
                   color="steelblue", edgecolor="black")
    plt.xlabel("RSSI Cycle Phase", fontsize=12)
    plt.ylabel("Average Collapse Rate", fontsize=12)
    plt.title("Collapse Rate by Market Phase", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f"{height:.2%}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T1_phase_collapse.png", dpi=150)
    plt.close()


# =============================================================================
# ACADEMIC SUMMARY REPORT GENERATION
# =============================================================================
def generate_academic_report(
    panel: pd.DataFrame,
    blk1: pd.DataFrame,
    blk2a: Dict,
    blk2b: pd.DataFrame,
    blk2c_stats: Dict,
    blk3_cycles: Dict,
    blk3_rates: pd.DataFrame,
    blk4_stats: Dict,
    blk5: pd.DataFrame,
    blk4_corr: Optional[pd.DataFrame] = None,
) -> None:
    """
    Write a comprehensive, publication‑style summary report in plain text.
    """
    report_path = REPORT_DIR / "T1_Summary_Report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("UNIVARIATE SIGNAL ANALYSIS – ACADEMIC SUMMARY REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        # Sample description
        f.write("1. SAMPLE DESCRIPTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total speculative observations (N)          : {len(panel):,}\n")
        f.write(f"Number of unique firms                       : {panel['Ticker'].nunique()}\n")
        f.write(f"Sectors analysed                             : {', '.join(SECTORS)}\n")
        config_counts = panel["Configuration"].value_counts()
        f.write("Configuration distribution:\n")
        for cfg, cnt in config_counts.items():
            f.write(f"  {cfg:5s} : {cnt:6d} ({100 * cnt / len(panel):5.1f}%)\n")
        f.write("\n")

        # Block 1
        f.write("2. FIRM‑LEVEL STANDALONE SIGNALS (D_t, B)\n")
        f.write("-" * 40 + "\n")
        if not blk1.empty:
            summary = blk1[blk1["Configuration"] == "All"][
                ["Variable", "Spearman_r", "Spearman_p", "AUC"]
            ]
            f.write(summary.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No results available.\n\n")

        # Block 2a
        f.write("3. RSSI CONCURRENT ASSOCIATION (t)\n")
        f.write("-" * 40 + "\n")
        if blk2a:
            f.write(f"N                = {blk2a['N']}\n")
            f.write(f"Spearman r       = {blk2a['Spearman_r']:.4f}\n")
            f.write(f"p‑value          = {blk2a['p_value']:.4e}\n")
            f.write(f"AUC              = {blk2a['AUC']:.4f}\n\n")
        else:
            f.write("No results.\n\n")

        # Block 2b
        f.write("4. RSSI LAG STRUCTURE (detrended, high‑RSSI firms only)\n")
        f.write("-" * 40 + "\n")
        if not blk2b.empty:
            best_idx = blk2b["Spearman_r"].abs().idxmax()
            best = blk2b.loc[best_idx]
            f.write(f"Optimal lag      : {int(best['Lag_quarters'])} quarters\n")
            f.write(f"Spearman r       : {best['Spearman_r']:.4f}\n")
            f.write(f"p‑value          : {best['p_value']:.4e}\n\n")
            f.write("Full lag profile:\n")
            f.write(blk2b.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No results.\n\n")

        # Block 2c
        f.write("5. PEAK‑TO‑COLLAPSE LAG DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        if blk2c_stats:
            f.write(f"Number of collapse events  : {blk2c_stats['N']}\n")
            f.write(f"Mean lag (quarters)        : {blk2c_stats['Mean']:.2f}\n")
            f.write(f"Median lag (quarters)      : {blk2c_stats['Median']:.2f}\n")
            f.write(f"Standard deviation         : {blk2c_stats['Std']:.2f}\n")
            f.write(f"Range                      : [{blk2c_stats['Min']}, {blk2c_stats['Max']}]\n")
            f.write(f"Interquartile range        : [{blk2c_stats['Q25']:.2f}, {blk2c_stats['Q75']:.2f}]\n\n")
        else:
            f.write("No results.\n\n")

        # Block 3
        f.write("6. TWO‑CYCLE MARKET PHASE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write("Detected RSSI peaks by sector:\n")
        for sec, info in blk3_cycles.items():
            dates_str = ", ".join(d.strftime("%Y‑%m") for d in info["peak_dates"])
            f.write(f"  {sec:12s} : {info['n_peaks']} peaks at {dates_str}\n")
        f.write("\nCollapse rate by phase and configuration:\n")
        f.write(blk3_rates.to_string(index=False))
        f.write("\n\n")

        # Block 4
        f.write("7. SECTOR CONTEXT – VARIANCE DECOMPOSITION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Within‑sector RSSI variance    : {blk4_stats['within_sector_var']:.4f}\n")
        f.write(f"Between‑sector RSSI variance   : {blk4_stats['between_sector_var']:.4f}\n")
        if blk4_corr is not None and not blk4_corr.empty:
            mean_corr_d = blk4_corr["corr_RSSI_Dt"].mean()
            mean_corr_b = blk4_corr["corr_RSSI_B"].mean()
            f.write(f"Mean cross‑sectional corr(RSSI, D_t) : {mean_corr_d:.4f}\n")
            f.write(f"Mean cross‑sectional corr(RSSI, B)   : {mean_corr_b:.4f}\n")
        f.write("\n")

        # Block 5
        f.write("8. CONFIGURATION‑SPECIFIC LAGGED RSSI EFFECTS\n")
        f.write("-" * 40 + "\n")
        if not blk5.empty:
            f.write(blk5.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No results.\n\n")

        f.write("=" * 80 + "\n")
        f.write("End of report.\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute the full univariate signal analysis pipeline."""
    print("=" * 70)
    print("UNIVARIATE SIGNAL ANALYSIS – ACADEMIC PIPELINE")
    print("=" * 70)

    # Load data
    panel = load_full_panel()
    panel = panel[panel["Configuration"] != "Normal"].copy()
    rssi_dict = load_rssi_series()
    print(f"Loaded {len(panel):,} speculative observations from {panel['Ticker'].nunique()} firms.\n")

    # Block 1
    print("Block 1: Firm‑level standalone signals...")
    blk1 = block1_firm_standalone(panel)
    blk1.to_csv(TABLE_DIR / "T1_block1_standalone.csv", index=False)

    # Block 2a
    print("Block 2a: RSSI concurrent...")
    blk2a = block2a_rssi_concurrent(panel)
    pd.DataFrame([blk2a]).to_csv(TABLE_DIR / "T1_block2a_concurrent.csv", index=False)

    # Block 2b
    print("Block 2b: RSSI lag structure...")
    blk2b = block2b_rssi_lag_structure(panel)
    blk2b.to_csv(TABLE_DIR / "T1_block2b_lag_structure.csv", index=False)
    plot_lag_correlation_curve(blk2b)

    # Block 2c
    print("Block 2c: Peak‑to‑collapse lag...")
    blk2c_df, blk2c_stats = block2c_peak_to_collapse(panel, rssi_dict)
    if blk2c_stats:
        blk2c_df.to_csv(TABLE_DIR / "T1_block2c_peak_lag.csv", index=False)
        pd.DataFrame([blk2c_stats]).to_csv(TABLE_DIR / "T1_block2c_stats.csv", index=False)
        plot_peak_lag_histogram(blk2c_df["lag_quarters"])

    # Block 3
    print("Block 3: Two‑cycle market phases...")
    blk3_cycles, blk3_rates = block3_cycle_analysis(panel, rssi_dict)
    blk3_rates.to_csv(TABLE_DIR / "T1_block3_phase_rates.csv", index=False)
    with open(TABLE_DIR / "T1_block3_cycles.txt", "w") as f:
        for sec, info in blk3_cycles.items():
            f.write(f"{sec}: {info['n_peaks']} peaks\n")
    plot_phase_collapse_rates(blk3_rates)

    # Block 4
    print("Block 4: Sector context...")
    blk4_stats, blk4_corr = block4_sector_context(panel)
    pd.DataFrame([blk4_stats]).to_csv(TABLE_DIR / "T1_block4_stats.csv", index=False)
    if not blk4_corr.empty:
        blk4_corr.to_csv(TABLE_DIR / "T1_block4_correlations.csv", index=False)

    # Block 5
    print("Block 5: Configuration‑specific lags...")
    blk5 = block5_by_configuration(panel)
    if not blk5.empty:
        blk5.to_csv(TABLE_DIR / "T1_block5_by_config.csv", index=False)

    print("\nAll results saved in:", TABLE_DIR.resolve())
    print("Figures saved in:", FIGURE_DIR.resolve())

    # Generate academic summary report
    generate_academic_report(
        panel, blk1, blk2a, blk2b, blk2c_stats,
        blk3_cycles, blk3_rates, blk4_stats, blk5, blk4_corr
    )


if __name__ == "__main__":
    main()