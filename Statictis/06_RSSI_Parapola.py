"""
RSSI Parabola – Sector‑Level Trajectory Validation (Test 6)
-------------------------------------------------------------
This module examines the sector‑level RSSI trajectory around firm collapse
events to validate its parabolic (inverted‑U) shape and predictive utility.

Key analyses:
    6A. Aligned RSSI trajectory across all collapses (median with 95% CI).
    6B. Formal shape test: quadratic vs. linear fit (F‑test, AIC, R²).
    6C. Peak‑to‑collapse lag distribution and uniformity test.
    6D. Two‑cycle structure detection via peak finding.
    6E. Mirror test: correlation between normalized RSSI and firm‑level B
        trajectories around collapse.
    6F. FC2 prediction: RSSI sign post‑collapse as an indicator of cyclical
        bottom vs. structural terminus.

Results are exported to `results/tables/` and `results/figures/`, with a
comprehensive summary report in `results/reports/T6_RSSI_Parabola_Report.txt`.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix

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

for directory in (TABLE_DIR, FIGURE_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SECTORS = ["Healthcare", "Technology", "Services"]
COLLAPSE_LAG_WINDOW = 12  # quarters before collapse
POST_COLLAPSE_WINDOW = 4   # quarters after collapse


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_full_panel() -> pd.DataFrame:
    """
    Load all firm‑quarter observations from classified CSV files.

    Returns
    -------
    pd.DataFrame
        Panel data with required columns and derived 'collapse_next'.
    """
    required_columns = ["Ticker", "period_end", "Configuration", "RSSI", "B", "R_t"]
    all_dfs: List[pd.DataFrame] = []

    for sector in SECTORS:
        sector_path = CLASSIFIED_DIR / sector
        if not sector_path.exists():
            continue
        for file_path in sector_path.glob("*_classified.csv"):
            try:
                df = pd.read_csv(file_path, parse_dates=["period_end"])

                if "Sector" not in df.columns:
                    df["Sector"] = sector

                if "collapse_next" not in df.columns:
                    df = df.sort_values(["Ticker", "period_end"])
                    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                    df["collapse_next"] = df["next_config"].isin(["C1", "C6"]).astype(int)
                    df.drop(columns=["next_config"], inplace=True)

                missing = [c for c in required_columns if c not in df.columns]
                if missing:
                    continue
                df = df[required_columns + ["Sector", "collapse_next"]].copy()
                all_dfs.append(df)
            except Exception:
                continue

    if not all_dfs:
        raise ValueError("No valid classified data could be loaded.")

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel


def prepare_collapse_events(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique collapse events (collapse_next == 1) with Ticker, Sector,
    and period_end.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel data.

    Returns
    -------
    pd.DataFrame
        Unique collapse events.
    """
    collapses = panel[panel["collapse_next"] == 1].copy()
    collapses = collapses[["Ticker", "Sector", "period_end"]].drop_duplicates()
    collapses = collapses.sort_values("period_end")
    return collapses


def align_rssi_to_collapse(
    rssi_series: pd.Series, collapse_date: pd.Timestamp
) -> pd.Series:
    """
    Extract a window of RSSI values around a collapse date.

    The window spans from COLLAPSE_LAG_WINDOW quarters before collapse to
    POST_COLLAPSE_WINDOW quarters after. The index is shifted so that t=0
    corresponds to the collapse quarter.

    Parameters
    ----------
    rssi_series : pd.Series
        Sector‑level RSSI time series (datetime index).
    collapse_date : pd.Timestamp
        Date of the collapse event.

    Returns
    -------
    pd.Series
        Aligned RSSI trace with relative quarter index.
    """
    rssi = rssi_series.dropna().sort_index()
    if collapse_date not in rssi.index:
        return pd.Series(dtype=float)

    idx = rssi.index.get_loc(collapse_date)
    start = max(0, idx - COLLAPSE_LAG_WINDOW)
    end = min(len(rssi), idx + POST_COLLAPSE_WINDOW + 1)
    aligned = rssi.iloc[start:end].copy()
    aligned.index = range(start - idx, end - idx)
    return aligned


# =============================================================================
# BLOCK 6A – ALIGNED RSSI TRAJECTORY
# =============================================================================
def block6a_aligned_trajectory(
    rssi_dict: Dict[str, pd.Series], collapses: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """
    Compute the median RSSI trajectory across all collapse events, with
    95% bootstrap confidence intervals.

    Parameters
    ----------
    rssi_dict : Dict[str, pd.Series]
        Sector‑level RSSI series.
    collapses : pd.DataFrame
        Collapse events (Ticker, Sector, period_end).

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns: t, median, ci_low, ci_high.
        Returns None if insufficient traces.
    """
    all_traces: List[pd.Series] = []
    for sector in SECTORS:
        if sector not in rssi_dict:
            continue
        rssi_series = rssi_dict[sector]
        sector_collapses = collapses[collapses["Sector"] == sector]["period_end"]
        for collapse_date in sector_collapses:
            trace = align_rssi_to_collapse(rssi_series, collapse_date)
            if len(trace) > 5:
                all_traces.append(trace)

    if not all_traces:
        return None

    df_traces = pd.concat(all_traces, axis=1, sort=True).T
    stats_dict = {}
    for t in df_traces.columns:
        vals = df_traces[t].dropna().values
        if len(vals) >= 5:
            med = np.median(vals)
            ci_low = np.percentile(vals, 2.5)
            ci_high = np.percentile(vals, 97.5)
        else:
            med = ci_low = ci_high = np.nan
        stats_dict[t] = {"median": med, "ci_low": ci_low, "ci_high": ci_high}

    traj_df = pd.DataFrame.from_dict(stats_dict, orient="index").reset_index()
    traj_df.columns = ["t", "median", "ci_low", "ci_high"]
    traj_df = traj_df.sort_values("t").dropna()
    traj_df.to_csv(TABLE_DIR / "T6_6A_aligned_trajectory.csv", index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(traj_df["t"], traj_df["median"], "b-", linewidth=2, label="Median RSSI")
    plt.fill_between(
        traj_df["t"], traj_df["ci_low"], traj_df["ci_high"],
        alpha=0.3, color="blue", label="95% CI"
    )
    plt.axvline(x=0, color="red", linestyle="--", label="Collapse (t=0)")
    plt.xlabel("Quarters Relative to Collapse", fontsize=12)
    plt.ylabel("RSSI", fontsize=12)
    plt.title("RSSI Trajectory Aligned by Collapse (All Sectors)", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T6_rssi_trajectory_aligned.png", dpi=150)
    plt.close()

    return traj_df


# =============================================================================
# BLOCK 6B – SHAPE TEST (QUADRATIC vs. LINEAR)
# =============================================================================
def block6b_shape_test(traj_df: Optional[pd.DataFrame]) -> Optional[Dict]:
    """
    Test whether the aligned RSSI trajectory follows a quadratic (inverted‑U)
    shape rather than a linear trend.

    Fits linear and quadratic models, computes R², AIC, and an F‑test for
    the improvement of the quadratic model.

    Parameters
    ----------
    traj_df : Optional[pd.DataFrame]
        Output from block6a_aligned_trajectory.

    Returns
    -------
    Optional[Dict]
        Dictionary with fit statistics and coefficient estimates.
    """
    if traj_df is None or len(traj_df) < 5:
        return None

    t = traj_df["t"].values
    y = traj_df["median"].values

    def linear(t, a, b):
        return a + b * t

    def quadratic(t, a, b, c):
        return a + b * t + c * t**2

    popt_lin, _ = curve_fit(linear, t, y)
    popt_quad, _ = curve_fit(quadratic, t, y)

    y_pred_lin = linear(t, *popt_lin)
    y_pred_quad = quadratic(t, *popt_quad)

    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
    ss_res_quad = np.sum((y - y_pred_quad) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2_lin = 1 - ss_res_lin / ss_tot
    r2_quad = 1 - ss_res_quad / ss_tot

    n = len(t)
    aic_lin = n * np.log(ss_res_lin / n) + 2 * 2
    aic_quad = n * np.log(ss_res_quad / n) + 2 * 3

    f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n - 3))
    p_f = 1 - stats.f.cdf(f_stat, 1, n - 3)

    results = {
        "R2_linear": r2_lin,
        "R2_quadratic": r2_quad,
        "AIC_linear": aic_lin,
        "AIC_quadratic": aic_quad,
        "F_stat": f_stat,
        "p_value": p_f,
        "quad_coef_a": popt_quad[0],
        "quad_coef_b": popt_quad[1],
        "quad_coef_c": popt_quad[2],
    }
    pd.DataFrame([results]).to_csv(TABLE_DIR / "T6_6B_shape_test.csv", index=False)

    # Plot fits
    t_smooth = np.linspace(t.min(), t.max(), 100)
    plt.figure(figsize=(10, 6))
    plt.scatter(t, y, color="black", label="Observed Median")
    plt.plot(t_smooth, linear(t_smooth, *popt_lin), "g--", label="Linear Fit")
    plt.plot(t_smooth, quadratic(t_smooth, *popt_quad), "r-", label="Quadratic Fit")
    plt.axvline(x=0, color="red", linestyle=":", alpha=0.5)
    plt.xlabel("t (Quarters Relative to Collapse)", fontsize=12)
    plt.ylabel("RSSI", fontsize=12)
    plt.title("Quadratic vs. Linear Fit for RSSI Trajectory", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T6_quadratic_fit.png", dpi=150)
    plt.close()

    return results


# =============================================================================
# BLOCK 6C – PEAK‑TO‑COLLAPSE LAG DISTRIBUTION
# =============================================================================
def block6c_peak_timing(
    rssi_dict: Dict[str, pd.Series], collapses: pd.DataFrame
) -> Optional[Dict]:
    """
    Compute the lag (in quarters) from the local RSSI peak preceding each
    collapse event. Test uniformity using a Kolmogorov–Smirnov test.

    Parameters
    ----------
    rssi_dict : Dict[str, pd.Series]
        Sector‑level RSSI series.
    collapses : pd.DataFrame
        Collapse events.

    Returns
    -------
    Optional[Dict]
        Summary statistics and KS test p‑value.
    """
    lags: List[float] = []
    for sector in SECTORS:
        if sector not in rssi_dict:
            continue
        rssi_series = rssi_dict[sector]
        sector_collapses = collapses[collapses["Sector"] == sector]["period_end"]
        for collapse_date in sector_collapses:
            window = rssi_series.loc[:collapse_date].tail(COLLAPSE_LAG_WINDOW + 1)
            if len(window) < 5:
                continue
            peak_date = window.idxmax()
            lag = (collapse_date - peak_date).days / 91.25
            if 0 <= lag <= COLLAPSE_LAG_WINDOW:
                lags.append(lag)

    if not lags:
        return None

    lags = np.array(lags)
    stats_dict = {
        "N": len(lags),
        "mean": np.mean(lags),
        "median": np.median(lags),
        "std": np.std(lags),
        "q25": np.percentile(lags, 25),
        "q75": np.percentile(lags, 75),
    }
    ks_stat, p_ks = stats.kstest(lags, "uniform", args=(0, COLLAPSE_LAG_WINDOW))
    results = {**stats_dict, "ks_p": p_ks}
    pd.DataFrame([results]).to_csv(TABLE_DIR / "T6_6C_peak_timing.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(lags, bins=np.arange(0, COLLAPSE_LAG_WINDOW + 1, 1),
             edgecolor="black", alpha=0.7)
    plt.axvline(np.median(lags), color="red", linestyle="--",
                label=f"Median = {np.median(lags):.1f} q")
    plt.xlabel("Quarters from RSSI Peak to Collapse", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Peak‑to‑Collapse Lag Distribution", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T6_peak_lag_histogram.png", dpi=150)
    plt.close()

    return results


# =============================================================================
# BLOCK 6D – TWO‑CYCLE STRUCTURE DETECTION
# =============================================================================
def block6d_two_cycle_structure(
    rssi_dict: Dict[str, pd.Series]
) -> Dict[str, Dict]:
    """
    Identify RSSI peaks in each sector's historical series to delineate
    market cycles. Peaks are detected using `find_peaks` with minimum height
    and distance constraints.

    Parameters
    ----------
    rssi_dict : Dict[str, pd.Series]
        Sector‑level RSSI series.

    Returns
    -------
    Dict[str, Dict]
        Dictionary with peak information per sector.
    """
    cycles_info = {}
    for sector in SECTORS:
        if sector not in rssi_dict:
            continue
        rssi = rssi_dict[sector].dropna()
        vals = rssi.values
        peaks, props = find_peaks(vals, height=0.5, distance=6)
        if len(peaks) > 0:
            peak_dates = rssi.index[peaks]
            peak_vals = vals[peaks]
            cycles_info[sector] = {
                "n_peaks": len(peaks),
                "peak_dates": peak_dates,
                "peak_values": peak_vals,
            }

            # Plot
            plt.figure(figsize=(12, 4))
            plt.plot(rssi.index, rssi.values, "b-", label="RSSI")
            plt.scatter(peak_dates, peak_vals, color="red", s=80,
                        marker="^", label="Detected Peaks")
            plt.title(f"{sector} – RSSI with Detected Peaks", fontsize=14)
            plt.xlabel("Quarter", fontsize=12)
            plt.ylabel("RSSI", fontsize=12)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURE_DIR / f"T6_two_cycles_{sector}.png", dpi=150)
            plt.close()

    # Save to CSV
    rows = []
    for sec, info in cycles_info.items():
        for i, (date, val) in enumerate(zip(info["peak_dates"], info["peak_values"])):
            rows.append({"Sector": sec, "Peak_index": i + 1,
                         "Date": date, "RSSI": val})
    if rows:
        pd.DataFrame(rows).to_csv(TABLE_DIR / "T6_6D_cycles.csv", index=False)

    return cycles_info


# =============================================================================
# BLOCK 6E – MIRROR TEST (RSSI vs. B)
# =============================================================================
def block6e_mirror_test(
    panel: pd.DataFrame, traj_df: Optional[pd.DataFrame]
) -> Optional[Dict]:
    """
    Compare normalized RSSI and firm‑level B trajectories around collapse
    events. A strong positive correlation suggests that the RSSI parabola
    mirrors firm‑level balance‑sheet dynamics.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel data.
    traj_df : Optional[pd.DataFrame]
        Aligned RSSI trajectory from Block 6A.

    Returns
    -------
    Optional[Dict]
        Spearman correlation and p‑value.
    """
    if traj_df is None:
        return None

    spec = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    collapses = spec[spec["collapse_next"] == 1][["Ticker", "period_end"]].drop_duplicates()

    b_traces = []
    for _, row in collapses.iterrows():
        ticker, collapse_date = row["Ticker"], row["period_end"]
        firm_data = spec[spec["Ticker"] == ticker].sort_values("period_end")
        if collapse_date not in firm_data["period_end"].values:
            continue
        idx = firm_data[firm_data["period_end"] == collapse_date].index[0]
        start = max(0, idx - COLLAPSE_LAG_WINDOW)
        end = min(len(firm_data), idx + POST_COLLAPSE_WINDOW + 1)
        trace = firm_data.iloc[start:end][["period_end", "B"]].copy()
        trace["t"] = range(start - idx, end - idx)
        b_traces.append(trace[["t", "B"]])

    if not b_traces:
        return None

    df_b = pd.concat(b_traces).groupby("t")["B"].median().reset_index(name="median_B")
    merged = df_b.merge(traj_df[["t", "median"]], on="t", how="inner")
    merged.rename(columns={"median": "median_RSSI"}, inplace=True)

    # Normalize to [0, 1]
    merged["B_norm"] = (merged["median_B"] - merged["median_B"].min()) / \
                       (merged["median_B"].max() - merged["median_B"].min())
    merged["RSSI_norm"] = (merged["median_RSSI"] - merged["median_RSSI"].min()) / \
                          (merged["median_RSSI"].max() - merged["median_RSSI"].min())

    corr, p_corr = stats.spearmanr(merged["RSSI_norm"], merged["B_norm"])

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(merged["t"], merged["RSSI_norm"], "b-", label="RSSI (normalized)")
    ax1.plot(merged["t"], merged["B_norm"], "r--", label="B (normalized)")
    ax1.axvline(x=0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_xlabel("Quarters Relative to Collapse", fontsize=12)
    ax1.set_ylabel("Normalized Value", fontsize=12)
    ax1.set_title("RSSI vs. B Trajectory Around Collapse", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T6_mirror_RSSI_B.png", dpi=150)
    plt.close()

    results = {"Spearman_r": corr, "p_value": p_corr}
    pd.DataFrame([results]).to_csv(TABLE_DIR / "T6_6E_mirror.csv", index=False)
    return results


# =============================================================================
# BLOCK 6F – FC2: CYCLE BOTTOM PREDICTION
# =============================================================================
def block6f_fc2_prediction(
    panel: pd.DataFrame, rssi_dict: Dict[str, pd.Series]
) -> Optional[Dict]:
    """
    Test the FC2 hypothesis: the sign of RSSI immediately after a collapse
    predicts whether the collapse marks a cyclical bottom (recovery) or a
    structural terminus (continued decline).

    Prediction rule:
        - RSSI > 0 → Cycle Bottom (recovery expected)
        - RSSI ≤ 0 → Structural Terminus

    Actual outcome determined by R_t improvement within POST_COLLAPSE_WINDOW.

    Parameters
    ----------
    panel : pd.DataFrame
        Full panel data (must contain 'R_t' column).
    rssi_dict : Dict[str, pd.Series]
        Sector‑level RSSI series.

    Returns
    -------
    Optional[Dict]
        Accuracy, number of samples, and binomial p‑value.
    """
    collapses = panel[panel["collapse_next"] == 1].copy()
    collapses = collapses[["Ticker", "Sector", "period_end"]].drop_duplicates()

    results = []
    for _, row in collapses.iterrows():
        ticker, sector, collapse_date = row["Ticker"], row["Sector"], row["period_end"]
        firm_data = panel[(panel["Ticker"] == ticker) &
                          (panel["period_end"] >= collapse_date)]
        firm_data = firm_data.sort_values("period_end").head(POST_COLLAPSE_WINDOW + 1)
        if len(firm_data) < 2:
            continue
        r_vals = firm_data["R_t"].dropna().values
        if len(r_vals) < 2:
            continue
        recovery = r_vals[-1] > r_vals[0] and r_vals[-1] > 0.1

        if sector not in rssi_dict:
            continue
        rssi_post = rssi_dict[sector].loc[
            collapse_date:collapse_date + pd.DateOffset(months=3 * POST_COLLAPSE_WINDOW)
        ]
        if len(rssi_post) == 0:
            continue

        rssi_sign = np.sign(rssi_post.mean())
        predicted = "Cycle Bottom" if rssi_sign > 0 else "Structural Terminus"
        actual = "Cycle Bottom" if recovery else "Structural Terminus"
        results.append({
            "Ticker": ticker,
            "Sector": sector,
            "collapse_date": collapse_date,
            "RSSI_sign": rssi_sign,
            "predicted": predicted,
            "actual": actual,
        })

    if not results:
        return None

    df_fc2 = pd.DataFrame(results)
    df_fc2.to_csv(TABLE_DIR / "T6_6F_FC2_results.csv", index=False)

    y_true = (df_fc2["actual"] == "Cycle Bottom").astype(int)
    y_pred = (df_fc2["predicted"] == "Cycle Bottom").astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    p_val = binomtest(np.sum(y_true == y_pred), n=len(y_true),
                      p=0.5, alternative="greater")

    metrics = {"accuracy": acc, "n_samples": len(y_true), "p_value": p_val}
    pd.DataFrame([metrics]).to_csv(TABLE_DIR / "T6_6F_FC2_metrics.csv", index=False)

    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Terminus", "Cycle Bottom"],
                yticklabels=["Terminus", "Cycle Bottom"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("FC2: RSSI Sign Post‑Collapse Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T6_FC2_confusion_matrix.png", dpi=150)
    plt.close()

    return metrics


# =============================================================================
# ACADEMIC SUMMARY REPORT GENERATION
# =============================================================================
def generate_academic_report(results: Dict) -> None:
    """
    Write a comprehensive, publication‑style summary report for Test 6.

    Parameters
    ----------
    results : Dict
        Dictionary containing outputs from all blocks.
    """
    report_path = REPORT_DIR / "T6_RSSI_Parabola_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("RSSI PARABOLA – SECTOR‑LEVEL TRAJECTORY VALIDATION (TEST 6)\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 100 + "\n\n")

        block_titles = [
            ("6A", "ALIGNED RSSI TRAJECTORY"),
            ("6B", "SHAPE TEST (QUADRATIC vs. LINEAR)"),
            ("6C", "PEAK‑TO‑COLLAPSE LAG DISTRIBUTION"),
            ("6D", "TWO‑CYCLE STRUCTURE DETECTION"),
            ("6E", "MIRROR TEST (RSSI vs. B)"),
            ("6F", "FC2 – CYCLE BOTTOM PREDICTION"),
        ]

        for block, title in block_titles:
            f.write(f"BLOCK {block}: {title}\n")
            f.write("-" * 60 + "\n")
            res = results.get(block)
            if res is not None:
                if isinstance(res, dict):
                    for k, v in res.items():
                        f.write(f"{k}: {v}\n")
                elif isinstance(res, pd.DataFrame):
                    f.write(res.to_string(index=False) + "\n")
                else:
                    f.write(str(res) + "\n")
            else:
                f.write("No data available.\n")
            f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute the full RSSI parabola validation pipeline."""
    print("=" * 80)
    print("RSSI PARABOLA ANALYSIS – ACADEMIC PIPELINE (TEST 6)")
    print("=" * 80)

    panel = load_full_panel()
    collapses = prepare_collapse_events(panel)
    print(f"Number of collapse events: {len(collapses):,}")

    # Load sector RSSI series
    rssi_dict: Dict[str, pd.Series] = {}
    for sector in SECTORS:
        file_path = PROCESSED_DIR / f"{sector}_RSSI_historical.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["period_end"])
            col = "RSSI_hist_winsor" if "RSSI_hist_winsor" in df.columns else "RSSI_hist"
            if col in df.columns:
                rssi_dict[sector] = df.set_index("period_end")[col].sort_index()

    results: Dict = {}

    print("\nBlock 6A: Aligned RSSI trajectory...")
    traj_df = block6a_aligned_trajectory(rssi_dict, collapses)
    results["6A"] = traj_df

    print("Block 6B: Shape test (quadratic vs. linear)...")
    if traj_df is not None:
        results["6B"] = block6b_shape_test(traj_df)

    print("Block 6C: Peak‑to‑collapse lag...")
    results["6C"] = block6c_peak_timing(rssi_dict, collapses)

    print("Block 6D: Two‑cycle structure...")
    results["6D"] = block6d_two_cycle_structure(rssi_dict)

    print("Block 6E: Mirror test (RSSI vs. B)...")
    if traj_df is not None:
        results["6E"] = block6e_mirror_test(panel, traj_df)

    print("Block 6F: FC2 prediction...")
    results["6F"] = block6f_fc2_prediction(panel, rssi_dict)

    print("\nGenerating academic summary report...")
    generate_academic_report(results)

    print("\nAll results saved to 'results/' directory.")


if __name__ == "__main__":
    main()