"""
Conditional Transition Analysis – Full Pipeline
-----------------------------------------------
This module examines how overvaluation (RSSI) influences the direction and
persistence of firm‑state transitions within the speculative lifecycle,
rather than directly predicting collapse events.

Key hypotheses:
    1. Intermediate RSSI (Mid) increases the probability of:
        - C3 → C4 deterioration (Block 3A)
        - Normal → C2 entry (Block 3B)
        - Self‑loop persistence (C2→C2, C3→C3) (Block 3C)
    2. Full transition matrices differ significantly between RSSI groups
       (Low, Mid, Extreme), with ΔP Mid‑Low and Mid‑Extreme quantified (Block 3D).
    3. Escape back to Normal within 8 quarters is less likely for Mid RSSI,
       indicating a barrier to recovery (Block 3E).

All results are exported to `results/tables/` and `results/figures/`, with a
comprehensive summary report in `results/reports/T3_Transition_Analysis_Report.txt`.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_full_panel() -> pd.DataFrame:
    """
    Load all firm‑quarter observations from classified CSV files.

    Ensures the presence of required columns and adds 'next_config' for
    transition analysis.

    Returns
    -------
    pd.DataFrame
        Panel data sorted by Ticker and period_end.
    """
    required_columns = [
        "Ticker", "period_end", "Configuration",
        "D_t", "B", "RSSI", "dRSSI_dt"
    ]
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
                    if "Configuration" in df.columns:
                        df = df.sort_values(["Ticker", "period_end"])
                        df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                        df["collapse_next"] = df["next_config"].isin(["C1", "C6"]).astype(int)
                        df.drop(columns=["next_config"], inplace=True)
                    else:
                        continue
                missing_cols = [c for c in required_columns if c not in df.columns]
                if missing_cols:
                    continue
                df = df[required_columns + ["Sector", "collapse_next"]].copy()
                all_dfs.append(df)
            except Exception:
                continue

    if not all_dfs:
        raise FileNotFoundError("No valid classified files were loaded.")

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel


def assign_rssi_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize RSSI into three groups based on quintiles:
        - Low    : quintiles 1‑2
        - Mid    : quintile 3
        - Extreme: quintiles 4‑5

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'RSSI' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns: RSSI_q5, RSSI_group.
    """
    df = df.copy()
    df["RSSI_q5"] = pd.qcut(df["RSSI"], 5, labels=False, duplicates="drop") + 1

    def group_label(q: float) -> str:
        if pd.isna(q):
            return np.nan
        if q <= 2:
            return "Low"
        elif q == 3:
            return "Mid"
        else:
            return "Extreme"

    df["RSSI_group"] = df["RSSI_q5"].apply(group_label)
    return df


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================
def bootstrap_ci(
    data: np.ndarray, n_bootstrap: int = 1000, alpha: float = 0.05
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the mean of a binary array.

    Parameters
    ----------
    data : np.ndarray
        Array of 0/1 values.
    n_bootstrap : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (two‑tailed).

    Returns
    -------
    Tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if len(data) == 0:
        return np.nan, np.nan
    means = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(sample))
    ci_low = np.percentile(means, 100 * alpha / 2)
    ci_high = np.percentile(means, 100 * (1 - alpha / 2))
    return ci_low, ci_high


def bootstrap_diff_ci(
    data1: np.ndarray,
    data2: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for the difference in means (data1 - data2).

    Parameters
    ----------
    data1, data2 : np.ndarray
        Binary arrays.
    n_bootstrap : int
        Number of resamples.
    alpha : float
        Significance level.

    Returns
    -------
    Tuple[float, float]
        CI bounds for the difference.
    """
    if len(data1) == 0 or len(data2) == 0:
        return np.nan, np.nan
    diffs = []
    n1, n2 = len(data1), len(data2)
    for _ in range(n_bootstrap):
        s1 = np.random.choice(data1, size=n1, replace=True)
        s2 = np.random.choice(data2, size=n2, replace=True)
        diffs.append(np.mean(s1) - np.mean(s2))
    ci_low = np.percentile(diffs, 100 * alpha / 2)
    ci_high = np.percentile(diffs, 100 * (1 - alpha / 2))
    return ci_low, ci_high


# =============================================================================
# BLOCK 3A – DIRECTION OF DETERIORATION (C3→C4, C4→C3)
# =============================================================================
def block3a_deterioration_direction(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transition probabilities C3→C4 and C4→C3 stratified by RSSI group.

    Returns
    -------
    pd.DataFrame
        Columns: From, To, RSSI_group, Rate, CI_low, CI_high, N.
    """
    df = assign_rssi_groups(panel.copy())
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
    df = df.dropna(subset=["next_config", "RSSI_group"])

    results = []
    # C3 → C4
    for group in ["Low", "Mid", "Extreme"]:
        sub = df[(df["Configuration"] == "C3") & (df["RSSI_group"] == group)]
        if len(sub) == 0:
            continue
        rate = (sub["next_config"] == "C4").mean()
        ci_low, ci_high = bootstrap_ci((sub["next_config"] == "C4").values)
        results.append({
            "From": "C3", "To": "C4", "RSSI_group": group,
            "Rate": rate, "CI_low": ci_low, "CI_high": ci_high, "N": len(sub)
        })
    # C4 → C3 (recovery)
    for group in ["Low", "Mid", "Extreme"]:
        sub = df[(df["Configuration"] == "C4") & (df["RSSI_group"] == group)]
        if len(sub) == 0:
            continue
        rate = (sub["next_config"] == "C3").mean()
        ci_low, ci_high = bootstrap_ci((sub["next_config"] == "C3").values)
        results.append({
            "From": "C4", "To": "C3", "RSSI_group": group,
            "Rate": rate, "CI_low": ci_low, "CI_high": ci_high, "N": len(sub)
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / "T3_3A_deterioration.csv", index=False)
    return df_res


# =============================================================================
# BLOCK 3B – ENTRY INTO CHIMERA (Normal → C2)
# =============================================================================
def block3b_entry_to_c2(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the probability of entering the Chimera state (Normal → C2)
    across RSSI groups.

    Returns
    -------
    pd.DataFrame
        Columns: RSSI_group, Rate, CI_low, CI_high, N.
    """
    df = assign_rssi_groups(panel.copy())
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
    df = df.dropna(subset=["next_config", "RSSI_group"])

    results = []
    for group in ["Low", "Mid", "Extreme"]:
        sub = df[(df["Configuration"] == "Normal") & (df["RSSI_group"] == group)]
        if len(sub) == 0:
            continue
        rate = (sub["next_config"] == "C2").mean()
        ci_low, ci_high = bootstrap_ci((sub["next_config"] == "C2").values)
        results.append({
            "RSSI_group": group, "Rate": rate,
            "CI_low": ci_low, "CI_high": ci_high, "N": len(sub)
        })

    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / "T3_3B_entry.csv", index=False)
    return df_res


# =============================================================================
# BLOCK 3C – SELF‑LOOP PERSISTENCE
# =============================================================================
def block3c_self_loop_persistence(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the probability that a firm remains in the same speculative state
    (C2→C2, C3→C3, C4→C4) across RSSI groups.

    Returns
    -------
    pd.DataFrame
        Columns: Configuration, RSSI_group, Rate, CI_low, CI_high, N.
    """
    df = assign_rssi_groups(panel.copy())
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
    df = df.dropna(subset=["next_config", "RSSI_group"])

    results = []
    for cfg in ["C2", "C3", "C4"]:
        for group in ["Low", "Mid", "Extreme"]:
            sub = df[(df["Configuration"] == cfg) & (df["RSSI_group"] == group)]
            if len(sub) == 0:
                continue
            rate = (sub["next_config"] == cfg).mean()
            ci_low, ci_high = bootstrap_ci((sub["next_config"] == cfg).values)
            results.append({
                "Configuration": cfg, "RSSI_group": group,
                "Rate": rate, "CI_low": ci_low, "CI_high": ci_high, "N": len(sub)
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / "T3_3C_self_loop.csv", index=False)
    return df_res


# =============================================================================
# BLOCK 3D – FULL TRANSITION MATRIX COMPARISON (ΔP)
# =============================================================================
def block3d_full_matrix_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct transition matrices for Low, Mid, and Extreme RSSI groups.
    Compute differences ΔP = P(Mid) - P(Low) and P(Mid) - P(Extreme) for each
    transition, with bootstrap confidence intervals.

    Saves heatmaps of each matrix and a bar chart of key transition differences.

    Returns
    -------
    pd.DataFrame
        Table of probability differences for all transitions.
    """
    df = assign_rssi_groups(panel.copy())
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
    df = df.dropna(subset=["next_config", "RSSI_group"])

    matrices = {}
    for group in ["Low", "Mid", "Extreme"]:
        sub = df[df["RSSI_group"] == group]
        if len(sub) == 0:
            matrices[group] = pd.DataFrame()
            continue
        trans = pd.crosstab(sub["Configuration"], sub["next_config"], normalize="index")
        matrices[group] = trans

    rows = []
    if "Mid" in matrices and not matrices["Mid"].empty:
        for from_cfg in matrices["Mid"].index:
            for to_cfg in matrices["Mid"].columns:
                rate_mid = matrices["Mid"].loc[from_cfg, to_cfg]
                # Low group comparison
                if from_cfg in matrices.get("Low", pd.DataFrame()).index and \
                   to_cfg in matrices["Low"].columns:
                    rate_low = matrices["Low"].loc[from_cfg, to_cfg]
                    sub_mid = df[(df["RSSI_group"] == "Mid") & (df["Configuration"] == from_cfg)]
                    sub_low = df[(df["RSSI_group"] == "Low") & (df["Configuration"] == from_cfg)]
                    if len(sub_mid) > 0 and len(sub_low) > 0:
                        ci_low_ml, ci_high_ml = bootstrap_diff_ci(
                            (sub_mid["next_config"] == to_cfg).values,
                            (sub_low["next_config"] == to_cfg).values
                        )
                    else:
                        ci_low_ml, ci_high_ml = np.nan, np.nan
                else:
                    rate_low = np.nan
                    ci_low_ml, ci_high_ml = np.nan, np.nan

                # Extreme group comparison
                if from_cfg in matrices.get("Extreme", pd.DataFrame()).index and \
                   to_cfg in matrices["Extreme"].columns:
                    rate_ext = matrices["Extreme"].loc[from_cfg, to_cfg]
                    sub_ext = df[(df["RSSI_group"] == "Extreme") & (df["Configuration"] == from_cfg)]
                    sub_mid = df[(df["RSSI_group"] == "Mid") & (df["Configuration"] == from_cfg)]
                    if len(sub_mid) > 0 and len(sub_ext) > 0:
                        ci_low_me, ci_high_me = bootstrap_diff_ci(
                            (sub_mid["next_config"] == to_cfg).values,
                            (sub_ext["next_config"] == to_cfg).values
                        )
                    else:
                        ci_low_me, ci_high_me = np.nan, np.nan
                else:
                    rate_ext = np.nan
                    ci_low_me, ci_high_me = np.nan, np.nan

                rows.append({
                    "From": from_cfg,
                    "To": to_cfg,
                    "Rate_Low": rate_low,
                    "Rate_Mid": rate_mid,
                    "Rate_Extreme": rate_ext,
                    "Delta_Mid_Low": rate_mid - rate_low,
                    "CI_low_Mid_Low": ci_low_ml,
                    "CI_high_Mid_Low": ci_high_ml,
                    "Delta_Mid_Extreme": rate_mid - rate_ext,
                    "CI_low_Mid_Extreme": ci_low_me,
                    "CI_high_Mid_Extreme": ci_high_me,
                })

    df_delta = pd.DataFrame(rows)
    df_delta.to_csv(TABLE_DIR / "T3_3D_delta_P.csv", index=False)

    # Save individual matrices
    for group, mat in matrices.items():
        if not mat.empty:
            mat.to_csv(TABLE_DIR / f"T3_3D_matrix_{group}.csv")

    # Heatmaps
    for group, mat in matrices.items():
        if mat.empty:
            continue
        plt.figure(figsize=(8, 6))
        sns.heatmap(mat, annot=True, fmt=".2%", cmap="Blues",
                    cbar_kws={"label": "Transition Probability"})
        plt.title(f"Transition Matrix – RSSI = {group}", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / f"T3_transition_matrix_{group}.png", dpi=150)
        plt.close()

    # Bar chart of key transition differences
    key_transitions = ["Normal→C2", "C2→C2", "C3→C4", "C4→C3", "C3→C3"]
    plot_data = df_delta[
        df_delta.apply(lambda r: f"{r['From']}→{r['To']}" in key_transitions, axis=1)
    ].copy()
    if not plot_data.empty:
        plot_data["Label"] = plot_data["From"] + "→" + plot_data["To"]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(plot_data))
        width = 0.35
        ax.bar(x - width/2, plot_data["Delta_Mid_Low"], width, label="Mid − Low")
        ax.bar(x + width/2, plot_data["Delta_Mid_Extreme"], width, label="Mid − Extreme")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data["Label"], rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend()
        ax.set_ylabel("Δ Probability", fontsize=12)
        ax.set_title("Transition Probability Differences by RSSI Group", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "T3_delta_P_chart.png", dpi=150)
        plt.close()

    return df_delta


# =============================================================================
# BLOCK 3E – ESCAPE BARRIER (RETURN TO NORMAL WITHIN 8 QUARTERS)
# =============================================================================
def block3e_escape_barrier(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the probability of returning to Normal within 8 quarters,
    conditional on current speculative state and RSSI group.

    Returns
    -------
    pd.DataFrame
        Columns: Config, RSSI_group, Escape_rate, CI_low, CI_high, N.
    """
    df = assign_rssi_groups(panel.copy())
    df = df.sort_values(["Ticker", "period_end"])
    df["escape_8Q"] = 0
    for lag in range(1, 9):
        col = f"cfg_t{lag}"
        df[col] = df.groupby("Ticker")["Configuration"].shift(-lag)
        df["escape_8Q"] = df["escape_8Q"] | (df[col] == "Normal")
    df["escape_8Q"] = df["escape_8Q"].astype(int)
    df = df[df["Configuration"].isin(["C2", "C3", "C4"])]

    results = []
    for cfg in ["C2", "C3", "C4", "All"]:
        for group in ["Low", "Mid", "Extreme"]:
            if cfg == "All":
                sub = df[df["RSSI_group"] == group]
            else:
                sub = df[(df["Configuration"] == cfg) & (df["RSSI_group"] == group)]
            if len(sub) == 0:
                continue
            rate = sub["escape_8Q"].mean()
            ci_low, ci_high = bootstrap_ci(sub["escape_8Q"].values)
            results.append({
                "Config": cfg,
                "RSSI_group": group,
                "Escape_rate": rate,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "N": len(sub)
            })

    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / "T3_3E_escape.csv", index=False)
    return df_res


# =============================================================================
# ACADEMIC SUMMARY REPORT GENERATION
# =============================================================================
def generate_academic_report(results: Dict[str, pd.DataFrame]) -> None:
    """
    Write a comprehensive, publication‑style summary report for Test 3.

    Parameters
    ----------
    results : Dict[str, pd.DataFrame]
        Dictionary containing results from all blocks (keys: '3A', '3B', '3C', '3D', '3E').
    """
    report_path = REPORT_DIR / "T3_Transition_Analysis_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("CONDITIONAL TRANSITION ANALYSIS – ACADEMIC SUMMARY REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 100 + "\n\n")
        f.write("Objective: RSSI does not directly forecast collapse; rather, it influences\n")
        f.write("the direction and persistence of transitions within the speculative lifecycle.\n")
        f.write("RSSI groups are defined by quintiles: Low (Q1‑2), Mid (Q3), Extreme (Q4‑5).\n\n")

        # Block 3A
        f.write("=" * 100 + "\n")
        f.write("1. DIRECTION OF DETERIORATION (C3→C4 and C4→C3) – Block 3A\n")
        f.write("=" * 100 + "\n")
        if "3A" in results and not results["3A"].empty:
            f.write(results["3A"].to_string(index=False))
            f.write("\n\nExpectation: Mid RSSI exhibits higher C3→C4 (deterioration) and lower C4→C3 (recovery).\n")
        else:
            f.write("No data available.\n")
        f.write("\n")

        # Block 3B
        f.write("=" * 100 + "\n")
        f.write("2. ENTRY INTO CHIMERA (Normal → C2) – Block 3B\n")
        f.write("=" * 100 + "\n")
        if "3B" in results and not results["3B"].empty:
            f.write(results["3B"].to_string(index=False))
            f.write("\n\nExpectation: Mid RSSI increases the probability of entering the speculative state.\n")
        else:
            f.write("No data.\n")
        f.write("\n")

        # Block 3C
        f.write("=" * 100 + "\n")
        f.write("3. SELF‑LOOP PERSISTENCE – Block 3C\n")
        f.write("=" * 100 + "\n")
        if "3C" in results and not results["3C"].empty:
            f.write(results["3C"].to_string(index=False))
            f.write("\n\nExpectation: Mid RSSI firms are more likely to remain in C2 and C3.\n")
        else:
            f.write("No data.\n")
        f.write("\n")

        # Block 3D
        f.write("=" * 100 + "\n")
        f.write("4. FULL TRANSITION MATRIX COMPARISON (ΔP) – Block 3D\n")
        f.write("=" * 100 + "\n")
        if "3D" in results and not results["3D"].empty:
            f.write("Probability differences: Mid − Low and Mid − Extreme.\n")
            f.write(results["3D"].to_string(index=False))
            f.write("\n\nKey transitions of interest: Normal→C2, C2→C2, C3→C4, C4→C3, C3→C3.\n")
            f.write("A confidence interval excluding zero indicates a statistically meaningful difference.\n")
        else:
            f.write("No data.\n")
        f.write("\n")

        # Block 3E
        f.write("=" * 100 + "\n")
        f.write("5. ESCAPE BARRIER (RETURN TO NORMAL WITHIN 8 QUARTERS) – Block 3E\n")
        f.write("=" * 100 + "\n")
        if "3E" in results and not results["3E"].empty:
            f.write(results["3E"].to_string(index=False))
            f.write("\n\nExpectation: Mid RSSI firms have a lower escape rate, indicating a barrier to recovery.\n")
        else:
            f.write("No data.\n")
        f.write("\n")

        f.write("=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute the full conditional transition analysis pipeline."""
    print("=" * 80)
    print("CONDITIONAL TRANSITION ANALYSIS – ACADEMIC PIPELINE")
    print("=" * 80)

    panel = load_full_panel()
    panel["next_config"] = panel.groupby("Ticker")["Configuration"].shift(-1)
    panel = panel.dropna(subset=["next_config"]).copy()
    print(f"Total observations (with next period): {len(panel):,}")

    results: Dict[str, pd.DataFrame] = {}

    print("\nBlock 3A: Direction of deterioration...")
    results["3A"] = block3a_deterioration_direction(panel)

    print("Block 3B: Entry into Chimera...")
    results["3B"] = block3b_entry_to_c2(panel)

    print("Block 3C: Self‑loop persistence...")
    results["3C"] = block3c_self_loop_persistence(panel)

    print("Block 3D: Full transition matrix comparison...")
    results["3D"] = block3d_full_matrix_comparison(panel)

    print("Block 3E: Escape barrier...")
    results["3E"] = block3e_escape_barrier(panel)

    print("\nGenerating academic summary report...")
    generate_academic_report(results)

    print("\nAll results saved to 'results/' directory.")


if __name__ == "__main__":
    main()