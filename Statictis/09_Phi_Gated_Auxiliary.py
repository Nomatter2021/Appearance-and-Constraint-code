"""
Phi‑Gated and Auxiliary Tests – Comprehensive Validation (Test 9)
------------------------------------------------------------------
This module consolidates several auxiliary empirical tests that complement
the core reflexivity framework by examining conditional effects, trajectory
dynamics, and robustness checks.

Sub‑tests included:
    T9  : CMH test for RSSI Mid effect on collapse_4Q, stratified by B,
          separately for active (Φ=1) and inactive (Φ=0) loop states.
    T10 : Mann‑Whitney U tests comparing collapse rates between RSSI phases
          (Mid vs. Low, Mid vs. Extreme) within each speculative configuration
          (C2, C3, C4).
    T11 : Analysis of dRSSI/dt (momentum of overvaluation) across C2
          trajectories leading to Collapse, Sustain, or Evolve.
    T12 : Comparison of RSSI levels at the moment of loop deactivation (Φ drop)
          between firms that subsequently collapse and those that survive.
    T13 : Post‑collapse restart probability (return to Normal or C2 within
          4 quarters) stratified by the sign of RSSI at collapse.
    T14 : Placebo (permutation) test to assess the statistical significance
          of the Φ=1 CMH result by shuffling Φ labels within firms.

All results are exported to `results/tables/` and `results/figures/`, with a
comprehensive academic report in `results/reports/T9_Phi_Gated_Auxiliary_Report.txt`.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, combine_pvalues, fisher_exact, mannwhitneyu

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
RANDOM_STATE = 42
N_PERMUTATIONS = 1000


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_full_panel() -> pd.DataFrame:
    """
    Load all firm‑quarter observations from classified CSV files.

    Ensures the presence of required columns:
        Ticker, period_end, Configuration, B, RSSI, Phi_t, dRSSI_dt,
        collapse_next. Sector and collapse_next are derived if missing.

    Returns
    -------
    pd.DataFrame
        Panel data sorted by Ticker and period_end.
    """
    required_columns = [
        "Ticker", "period_end", "Configuration", "B", "RSSI", "Phi_t",
        "dRSSI_dt", "collapse_next"
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
                    df = df.sort_values(["Ticker", "period_end"])
                    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                    df["collapse_next"] = df["next_config"].isin(["C1", "C6"]).astype(int)
                    df.drop(columns=["next_config"], inplace=True)
                missing = [c for c in required_columns if c not in df.columns]
                if missing:
                    continue
                df = df[required_columns + ["Sector"]].copy()
                all_dfs.append(df)
            except Exception:
                continue

    if not all_dfs:
        raise ValueError("No valid data files found.")

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel


def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features required for auxiliary tests.

    Creates:
        - collapse_4Q: binary collapse indicator within next 4 quarters.
        - RSSI_q: RSSI tertiles (Low, Mid, Extreme).
        - RSSI_Mid: binary indicator for Mid RSSI.
        - Phi_prev, Phi_drop: loop deactivation events.
        - B_q: B tertiles (Low, Mid, High).
        - next_config: for restart analysis.
    """
    df = panel.copy()
    df = df.sort_values(["Ticker", "period_end"])

    # 4‑quarter forward collapse
    df["collapse_4Q"] = 0
    for lag in range(1, 5):
        col = f"c_{lag}"
        df[col] = df.groupby("Ticker")["collapse_next"].shift(-lag)
        df["collapse_4Q"] = df["collapse_4Q"] | df[col].fillna(0).astype(int)
    df["collapse_4Q"] = df["collapse_4Q"].astype(int)

    # RSSI tertiles
    try:
        df["RSSI_q"] = pd.qcut(df["RSSI"], 3, labels=["Low", "Mid", "Extreme"], duplicates="drop")
    except ValueError:
        df["RSSI_q"] = "Mid"
    df["RSSI_Mid"] = (df["RSSI_q"] == "Mid").astype(int)

    # Φ deactivation
    df["Phi_prev"] = df.groupby("Ticker")["Phi_t"].shift(1)
    df["Phi_drop"] = ((df["Phi_prev"] == 1) & (df["Phi_t"] == 0)).astype(int)

    # B tertiles
    df["B_q"] = pd.qcut(df["B"], 3, labels=["Low", "Mid", "High"], duplicates="drop")

    # Next configuration for restart analysis
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)

    return df


def filter_speculative(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only observations in speculative states C2, C3, C4."""
    return df[df["Configuration"].isin(["C2", "C3", "C4"])].copy()


# =============================================================================
# T9: CMH TEST FOR RSSI MID EFFECT STRATIFIED BY B (BY Φ STATE)
# =============================================================================
def test9_phi_gated_cmh(df: pd.DataFrame) -> Dict:
    """
    Perform a Cochran–Mantel–Haenszel (CMH) style test for the association
    between RSSI_Mid and collapse_4Q, stratified by B tertiles, separately
    for active (Φ=1) and inactive (Φ=0) loop states.

    Uses Fisher's method to combine p‑values from individual 2×2 tables.
    Also computes the average odds ratio across strata.

    Parameters
    ----------
    df : pd.DataFrame
        Speculative panel with columns: Phi_t, B_q, RSSI_Mid, collapse_4Q.

    Returns
    -------
    Dict
        For each Φ state: combined_p, avg_odds_ratio, N_strata, N_total.
    """
    results = {}
    for phi_val, label in [(1, "Phi=1"), (0, "Phi=0")]:
        sub = df[df["Phi_t"] == phi_val].copy()
        if len(sub) < 20:
            results[label] = {"error": "Insufficient data"}
            continue

        p_values = []
        odds_ratios = []
        for b_level in ["Low", "Mid", "High"]:
            stratum = sub[sub["B_q"] == b_level]
            if len(stratum) < 5:
                continue
            table = pd.crosstab(stratum["RSSI_Mid"], stratum["collapse_4Q"])
            if table.shape == (2, 2):
                try:
                    _, p, _, _ = chi2_contingency(table.values)
                    p_values.append(p)
                    if (table.values[0, 0] > 0 and table.values[0, 1] > 0 and
                        table.values[1, 0] > 0 and table.values[1, 1] > 0):
                        or_val = (table.values[0, 0] * table.values[1, 1]) / \
                                 (table.values[0, 1] * table.values[1, 0])
                        odds_ratios.append(or_val)
                except Exception:
                    continue

        if not p_values:
            results[label] = {"error": "No valid strata"}
            continue

        _, combined_p = combine_pvalues(p_values, method="fisher")
        avg_or = np.exp(np.mean(np.log(odds_ratios))) if odds_ratios else np.nan

        results[label] = {
            "combined_p": combined_p,
            "avg_odds_ratio": avg_or,
            "N_strata": len(p_values),
            "N_total": len(sub),
        }

    return results


# =============================================================================
# T10: MANN‑WHITNEY U BY CONFIGURATION
# =============================================================================
def test10_configuration_mannwhitney(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare collapse_4Q rates between RSSI phases (Mid vs. Low, Mid vs. Extreme)
    within each speculative configuration (C2, C3, C4) using Mann‑Whitney U tests.

    Parameters
    ----------
    df : pd.DataFrame
        Speculative panel with Configuration, RSSI_q, collapse_4Q.

    Returns
    -------
    pd.DataFrame
        Test statistics and sample sizes per configuration.
    """
    configs = ["C2", "C3", "C4"]
    res_list = []
    for cfg in configs:
        sub = df[df["Configuration"] == cfg]
        if len(sub) < 10:
            continue
        mid = sub[sub["RSSI_q"] == "Mid"]["collapse_4Q"]
        low = sub[sub["RSSI_q"] == "Low"]["collapse_4Q"]
        ext = sub[sub["RSSI_q"] == "Extreme"]["collapse_4Q"]

        if len(mid) >= 5 and len(low) >= 5:
            _, p1 = mannwhitneyu(mid, low, alternative="two-sided")
        else:
            p1 = np.nan

        if len(mid) >= 5 and len(ext) >= 5:
            _, p2 = mannwhitneyu(mid, ext, alternative="two-sided")
        else:
            p2 = np.nan

        res_list.append({
            "Config": cfg,
            "N_mid": len(mid),
            "N_low": len(low),
            "N_ext": len(ext),
            "MW_p_Mid_vs_Low": p1,
            "MW_p_Mid_vs_Extreme": p2,
        })

    df_res = pd.DataFrame(res_list)
    df_res.to_csv(TABLE_DIR / "T10_config_mw.csv", index=False)
    return df_res


# =============================================================================
# T11: dRSSI/dt TRAJECTORIES IN C2
# =============================================================================
def test11_drssi_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse the derivative of RSSI (dRSSI/dt) for C2 observations, stratified
    by the subsequent trajectory: Collapse, Sustain, or Evolve.

    Trajectory is defined by the next observed configuration:
        - Collapse: next_config in ['C1', 'C6']
        - Sustain : next_config == 'C2'
        - Evolve  : next_config in ['C3', 'C4']

    Parameters
    ----------
    df : pd.DataFrame
        Speculative panel with Configuration, next_config, dRSSI_dt.

    Returns
    -------
    pd.DataFrame
        Summary statistics (mean, median, std, count) by trajectory.
    """
    c2 = df[df["Configuration"] == "C2"].copy()

    def assign_trajectory(row: pd.Series) -> Optional[str]:
        nxt = row["next_config"]
        if pd.isna(nxt):
            return np.nan
        if nxt in ["C1", "C6"]:
            return "Collapse"
        elif nxt in ["C3", "C4"]:
            return "Evolve"
        elif nxt == "C2":
            return "Sustain"
        return np.nan

    c2["Trajectory"] = c2.apply(assign_trajectory, axis=1)
    c2 = c2.dropna(subset=["Trajectory", "dRSSI_dt"])

    stats_df = c2.groupby("Trajectory")["dRSSI_dt"].agg(
        ["mean", "median", "std", "count"]
    ).reset_index()
    stats_df.to_csv(TABLE_DIR / "T11_drssi_trajectories.csv", index=False)

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=c2, x="Trajectory", y="dRSSI_dt",
                order=["Collapse", "Sustain", "Evolve"])
    plt.title("dRSSI/dt by C2 Trajectory", fontsize=14)
    plt.xlabel("Trajectory", fontsize=12)
    plt.ylabel("dRSSI/dt", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T11_drssi_boxplot.png", dpi=150)
    plt.close()

    return stats_df


# =============================================================================
# T12: RSSI AT PHI DROP AND SUBSEQUENT COLLAPSE
# =============================================================================
def test12_phi_drop_rssi(df: pd.DataFrame) -> Dict:
    """
    Compare RSSI levels at the moment of loop deactivation (Φ drops from 1 to 0)
    between firms that subsequently collapse within 4 quarters and those that
    survive.

    Parameters
    ----------
    df : pd.DataFrame
        Speculative panel with Phi_drop, RSSI, collapse_next.

    Returns
    -------
    Dict
        Counts, median RSSI values, and Mann‑Whitney U p‑value.
    """
    drops = df[df["Phi_drop"] == 1].copy()
    if len(drops) == 0:
        return {"error": "No Phi drop events"}

    drops["future_collapse"] = 0
    for lag in range(1, 5):
        drops[f"c_{lag}"] = drops.groupby("Ticker")["collapse_next"].shift(-lag)
        drops["future_collapse"] = drops["future_collapse"] | drops[f"c_{lag}"].fillna(0).astype(int)
    drops["future_collapse"] = drops["future_collapse"].astype(int)

    coll = drops[drops["future_collapse"] == 1]["RSSI"]
    surv = drops[drops["future_collapse"] == 0]["RSSI"]

    res = {
        "N_drop_events": len(drops),
        "N_collapse": len(coll),
        "N_survive": len(surv),
    }

    if len(coll) >= 3 and len(surv) >= 3:
        _, p = mannwhitneyu(coll, surv, alternative="two-sided")
        res["MW_p"] = p
        res["median_collapse"] = coll.median()
        res["median_survive"] = surv.median()
    else:
        res["MW_p"] = np.nan

    pd.DataFrame([res]).to_csv(TABLE_DIR / "T12_phi_drop_rssi.csv", index=False)
    return res


# =============================================================================
# T13: POST‑COLLAPSE RESTART BY RSSI SIGN
# =============================================================================
def test13_restart_by_rssi_sign(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Evaluate the probability of restarting (return to Normal or C2) within
    4 quarters after a collapse event, stratified by the sign of RSSI at the
    time of collapse.

    Parameters
    ----------
    df : pd.DataFrame
        Full panel with collapse_next, RSSI, Configuration.

    Returns
    -------
    Optional[pd.DataFrame]
        Restart rates by RSSI sign, with Fisher exact test p‑value.
    """
    collapses = df[df["collapse_next"] == 1].copy()
    if len(collapses) == 0:
        return None

    collapses["RSSI_sign"] = np.sign(collapses["RSSI"])
    restart_data = []

    for _, row in collapses.iterrows():
        ticker = row["Ticker"]
        date = row["period_end"]
        firm = df[(df["Ticker"] == ticker) & (df["period_end"] >= date)].sort_values("period_end")
        if len(firm) < 2:
            continue
        future_cfgs = firm["Configuration"].iloc[1:5].values
        restarted = any(c in ["Normal", "C2"] for c in future_cfgs)
        restart_data.append({
            "Ticker": ticker,
            "collapse_date": date,
            "RSSI_sign": row["RSSI_sign"],
            "restarted": int(restarted),
        })

    if not restart_data:
        return None

    df_restart = pd.DataFrame(restart_data)
    rates = df_restart.groupby("RSSI_sign")["restarted"].agg(
        ["mean", "count", "sum"]
    ).reset_index()
    rates.columns = ["RSSI_sign", "restart_rate", "N", "n_restart"]

    pos = df_restart[df_restart["RSSI_sign"] == 1]["restarted"]
    neg = df_restart[df_restart["RSSI_sign"] == -1]["restarted"]
    if len(pos) > 0 and len(neg) > 0:
        table = pd.crosstab(df_restart["RSSI_sign"], df_restart["restarted"])
        _, p_fisher = fisher_exact(table)
    else:
        p_fisher = np.nan

    rates["fisher_p"] = p_fisher
    rates.to_csv(TABLE_DIR / "T13_restart_by_rssi.csv", index=False)
    return rates


# =============================================================================
# T14: PLACEBO TEST – SHUFFLED PHI LABELS
# =============================================================================
def test14_placebo_phi(
    df: pd.DataFrame, n_perm: int = N_PERMUTATIONS
) -> Optional[Dict]:
    np.random.seed(RANDOM_STATE)
    """
    Perform a permutation‑based placebo test for the CMH result under Φ=1.

    The Φ_t labels are randomly shuffled within each firm, and the CMH test
    (T9) is re‑run on each permuted dataset. The proportion of permuted
    p‑values smaller than the original p‑value provides an empirical
    significance level.

    Parameters
    ----------
    df : pd.DataFrame
        Speculative panel.
    n_perm : int
        Number of permutations.

    Returns
    -------
    Optional[Dict]
        Original p‑value, placebo p‑value, and number of valid permutations.
    """
    original = test9_phi_gated_cmh(df)
    if "Phi=1" not in original or "combined_p" not in original["Phi=1"]:
        return None

    orig_p = original["Phi=1"]["combined_p"]
    null_dist = []

    for _ in range(n_perm):
        df_perm = df.copy()
        df_perm["Phi_t"] = df_perm.groupby("Ticker")["Phi_t"].transform(
            np.random.permutation
        )
        res_perm = test9_phi_gated_cmh(df_perm)
        if "Phi=1" in res_perm and "combined_p" in res_perm["Phi=1"]:
            null_dist.append(res_perm["Phi=1"]["combined_p"])

    if not null_dist:
        return None

    null_dist = np.array(null_dist)
    placebo_p = np.mean(null_dist <= orig_p)
    placebo_median = np.median(null_dist)

    res = {
        "original_p": orig_p,
        "placebo_p": placebo_p,
        "placebo_median_p": placebo_median,
        "n_perm": len(null_dist),
    }
    pd.DataFrame([res]).to_csv(TABLE_DIR / "T14_placebo.csv", index=False)
    return res


# =============================================================================
# ACADEMIC REPORT GENERATION
# =============================================================================
def generate_academic_report(results: Dict) -> None:
    """Write a comprehensive academic report for Test 9 (auxiliary tests)."""
    report_path = REPORT_DIR / "T9_Phi_Gated_Auxiliary_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("PHI‑GATED AND AUXILIARY TESTS – ACADEMIC SUMMARY REPORT (TEST 9)\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 100 + "\n\n")

        f.write("T9: CMH TEST FOR RSSI MID EFFECT STRATIFIED BY B (BY Φ STATE)\n")
        f.write("-" * 60 + "\n")
        f.write(str(results.get("T9", "Not run")) + "\n\n")

        f.write("T10: MANN‑WHITNEY U BY CONFIGURATION\n")
        f.write("-" * 60 + "\n")
        if results.get("T10") is not None:
            f.write(results["T10"].to_string(index=False) + "\n\n")
        else:
            f.write("Not run.\n\n")

        f.write("T11: dRSSI/dt ACROSS C2 TRAJECTORIES\n")
        f.write("-" * 60 + "\n")
        if results.get("T11") is not None:
            f.write(results["T11"].to_string(index=False) + "\n\n")
        else:
            f.write("Not run.\n\n")

        f.write("T12: RSSI AT PHI DROP AND SUBSEQUENT COLLAPSE\n")
        f.write("-" * 60 + "\n")
        f.write(str(results.get("T12", "Not run")) + "\n\n")

        f.write("T13: POST‑COLLAPSE RESTART BY RSSI SIGN\n")
        f.write("-" * 60 + "\n")
        if results.get("T13") is not None:
            f.write(results["T13"].to_string(index=False) + "\n\n")
        else:
            f.write("Not run.\n\n")

        f.write("T14: PLACEBO TEST (SHUFFLED PHI LABELS)\n")
        f.write("-" * 60 + "\n")
        f.write(str(results.get("T14", "Not run")) + "\n\n")

        f.write("=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute all auxiliary and phi‑gated tests."""
    print("=" * 80)
    print("PHI‑GATED AND AUXILIARY TESTS – ACADEMIC PIPELINE (TEST 9)")
    print("=" * 80)

    panel = load_full_panel()
    panel = prepare_panel(panel)
    df_spec = filter_speculative(panel)
    print(f"Speculative observations: {len(df_spec):,}")

    results: Dict = {}

    print("\nT9: CMH test stratified by B (by Φ state)...")
    results["T9"] = test9_phi_gated_cmh(df_spec)

    print("T10: Mann‑Whitney U by configuration...")
    results["T10"] = test10_configuration_mannwhitney(df_spec)

    print("T11: dRSSI/dt trajectories in C2...")
    results["T11"] = test11_drssi_trajectories(df_spec)

    print("T12: RSSI at Phi drop and subsequent collapse...")
    results["T12"] = test12_phi_drop_rssi(df_spec)

    print("T13: Post‑collapse restart by RSSI sign...")
    results["T13"] = test13_restart_by_rssi_sign(df_spec)

    print("T14: Placebo test with shuffled Phi...")
    results["T14"] = test14_placebo_phi(df_spec)

    print("\nGenerating academic report...")
    generate_academic_report(results)

    print("\nAll tests completed. Results saved to 'results/' directory.")


if __name__ == "__main__":
    main()