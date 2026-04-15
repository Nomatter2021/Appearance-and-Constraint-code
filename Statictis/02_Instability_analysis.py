"""
Instability Region Analysis – Full Pipeline
--------------------------------------------
This module examines the non‑linear relationship between firm‑level
overvaluation (RSSI) and subsequent collapse risk, with particular attention
to the moderating role of balance‑sheet strength (B).

Key hypotheses tested:
    1. RSSI effect is most pronounced when B is high (Block 2A, 2D).
    2. The joint distribution B × RSSI exhibits a peak at intermediate RSSI
       among high‑B firms (Block 2B).
    3. Firms with high RSSI are more likely to transition toward collapse
       states (Block 2C).
    4. The relationship between RSSI and collapse risk follows an inverted‑U
       shape, formally tested with a quadratic term (Block 2E).

All results are exported to `results/tables/` and `results/figures/`, with a
comprehensive summary report in `results/reports/T2_Instability_Report.txt`.

Input: Classified firm‑quarter observations from `data/classified/<Sector>/`.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# =============================================================================
# OPTIONAL DEPENDENCIES
# =============================================================================
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not installed. Inverted‑U test will report coefficients only (no p‑values).")

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

    Ensures the presence of required columns:
        - Ticker, period_end, Configuration, D_t, B, RSSI, dRSSI_dt
        - Sector (inferred from directory if missing)
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


def add_collapse_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create forward‑looking collapse indicators over 4‑quarter and 6‑quarter horizons.

    Collapse is defined as a transition to configuration C1 or C6 within the window.
    Observations with fewer than 7 subsequent quarters are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with 'Ticker', 'period_end', 'Configuration'.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional columns: collapse_4Q, collapse_6Q.
    """
    df = df.sort_values(["Ticker", "period_end"])
    df["collapse_4Q"] = 0
    for lag in range(1, 5):
        col = f"collapse_t{lag}"
        df[col] = df.groupby("Ticker")["Configuration"].shift(-lag).isin(["C1", "C6"]).astype(int)
        df["collapse_4Q"] = df["collapse_4Q"] | df[col]
    df["collapse_4Q"] = df["collapse_4Q"].astype(int)

    df["collapse_6Q"] = df["collapse_4Q"]
    for lag in range(5, 7):
        col = f"collapse_t{lag}"
        df[col] = df.groupby("Ticker")["Configuration"].shift(-lag).isin(["C1", "C6"]).astype(int)
        df["collapse_6Q"] = df["collapse_6Q"] | df[col]
    df["collapse_6Q"] = df["collapse_6Q"].astype(int)

    min_periods = df.groupby("Ticker")["period_end"].transform("count")
    df = df[min_periods >= 7].copy()
    return df


# =============================================================================
# FEATURE ENGINEERING: DISCRETISATION
# =============================================================================
def assign_quantile_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign quintile‑based categorical levels to RSSI and B.

    Levels:
        - RSSI_q5, B_q5 : 1 (lowest) to 5 (highest)
        - RSSI_level, B_level : 'Low' (q1‑2), 'Mid' (q3), 'High' (q4‑5)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'RSSI' and 'B' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with added discretised columns.
    """
    df = df.copy()
    df["RSSI_q5"] = pd.qcut(df["RSSI"], 5, labels=False, duplicates="drop") + 1
    df["B_q5"] = pd.qcut(df["B"], 5, labels=False, duplicates="drop") + 1

    def to_level(q: float) -> str:
        if pd.isna(q):
            return np.nan
        if q <= 2:
            return "Low"
        elif q == 3:
            return "Mid"
        else:
            return "High"

    df["RSSI_level"] = df["RSSI_q5"].apply(to_level)
    df["B_level"] = df["B_q5"].apply(to_level)
    return df


# =============================================================================
# BLOCK 2A – RSSI EFFECT CONDITIONAL ON HIGH B
# =============================================================================
def block2a_rssi_controlled_by_b(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute collapse rates across RSSI levels, restricting to firms with high B.

    Outputs a bar plot with 95% confidence intervals.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel data limited to speculative configurations (C2, C3, C4).

    Returns
    -------
    pd.DataFrame
        Summary table with columns: RSSI_level, collapse_rate, N, n_collapse,
        ci_low, ci_high.
    """
    df = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df = assign_quantile_levels(df)
    df = df.dropna(subset=["RSSI_level", "B_level"])
    df_high_b = df[df["B_level"] == "High"]

    if len(df_high_b) == 0:
        return pd.DataFrame()

    rates = df_high_b.groupby("RSSI_level")["collapse_4Q"].agg(["mean", "count", "sum"]).reset_index()
    rates.columns = ["RSSI_level", "collapse_rate", "N", "n_collapse"]
    se = np.sqrt(rates["collapse_rate"] * (1 - rates["collapse_rate"]) / rates["N"])
    rates["ci_low"] = (rates["collapse_rate"] - 1.96 * se).clip(0, 1)
    rates["ci_high"] = (rates["collapse_rate"] + 1.96 * se).clip(0, 1)

    rates.to_csv(TABLE_DIR / "T2_block2A_RSSI_controlled_by_B.csv", index=False)

    if not rates.empty:
        order = ["Low", "Mid", "High"]
        rates_plot = rates.set_index("RSSI_level").reindex(order).dropna()
        yerr = [
            rates_plot["collapse_rate"] - rates_plot["ci_low"],
            rates_plot["ci_high"] - rates_plot["collapse_rate"]
        ]
        plt.figure(figsize=(8, 5))
        plt.bar(rates_plot.index, rates_plot["collapse_rate"], yerr=yerr, capsize=5)
        plt.xlabel("RSSI Level (Conditional on High B)", fontsize=12)
        plt.ylabel("4‑Quarter Collapse Rate", fontsize=12)
        plt.title("Collapse Rate by RSSI Level (High B Only)", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "T2_RSSI_controlled_by_B.png", dpi=150)
        plt.close()

    return rates


# =============================================================================
# BLOCK 2B – JOINT MATRIX B × RSSI
# =============================================================================
def block2b_joint_matrix_b_rssi(panel: pd.DataFrame) -> Dict:
    """
    Construct a 5×5 heatmap of collapse rates across B and RSSI quintiles.

    Also extracts collapse rates for the highest B quintile (B5) across RSSI
    levels: low (R1), mid (R3), and extreme (R5).

    Parameters
    ----------
    panel : pd.DataFrame
        Speculative panel (C2, C3, C4).

    Returns
    -------
    Dict
        Keys: 'matrix' (pivot table), 'B5_R1', 'B5_R3', 'B5_R5'.
    """
    df = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    if len(df) == 0:
        return {}

    df = assign_quantile_levels(df)
    df = df.dropna(subset=["B_q5", "RSSI_q5"])

    matrix = df.groupby(["B_q5", "RSSI_q5"])["collapse_4Q"].agg(["mean", "count"]).reset_index()
    pivot_mean = matrix.pivot(index="B_q5", columns="RSSI_q5", values="mean")
    pivot_count = matrix.pivot(index="B_q5", columns="RSSI_q5", values="count")

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_mean, annot=True, fmt=".2%", cmap="Reds", cbar_kws={"label": "Collapse Rate"})
    plt.xlabel("RSSI Quintile", fontsize=12)
    plt.ylabel("B Quintile", fontsize=12)
    plt.title("Collapse Rate: Joint B × RSSI Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T2_heatmap_B_RSSI.png", dpi=150)
    plt.close()

    # Extract values for B5 (highest B quintile)
    b5_subset = df[df["B_q5"] == 5]
    if len(b5_subset) > 0:
        rates_by_rssi = b5_subset.groupby("RSSI_q5")["collapse_4Q"].mean()
        rate_b5_r1 = rates_by_rssi.get(1, np.nan)
        rate_b5_r3 = rates_by_rssi.get(3, np.nan)
        rate_b5_r5 = rates_by_rssi.get(5, np.nan)
    else:
        rate_b5_r1 = rate_b5_r3 = rate_b5_r5 = np.nan

    pivot_mean.to_csv(TABLE_DIR / "T2_joint_matrix.csv")
    pivot_count.to_csv(TABLE_DIR / "T2_joint_matrix_counts.csv")

    return {
        "matrix": pivot_mean,
        "B5_R1": rate_b5_r1,
        "B5_R3": rate_b5_r3,
        "B5_R5": rate_b5_r5,
    }


# =============================================================================
# BLOCK 2C – MARKOV TRANSITION PROBABILITIES BY RSSI LEVEL
# =============================================================================
def block2c_markov_direction(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one‑quarter transition probabilities from C2/C3/C4 to collapse
    (C1/C6) versus sustained speculative states (C3/C4), stratified by RSSI level.

    Parameters
    ----------
    panel : pd.DataFrame
        Speculative panel.

    Returns
    -------
    pd.DataFrame
        Columns: RSSI_level, From, P_collapse, P_sustain, N_from.
    """
    df = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df = assign_quantile_levels(df)
    df = df.dropna(subset=["RSSI_level"])
    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
    df = df.dropna(subset=["next_config"])

    results = []
    for level in ["Low", "Mid", "High"]:
        sub = df[df["RSSI_level"] == level]
        if len(sub) == 0:
            continue
        transition_table = pd.crosstab(sub["Configuration"], sub["next_config"], normalize="index")
        for from_cfg in ["C2", "C3", "C4"]:
            if from_cfg in transition_table.index:
                p_collapse = sum(transition_table.loc[from_cfg].get(c, 0.0) for c in ["C1", "C6"])
                p_sustain = sum(transition_table.loc[from_cfg].get(c, 0.0) for c in ["C3", "C4"])
                n_from = len(sub[sub["Configuration"] == from_cfg])
                results.append({
                    "RSSI_level": level,
                    "From": from_cfg,
                    "P_collapse": p_collapse,
                    "P_sustain": p_sustain,
                    "N_from": n_from,
                })

    trans_df = pd.DataFrame(results)
    if not trans_df.empty:
        trans_df.to_csv(TABLE_DIR / "T2_markov_direction.csv", index=False)
    return trans_df


# =============================================================================
# BLOCK 2D – WINDOW COLLAPSE RATES (HIGH B × RSSI)
# =============================================================================
def block2d_window_by_b_rssi(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 4‑quarter and 6‑quarter collapse rates for high‑B firms across
    RSSI levels.

    Parameters
    ----------
    panel : pd.DataFrame
        Speculative panel.

    Returns
    -------
    pd.DataFrame
        Columns: RSSI_level, N, rate_4Q, rate_6Q.
    """
    df = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    if len(df) == 0:
        return pd.DataFrame()

    df = assign_quantile_levels(df)
    df = df.dropna(subset=["B_level", "RSSI_level"])
    df_high_b = df[df["B_level"] == "High"]
    if len(df_high_b) == 0:
        return pd.DataFrame()

    def compute_window_rates(sub: pd.DataFrame) -> pd.Series:
        n = len(sub)
        rate_4q = sub["collapse_4Q"].mean() if n > 0 else np.nan
        rate_6q = sub["collapse_6Q"].mean() if n > 0 else np.nan
        return pd.Series({"N": n, "rate_4Q": rate_4q, "rate_6Q": rate_6q})

    rates = df_high_b.groupby("RSSI_level").apply(compute_window_rates).reset_index()
    rates.to_csv(TABLE_DIR / "T2_window_highB_by_RSSI.csv", index=False)

    if not rates.empty:
        order = ["Low", "Mid", "High"]
        rates_plot = rates.set_index("RSSI_level").reindex(order).dropna()
        plt.figure(figsize=(8, 5))
        plt.bar(rates_plot.index, rates_plot["rate_4Q"], color="steelblue")
        plt.xlabel("RSSI Level (High B)", fontsize=12)
        plt.ylabel("4‑Quarter Collapse Rate", fontsize=12)
        plt.title("Window Collapse Rate: High B × RSSI", fontsize=14)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "T2_window_highB_RSSI.png", dpi=150)
        plt.close()

    return rates


# =============================================================================
# BLOCK 2E – FORMAL TEST OF INVERTED‑U SHAPE (QUADRATIC RSSI)
# =============================================================================
def block2e_inverted_u_test(panel: pd.DataFrame) -> Optional[Dict]:
    """
    Perform a logistic regression of collapse_4Q on B, RSSI, RSSI², and the
    interaction B × RSSI. A negative and significant coefficient on RSSI²
    supports the inverted‑U hypothesis.

    Uses statsmodels if available; otherwise falls back to sklearn (no p‑values).

    Parameters
    ----------
    panel : pd.DataFrame
        Speculative panel.

    Returns
    -------
    Optional[Dict]
        Dictionary with coefficient estimates, p‑values, pseudo R², and N.
    """
    df = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    if len(df) == 0:
        return None
    df = df.dropna(subset=["B", "RSSI", "collapse_4Q"])
    if len(df) == 0:
        return None

    df["RSSI_sq"] = df["RSSI"] ** 2
    df["B_x_RSSI"] = df["B"] * df["RSSI"]

    X = df[["B", "RSSI", "RSSI_sq", "B_x_RSSI"]]
    y = df["collapse_4Q"]

    if STATSMODELS_AVAILABLE:
        X = sm.add_constant(X)
        try:
            model = sm.Logit(y, X).fit(disp=0)
            coef = model.params
            pvals = model.pvalues
            results = {
                "coef_B": coef.get("B", np.nan),
                "p_B": pvals.get("B", np.nan),
                "coef_RSSI": coef.get("RSSI", np.nan),
                "p_RSSI": pvals.get("RSSI", np.nan),
                "coef_RSSI_sq": coef.get("RSSI_sq", np.nan),
                "p_RSSI_sq": pvals.get("RSSI_sq", np.nan),
                "coef_BxRSSI": coef.get("B_x_RSSI", np.nan),
                "p_BxRSSI": pvals.get("B_x_RSSI", np.nan),
                "pseudo_r2": model.prsquared,
                "n_obs": len(df),
            }
        except Exception as e:
            results = {"error": str(e)}
    else:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        coef = model.coef_[0]
        results = {
            "coef_B": coef[0],
            "coef_RSSI": coef[1],
            "coef_RSSI_sq": coef[2],
            "coef_BxRSSI": coef[3],
            "p_B": np.nan,
            "p_RSSI": np.nan,
            "p_RSSI_sq": np.nan,
            "p_BxRSSI": np.nan,
            "pseudo_r2": np.nan,
            "n_obs": len(df),
            "warning": "statsmodels not available, p‑values not computed",
        }

    pd.DataFrame([results]).to_csv(TABLE_DIR / "T2_inverted_U_logit.csv", index=False)
    return results


# =============================================================================
# ACADEMIC SUMMARY REPORT GENERATION
# =============================================================================
def generate_academic_report(panel: pd.DataFrame, results: Dict) -> None:
    """
    Write a comprehensive summary report for Test 2.

    Parameters
    ----------
    panel : pd.DataFrame
        Speculative panel used for the analysis.
    results : Dict
        Dictionary containing outputs from all blocks.
    """
    report_path = REPORT_DIR / "T2_Instability_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("INSTABILITY REGION ANALYSIS – ACADEMIC SUMMARY REPORT\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total speculative observations (C2/C3/C4): {len(panel):,}\n\n")

        # Block 2A
        f.write("1. RSSI EFFECT CONDITIONAL ON HIGH B (Block 2A)\n")
        f.write("-" * 50 + "\n")
        res_a = results.get("A")
        if res_a is not None and not res_a.empty:
            f.write(res_a.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No data available.\n\n")

        # Block 2B
        f.write("2. JOINT MATRIX B × RSSI (Block 2B)\n")
        f.write("-" * 50 + "\n")
        res_b = results.get("B", {})
        if "matrix" in res_b and not res_b["matrix"].empty:
            f.write(res_b["matrix"].to_string())
            f.write(f"\n\nHighest B quintile (B5) collapse rates:\n")
            f.write(f"  Low RSSI (R1)   : {res_b.get('B5_R1', np.nan):.3%}\n")
            f.write(f"  Mid RSSI (R3)   : {res_b.get('B5_R3', np.nan):.3%}\n")
            f.write(f"  High RSSI (R5)  : {res_b.get('B5_R5', np.nan):.3%}\n")
            f.write("(Interpretation: peak risk at intermediate RSSI when B is high)\n\n")
        else:
            f.write("No data.\n\n")

        # Block 2C
        f.write("3. MARKOV TRANSITION PROBABILITIES BY RSSI LEVEL (Block 2C)\n")
        f.write("-" * 50 + "\n")
        res_c = results.get("C", pd.DataFrame())
        if not res_c.empty:
            f.write(res_c.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No data.\n\n")

        # Block 2D
        f.write("4. WINDOW COLLAPSE RATES (HIGH B, 4Q & 6Q) (Block 2D)\n")
        f.write("-" * 50 + "\n")
        res_d = results.get("D", pd.DataFrame())
        if not res_d.empty:
            f.write(res_d.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("No data.\n\n")

        # Block 2E
        f.write("5. FORMAL TEST OF INVERTED‑U SHAPE (QUADRATIC RSSI) (Block 2E)\n")
        f.write("-" * 50 + "\n")
        res_e = results.get("E", {})
        if "error" in res_e:
            f.write(f"Model estimation failed: {res_e['error']}\n")
        elif res_e:
            f.write(f"Observations          : {res_e.get('n_obs', 'N/A')}\n")
            f.write(f"Pseudo R²             : {res_e.get('pseudo_r2', np.nan):.4f}\n\n")
            f.write("Coefficient estimates:\n")
            f.write(f"  B                 : {res_e.get('coef_B', np.nan):.4f} (p = {res_e.get('p_B', np.nan):.4f})\n")
            f.write(f"  RSSI              : {res_e.get('coef_RSSI', np.nan):.4f} (p = {res_e.get('p_RSSI', np.nan):.4f})\n")
            f.write(f"  RSSI²             : {res_e.get('coef_RSSI_sq', np.nan):.4f} (p = {res_e.get('p_RSSI_sq', np.nan):.4f})\n")
            f.write(f"  B × RSSI          : {res_e.get('coef_BxRSSI', np.nan):.4f} (p = {res_e.get('p_BxRSSI', np.nan):.4f})\n")
            if "warning" in res_e:
                f.write(f"  NOTE: {res_e['warning']}\n")
            if res_e.get("coef_RSSI_sq", 0) < 0 and res_e.get("p_RSSI_sq", 1) < 0.05:
                f.write("\nInterpretation: The negative and significant RSSI² coefficient supports\n")
                f.write("the inverted‑U hypothesis — collapse risk rises then falls with RSSI.\n")
            else:
                f.write("\nInterpretation: No statistically significant inverted‑U shape detected.\n")
        else:
            f.write("No data.\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of report.\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute the full instability region analysis pipeline."""
    print("=" * 70)
    print("INSTABILITY REGION ANALYSIS – ACADEMIC PIPELINE")
    print("=" * 70)

    panel = load_full_panel()
    panel = add_collapse_windows(panel)
    panel_spec = panel[panel["Configuration"].isin(["C2", "C3", "C4"])].copy()
    print(f"Speculative observations (C2/C3/C4): {len(panel_spec):,}")

    if len(panel_spec) == 0:
        print("No speculative observations. Exiting.")
        return

    results: Dict = {}

    print("\nBlock 2A: RSSI effect conditional on High B...")
    results["A"] = block2a_rssi_controlled_by_b(panel_spec)

    print("Block 2B: Joint matrix B × RSSI...")
    results["B"] = block2b_joint_matrix_b_rssi(panel_spec)

    print("Block 2C: Markov transition direction by RSSI level...")
    results["C"] = block2c_markov_direction(panel_spec)

    print("Block 2D: Window collapse rates (High B)...")
    results["D"] = block2d_window_by_b_rssi(panel_spec)

    print("Block 2E: Formal test of inverted‑U shape (quadratic RSSI)...")
    results["E"] = block2e_inverted_u_test(panel_spec)

    print("\nGenerating academic report...")
    generate_academic_report(panel_spec, results)

    print("\nAll results saved to 'results/' directory.")


if __name__ == "__main__":
    main()