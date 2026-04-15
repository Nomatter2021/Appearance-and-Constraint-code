"""
Cluster Separation Analysis – Joint Framework Validation (Test 5)
-----------------------------------------------------------------
This module validates the joint B–RSSI framework by examining whether adding
RSSI as a third dimension improves the separation of C2 firm outcomes
(Collapse, Evolve, Sustain) in the feature space.

Key analyses:
    5A. Baseline 2D separation using B and B_Acceleration (winsorized).
    5B. 3D separation with RSSI added; includes a permutation test for
        statistical significance.
    5C. Comparison of alternative 3D spaces from prior literature.
    5D. Trajectory analysis via median centroids of each outcome group.
    5E. Cluster separation stratified by RSSI phase (Low, Mid, Extreme).
    5F. Random Forest feature importance across the three dimensions.
    5G. Conditional collapse probability heatmaps for each RSSI phase.

All variables except RSSI are winsorized at 1% and 99% to mitigate outlier
influence. Features are standardized prior to clustering metric computation.

Results are exported to `results/tables/` and `results/figures/`, with a
comprehensive summary report in `results/reports/T5_Cluster_Separation_Report.txt`.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (used for 3D projection)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

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

# Winsorization quantiles
WINSOR_LOWER = 0.01
WINSOR_UPPER = 0.99


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_and_prepare_c2_data() -> pd.DataFrame:
    """
    Load classified data, compute kinematics, winsorize, and retain only C2
    observations with a valid forward‑looking outcome.

    Outcome categories:
        - Collapse : next_config in ['C1', 'C6']
        - Evolve   : next_config in ['C3', 'C4']
        - Sustain  : next_config == 'C2'

    Winsorization is applied to B and B_Acceleration at 1% / 99%.
    RSSI is assumed already winsorized from prior processing.

    Returns
    -------
    pd.DataFrame
        Cleaned C2 dataset with columns: B_winsor, B_Acceleration_winsor,
        RSSI_winsor, outcome, collapse_binary, and metadata.
    """
    required_columns = ["Ticker", "period_end", "Configuration", "B", "RSSI"]
    all_dfs: List[pd.DataFrame] = []

    for sector in SECTORS:
        sector_path = CLASSIFIED_DIR / sector
        if not sector_path.exists():
            print(f"  WARNING: Sector directory not found: {sector_path}")
            continue
        files = list(sector_path.glob("*_classified.csv"))
        print(f"  {sector}: {len(files)} files")
        for file_path in files:
            try:
                df = pd.read_csv(file_path, parse_dates=["period_end"])
                if "next_config" not in df.columns:
                    df = df.sort_values(["Ticker", "period_end"])
                    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                missing = [c for c in required_columns if c not in df.columns]
                if missing:
                    continue
                df = df[required_columns + ["next_config"]].copy()
                df["Sector"] = sector
                all_dfs.append(df)
            except Exception:
                continue

    if not all_dfs:
        raise ValueError("No valid C2 data could be loaded.")

    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    print(f"  Total raw observations: {len(panel):,}")

    # Compute kinematics
    panel["B_lag1"] = panel.groupby("Ticker")["B"].shift(1)
    panel["B_Change"] = panel["B"] - panel["B_lag1"]
    panel["B_Change_lag1"] = panel.groupby("Ticker")["B_Change"].shift(1)
    panel["B_Acceleration"] = panel["B_Change"] - panel["B_Change_lag1"]

    # Retain only C2
    c2 = panel[panel["Configuration"] == "C2"].copy()
    print(f"  C2 observations: {len(c2):,}")

    # Define outcome
    def assign_outcome(row: pd.Series) -> str:
        next_cfg = row["next_config"]
        if pd.isna(next_cfg):
            return np.nan
        if next_cfg in ["C1", "C6"]:
            return "Collapse"
        elif next_cfg in ["C3", "C4"]:
            return "Evolve"
        elif next_cfg == "C2":
            return "Sustain"
        return np.nan

    c2["outcome"] = c2.apply(assign_outcome, axis=1)
    c2 = c2.dropna(subset=["outcome"]).copy()
    c2["collapse_binary"] = (c2["outcome"] == "Collapse").astype(int)

    # Drop rows with missing kinematics or RSSI
    c2 = c2.dropna(subset=["B", "B_Acceleration", "RSSI"]).copy()

    # Winsorization
    b_lower, b_upper = c2["B"].quantile(WINSOR_LOWER), c2["B"].quantile(WINSOR_UPPER)
    c2["B_winsor"] = c2["B"].clip(lower=b_lower, upper=b_upper)

    bacc_lower = c2["B_Acceleration"].quantile(WINSOR_LOWER)
    bacc_upper = c2["B_Acceleration"].quantile(WINSOR_UPPER)
    c2["B_Acceleration_winsor"] = c2["B_Acceleration"].clip(lower=bacc_lower, upper=bacc_upper)

    c2["RSSI_winsor"] = c2["RSSI"]  # RSSI is assumed already winsorized upstream

    print(f"  Final C2 sample after winsorization: {len(c2):,}")
    return c2


# =============================================================================
# FEATURE SCALING AND CLUSTERING METRICS
# =============================================================================
def get_scaled_features(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """
    Standardize selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Data source.
    columns : List[str]
        Column names to scale.

    Returns
    -------
    np.ndarray
        Scaled feature matrix.
    """
    scaler = StandardScaler()
    X = df[columns].values
    return scaler.fit_transform(X)


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Compute Silhouette and Davies‑Bouldin scores.

    Returns (np.nan, np.nan) if fewer than two unique labels exist.
    """
    if len(np.unique(labels)) < 2:
        return np.nan, np.nan
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return sil, db


def permutation_test_rssi(
    df: pd.DataFrame, y: np.ndarray, n_permutations: int = 1000
) -> Tuple[float, float]:
    np.random.seed(42)
    """
    Perform a permutation test on the Silhouette score to assess whether
    adding RSSI significantly improves separation over the 2D baseline.

    The baseline features (B_winsor, B_Acceleration_winsor) are held fixed,
    while RSSI is randomly permuted.

    Returns
    -------
    Tuple[float, float]
        (true_silhouette, p_value)
    """
    X_base = df[["B_winsor", "B_Acceleration_winsor"]].values
    rssi = df["RSSI_winsor"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X_true = np.hstack([X_base, rssi])
    X_true = scaler.fit_transform(X_true)
    true_sil, _ = cluster_metrics(X_true, y)

    perm_sils = []
    for _ in range(n_permutations):
        rssi_perm = np.random.permutation(rssi)
        X_perm = np.hstack([X_base, rssi_perm])
        X_perm = scaler.fit_transform(X_perm)
        sil_perm, _ = cluster_metrics(X_perm, y)
        perm_sils.append(sil_perm)

    p_val = np.mean(np.array(perm_sils) >= true_sil)
    return true_sil, p_val


# =============================================================================
# BLOCK 5A – BASELINE 2D SEPARATION
# =============================================================================
def block5a_baseline_2d(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate cluster separation in the 2D space (B, B_Acceleration).

    Returns
    -------
    Dict[str, float]
        Silhouette and Davies‑Bouldin scores.
    """
    X = get_scaled_features(df, ["B_winsor", "B_Acceleration_winsor"])
    y = df["outcome"].values
    sil, db = cluster_metrics(X, y)

    # Scatter plot
    plt.figure(figsize=(8, 6))
    colors = {"Collapse": "red", "Evolve": "orange", "Sustain": "blue"}
    for label, color in colors.items():
        mask = df["outcome"] == label
        plt.scatter(
            df.loc[mask, "B_winsor"],
            df.loc[mask, "B_Acceleration_winsor"],
            c=color, label=label, alpha=0.6, s=30
        )
    plt.xlabel("B (winsorized)", fontsize=12)
    plt.ylabel("B Acceleration (winsorized)", fontsize=12)
    plt.title("2D Baseline Separation (Winsorized)", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T5_scatter_2D_winsor.png", dpi=150)
    plt.close()

    return {"Silhouette_2D": sil, "Davies_Bouldin_2D": db}


# =============================================================================
# BLOCK 5B – 3D SEPARATION WITH RSSI
# =============================================================================
def block5b_3d_with_rssi(df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate cluster separation in 3D space (B, B_Acceleration, RSSI) and
    perform permutation test for significance.

    Returns
    -------
    Dict[str, float]
        Silhouette, Davies‑Bouldin, and permutation p‑value.
    """
    X = get_scaled_features(df, ["B_winsor", "B_Acceleration_winsor", "RSSI_winsor"])
    y = df["outcome"].values
    sil, db = cluster_metrics(X, y)
    true_sil, p_val = permutation_test_rssi(df, y)

    # 3D scatter plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"Collapse": "red", "Evolve": "orange", "Sustain": "blue"}
    for label, color in colors.items():
        mask = df["outcome"] == label
        ax.scatter(
            df.loc[mask, "B_winsor"],
            df.loc[mask, "B_Acceleration_winsor"],
            df.loc[mask, "RSSI_winsor"],
            c=color, label=label, alpha=0.5, s=20
        )
    ax.set_xlabel("B (winsorized)", fontsize=10)
    ax.set_ylabel("B Acceleration (winsorized)", fontsize=10)
    ax.set_zlabel("RSSI", fontsize=10)
    ax.set_title("3D Separation with RSSI (Winsorized)", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T5_scatter_3D_winsor.png", dpi=150)
    plt.close()

    return {"Silhouette_3D": sil, "Davies_Bouldin_3D": db, "Permutation_p": p_val}


# =============================================================================
# BLOCK 5C – COMPARISON OF ALTERNATIVE 3D SPACES
# =============================================================================
def block5c_alternative_spaces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare cluster separation metrics for three alternative 3D spaces:
        1. (B, B_Change, B_Acceleration) – Paper 1
        2. (B, B_Change, RSSI)            – Paper 2
        3. (B, B_Acceleration, RSSI)      – Joint framework

    Returns
    -------
    pd.DataFrame
        Columns: Space, Silhouette, Davies_Bouldin.
    """
    scaler = StandardScaler()
    results: List[Tuple[str, float, float]] = []

    # Space 1
    sub1 = df[["B_winsor", "B_Change", "B_Acceleration_winsor"]].dropna()
    if len(sub1) > 10:
        X1 = scaler.fit_transform(sub1)
        y1 = df.loc[sub1.index, "outcome"].values
        sil1, db1 = cluster_metrics(X1, y1)
        results.append(("Space1_Paper1", sil1, db1))

    # Space 2
    sub2 = df[["B_winsor", "B_Change", "RSSI_winsor"]].dropna()
    if len(sub2) > 10:
        X2 = scaler.fit_transform(sub2)
        y2 = df.loc[sub2.index, "outcome"].values
        sil2, db2 = cluster_metrics(X2, y2)
        results.append(("Space2_Paper2", sil2, db2))

    # Space 3 (Joint)
    sub3 = df[["B_winsor", "B_Acceleration_winsor", "RSSI_winsor"]].dropna()
    if len(sub3) > 10:
        X3 = scaler.fit_transform(sub3)
        y3 = df.loc[sub3.index, "outcome"].values
        sil3, db3 = cluster_metrics(X3, y3)
        results.append(("Space3_Joint", sil3, db3))

    df_res = pd.DataFrame(results, columns=["Space", "Silhouette", "Davies_Bouldin"])
    if not df_res.empty:
        df_res.to_csv(TABLE_DIR / "T5_5C_space_comparison.csv", index=False)
    return df_res


# =============================================================================
# BLOCK 5D – TRAJECTORY ANALYSIS (MEDIAN CENTROIDS)
# =============================================================================
def block5d_trajectory_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute median centroids for each outcome group and plot them overlaid on
    the 3D scatter.

    Returns
    -------
    pd.DataFrame
        Median coordinates for Collapse, Evolve, and Sustain.
    """
    centroids = df.groupby("outcome")[
        ["B_winsor", "B_Acceleration_winsor", "RSSI_winsor"]
    ].median()
    centroids.to_csv(TABLE_DIR / "T5_5D_centroids.csv")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"Collapse": "red", "Evolve": "orange", "Sustain": "blue"}

    # Scatter all points with low opacity
    for label, color in colors.items():
        mask = df["outcome"] == label
        ax.scatter(
            df.loc[mask, "B_winsor"],
            df.loc[mask, "B_Acceleration_winsor"],
            df.loc[mask, "RSSI_winsor"],
            c=color, alpha=0.2, s=10, label=label
        )

    # Plot centroids as large X markers
    for label, color in colors.items():
        if label in centroids.index:
            c = centroids.loc[label]
            ax.scatter(
                c["B_winsor"], c["B_Acceleration_winsor"], c["RSSI_winsor"],
                c=color, s=200, marker="X", edgecolor="black", linewidth=2
            )

    ax.set_xlabel("B (winsorized)", fontsize=10)
    ax.set_ylabel("B Acceleration (winsorized)", fontsize=10)
    ax.set_zlabel("RSSI", fontsize=10)
    ax.set_title("C2 Trajectories – Median Centroids", fontsize=14)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T5_trajectory_3D_winsor.png", dpi=150)
    plt.close()

    return centroids


# =============================================================================
# BLOCK 5E – CLUSTER SEPARATION BY RSSI PHASE
# =============================================================================
def block5e_by_rssi_phase(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Stratify the sample by RSSI tertiles (Low, Mid, Extreme) and compute
    separation metrics within each phase.

    Returns
    -------
    Optional[pd.DataFrame]
        Table with columns: RSSI_phase, Silhouette, Davies_Bouldin, N.
    """
    df = df.copy()
    try:
        df["RSSI_phase"] = pd.qcut(
            df["RSSI_winsor"], 3, labels=["Low", "Mid", "Extreme"], duplicates="drop"
        )
    except ValueError:
        print("    WARNING: Could not bin RSSI into tertiles.")
        return None

    results = []
    scaler = StandardScaler()
    for phase in ["Low", "Mid", "Extreme"]:
        sub = df[df["RSSI_phase"] == phase]
        if len(sub) < 10:
            continue
        X = sub[["B_winsor", "B_Acceleration_winsor", "RSSI_winsor"]].values
        X = scaler.fit_transform(X)
        y = sub["outcome"].values
        sil, db = cluster_metrics(X, y)
        results.append({
            "RSSI_phase": phase,
            "Silhouette": sil,
            "Davies_Bouldin": db,
            "N": len(sub)
        })

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res.to_csv(TABLE_DIR / "T5_5E_by_phase.csv", index=False)
    return df_res


# =============================================================================
# BLOCK 5F – FEATURE IMPORTANCE (RANDOM FOREST)
# =============================================================================
def block5f_feature_importance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train a Random Forest classifier on the three standardized features and
    extract feature importances.

    Returns
    -------
    pd.DataFrame
        Columns: Feature, Importance.
    """
    X = get_scaled_features(df, ["B_winsor", "B_Acceleration_winsor", "RSSI_winsor"])
    y = df["outcome"].values
    rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=5)
    rf.fit(X, y)

    feat_df = pd.DataFrame({
        "Feature": ["B", "B_Acceleration", "RSSI"],
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    feat_df.to_csv(TABLE_DIR / "T5_5F_feature_importance.csv", index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
    plt.title("Feature Importance (Random Forest)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T5_feature_importance.png", dpi=150)
    plt.close()

    return feat_df


# =============================================================================
# BLOCK 5G – CONDITIONAL PROBABILITY HEATMAPS
# =============================================================================
def block5g_conditional_probability_heatmap(df: pd.DataFrame) -> bool:
    """
    Create heatmaps of collapse probability P(collapse | B, B_Acceleration)
    for each RSSI phase (Low, Mid, Extreme). Data are binned into 15×15 grids.

    Returns
    -------
    bool
        True if heatmaps were successfully generated.
    """
    df = df.copy()
    try:
        df["RSSI_phase"] = pd.qcut(
            df["RSSI_winsor"], 3, labels=["Low", "Mid", "Extreme"], duplicates="drop"
        )
    except ValueError:
        df["RSSI_phase"] = "All"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True, sharex=True)
    phases = ["Low", "Mid", "Extreme"]
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)

    for i, phase in enumerate(phases):
        ax = axes[i]
        sub = df[df["RSSI_phase"] == phase]
        if len(sub) < 20:
            ax.text(0.5, 0.5, f"Insufficient data\n(n={len(sub)})",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"RSSI = {phase}")
            continue

        sub["B_bin"] = pd.qcut(sub["B_winsor"], 15, labels=False, duplicates="drop")
        sub["Bacc_bin"] = pd.qcut(sub["B_Acceleration_winsor"], 15, labels=False, duplicates="drop")
        heat_data = sub.groupby(["B_bin", "Bacc_bin"])["collapse_binary"].mean().unstack()

        b_edges = np.percentile(sub["B_winsor"], np.linspace(0, 100, 16))
        bacc_edges = np.percentile(sub["B_Acceleration_winsor"], np.linspace(0, 100, 16))
        xticks = np.arange(0, 15, 3)
        yticks = np.arange(0, 15, 3)

        sns.heatmap(
            heat_data, ax=ax, cmap=cmap, vmin=0, vmax=1,
            cbar=i == 2, cbar_kws={"label": "P(collapse)" if i == 2 else ""}
        )
        ax.set_title(f"RSSI = {phase} (n={len(sub):,})")
        ax.set_xlabel("B (winsorized, percentile bins)")
        ax.set_ylabel("B Acceleration (winsorized, percentile bins)")
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{b_edges[x]:.1f}" for x in xticks], rotation=45)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{bacc_edges[y]:.1f}" for y in yticks])

    plt.suptitle("Conditional Collapse Probability by RSSI Phase (Winsorized)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "T5_conditional_probability_heatmap_winsor.png", dpi=150)
    plt.close()
    print("  Conditional probability heatmaps saved.")
    return True


# =============================================================================
# ACADEMIC SUMMARY REPORT GENERATION
# =============================================================================
def generate_academic_report(results: Dict) -> None:
    """
    Write a comprehensive, publication‑style summary report for Test 5.

    Parameters
    ----------
    results : Dict
        Dictionary containing outputs from all blocks.
    """
    report_path = REPORT_DIR / "T5_Cluster_Separation_Report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("CLUSTER SEPARATION ANALYSIS – ACADEMIC SUMMARY REPORT (TEST 5)\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Number of C2 observations analyzed: {results.get('n_obs', 'N/A'):,}\n")
        f.write("(B and B_Acceleration winsorized at 1%/99%; RSSI assumed winsorized)\n\n")

        # 5A
        f.write("1. BASELINE 2D SEPARATION (B, B_Acceleration) – Block 5A\n")
        f.write("-" * 60 + "\n")
        res_a = results.get("5A")
        if res_a:
            f.write(f"   Silhouette Score    : {res_a.get('Silhouette_2D', np.nan):.4f}\n")
            f.write(f"   Davies‑Bouldin Score: {res_a.get('Davies_Bouldin_2D', np.nan):.4f}\n\n")
        else:
            f.write("   No data.\n\n")

        # 5B
        f.write("2. 3D SEPARATION WITH RSSI (B, B_Acceleration, RSSI) – Block 5B\n")
        f.write("-" * 60 + "\n")
        res_b = results.get("5B")
        if res_b:
            f.write(f"   Silhouette Score (3D): {res_b.get('Silhouette_3D', np.nan):.4f}\n")
            f.write(f"   Davies‑Bouldin (3D)  : {res_b.get('Davies_Bouldin_3D', np.nan):.4f}\n")
            f.write(f"   Permutation test p    : {res_b.get('Permutation_p', np.nan):.4f}\n")
            if res_b.get("Permutation_p", 1) < 0.05:
                f.write("   → RSSI significantly improves separation (p < 0.05).\n")
            else:
                f.write("   → No significant improvement detected.\n")
            f.write("\n")
        else:
            f.write("   No data.\n\n")

        # 5C
        f.write("3. COMPARISON OF ALTERNATIVE 3D SPACES – Block 5C\n")
        f.write("-" * 60 + "\n")
        res_c = results.get("5C")
        if res_c is not None and not res_c.empty:
            f.write(res_c.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("   No data.\n\n")

        # 5D
        f.write("4. TRAJECTORY ANALYSIS (MEDIAN CENTROIDS) – Block 5D\n")
        f.write("-" * 60 + "\n")
        res_d = results.get("5D")
        if res_d is not None and not res_d.empty:
            f.write(res_d.to_string())
            f.write("\n\n")
        else:
            f.write("   No data.\n\n")

        # 5E
        f.write("5. CLUSTER SEPARATION BY RSSI PHASE – Block 5E\n")
        f.write("-" * 60 + "\n")
        res_e = results.get("5E")
        if res_e is not None and not res_e.empty:
            f.write(res_e.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("   No data.\n\n")

        # 5F
        f.write("6. FEATURE IMPORTANCE (RANDOM FOREST) – Block 5F\n")
        f.write("-" * 60 + "\n")
        res_f = results.get("5F")
        if res_f is not None and not res_f.empty:
            f.write(res_f.to_string(index=False))
            f.write("\n\n")
        else:
            f.write("   No data.\n\n")

        # 5G
        f.write("7. CONDITIONAL PROBABILITY HEATMAPS – Block 5G\n")
        f.write("-" * 60 + "\n")
        if results.get("5G"):
            f.write("   Heatmaps saved to figures/T5_conditional_probability_heatmap_winsor.png\n\n")
        else:
            f.write("   Heatmap generation failed.\n\n")

        f.write("=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")

    print(f"Academic report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main() -> None:
    """Execute the full cluster separation analysis pipeline."""
    print("=" * 80)
    print("CLUSTER SEPARATION ANALYSIS – ACADEMIC PIPELINE (TEST 5)")
    print("=" * 80)

    try:
        df = load_and_prepare_c2_data()
    except ValueError as e:
        print(f"Data loading failed: {e}")
        return

    if len(df) < 20:
        print("Insufficient C2 observations. Exiting.")
        return

    results: Dict = {"n_obs": len(df)}

    print("\nBlock 5A: Baseline 2D separation...")
    results["5A"] = block5a_baseline_2d(df)

    print("Block 5B: 3D separation with RSSI...")
    results["5B"] = block5b_3d_with_rssi(df)

    print("Block 5C: Alternative spaces comparison...")
    results["5C"] = block5c_alternative_spaces(df)

    print("Block 5D: Trajectory centroids...")
    results["5D"] = block5d_trajectory_centroids(df)

    print("Block 5E: By RSSI phase...")
    results["5E"] = block5e_by_rssi_phase(df)

    print("Block 5F: Feature importance...")
    results["5F"] = block5f_feature_importance(df)

    print("Block 5G: Conditional probability heatmaps...")
    results["5G"] = block5g_conditional_probability_heatmap(df)

    print("\nGenerating academic summary report...")
    generate_academic_report(results)

    print("\nAll results saved to 'results/' directory.")


if __name__ == "__main__":
    main()