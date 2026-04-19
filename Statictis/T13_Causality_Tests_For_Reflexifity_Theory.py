"""
ACADEMIC_CAUSALITY_TESTS_REFLEXIVITY.py

Performs three causality tests for reflexivity theory:
1. Firm-level Granger causality: MRF → MCF under reflexive regime (Phi=1, B_high, C2/C3)
2. Sector-level Granger causality: MCF_sector → RSSI using all data (no regime filter, simple Granger)
3. Contemporaneous permutation test: B → collapse (4-quarter horizon) under Phi=1 & RSSI=Mid

All results are saved to results/reports/ with full reproducibility (random seed = 42).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import f
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASSIFIED_DIR = Path("../data/classified")   # Input data directory
OUTPUT_DIR = Path("results/reports")          # Output directory for reports
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
SECTORS = ["Healthcare", "Technology", "Services"]

# Create output directory if it does not exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================
def load_full_panel():
    """Load and concatenate all classified firm-quarter data."""
    required = [
        "Ticker", "period_end", "Configuration", "B", "RSSI", "Phi_t",
        "dRSSI_dt", "collapse_next", "MCF_t", "MRF_t_placeholder"
    ]
    all_dfs = []
    for sector in SECTORS:
        sector_path = CLASSIFIED_DIR / sector
        if not sector_path.exists():
            continue
        for fpath in sector_path.glob("*_classified.csv"):
            try:
                df = pd.read_csv(fpath, parse_dates=["period_end"])
                if "Sector" not in df.columns:
                    df["Sector"] = sector
                if "collapse_next" not in df.columns:
                    df = df.sort_values(["Ticker", "period_end"])
                    df["next_config"] = df.groupby("Ticker")["Configuration"].shift(-1)
                    df["collapse_next"] = df["next_config"].isin(["C1", "C6"]).astype(int)
                    df.drop(columns=["next_config"], inplace=True)
                missing = [c for c in required if c not in df.columns]
                if missing:
                    continue
                df = df[required + ["Sector"]].copy()
                all_dfs.append(df)
            except Exception:
                continue
    if not all_dfs:
        raise ValueError("No valid data found. Check CLASSIFIED_DIR.")
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel

def prepare_panel(panel):
    """Add derived variables: collapse_4Q, RSSI categories, B_high."""
    df = panel.copy()
    df = df.sort_values(["Ticker", "period_end"])
    
    # Collapse within 4 quarters
    df["collapse_4Q"] = 0
    for lag in range(1, 5):
        col = f"c_{lag}"
        df[col] = df.groupby("Ticker")["collapse_next"].shift(-lag)
        df["collapse_4Q"] = df["collapse_4Q"] | df[col].fillna(0).astype(int)
    df["collapse_4Q"] = df["collapse_4Q"].astype(int)
    
    # RSSI terciles
    try:
        df["RSSI_cat"] = pd.qcut(df["RSSI"], 3, labels=["Low", "Mid", "Extreme"], duplicates="drop")
    except ValueError:
        df["RSSI_cat"] = "Mid"
    
    # B_high (top quartile)
    df["B_high"] = df["B"] > df["B"].quantile(0.75)
    return df

# =============================================================================
# GRANGER TESTS
# =============================================================================
def granger_test_robust(y, x, lag):
    """
    Robust Granger causality test (used for Test 1).
    Returns (F_stat, p_value). Returns (NaN, NaN) if assumptions fail.
    """
    n = len(y)
    if n < lag + 10:
        return np.nan, np.nan
    
    y_target = y[lag:]
    y_lags = []
    x_lags = []
    for t in range(lag, n):
        y_lags.append(y[t-lag:t][::-1])
        x_lags.append(x[t-lag:t][::-1])
    
    # Restricted model (only y lags)
    X_rest = np.column_stack((np.ones(len(y_lags)), np.array(y_lags)))
    try:
        coeff_rest, _, _, _ = np.linalg.lstsq(X_rest, y_target, rcond=None)
        rss_rest = np.sum((y_target - X_rest @ coeff_rest)**2)
    except:
        return np.nan, np.nan
    
    # Unrestricted model (y lags + x lags)
    X_unrest = np.column_stack((np.ones(len(y_lags)), np.array(y_lags), np.array(x_lags)))
    try:
        coeff_unrest, _, _, _ = np.linalg.lstsq(X_unrest, y_target, rcond=None)
        rss_unrest = np.sum((y_target - X_unrest @ coeff_unrest)**2)
    except:
        return np.nan, np.nan
    
    if rss_unrest > rss_rest + 1e-8:
        return np.nan, np.nan
    
    df_num = lag
    df_den = len(y_target) - (2*lag + 1)
    if df_den <= 0:
        return np.nan, np.nan
    
    f_stat = ((rss_rest - rss_unrest) / df_num) / (rss_unrest / df_den)
    if f_stat < 0:
        return np.nan, np.nan
    p_val = 1 - f.cdf(f_stat, df_num, df_den)
    return f_stat, p_val

def granger_test_simple(y, x, lag):
    """
    Simple Granger causality test (used for Test 2).
    Returns (F_stat, p_value). No additional robustness checks.
    """
    n = len(y)
    y_target = y[lag:]
    X = []
    for t in range(lag, n):
        row = []
        for l in range(1, lag+1):
            row.append(y[t-l])
        for l in range(1, lag+1):
            row.append(x[t-l])
        X.append(row)
    X = np.column_stack((np.ones(len(X)), np.array(X)))
    
    # Unrestricted
    coeff, _, _, _ = np.linalg.lstsq(X, y_target, rcond=None)
    y_pred = X @ coeff
    rss_unrest = np.sum((y_target - y_pred)**2)
    
    # Restricted (only y lags)
    X_rest = np.column_stack((np.ones(len(X)), np.array(X)[:,:lag]))
    coeff_rest, _, _, _ = np.linalg.lstsq(X_rest, y_target, rcond=None)
    y_pred_rest = X_rest @ coeff_rest
    rss_rest = np.sum((y_target - y_pred_rest)**2)
    
    f_stat = ((rss_rest - rss_unrest) / lag) / (rss_unrest / (len(y_target) - 2*lag - 1))
    p_val = 1 - f.cdf(f_stat, lag, len(y_target) - 2*lag - 1)
    return f_stat, p_val

# =============================================================================
# DATA PREPARATION FOR SPECIFIC TESTS
# =============================================================================
def prepare_pooled_ts(df, condition, vars):
    """Extract pooled time series for firm-level tests."""
    sub = df[condition].copy()
    sub = sub.sort_values(['Ticker', 'period_end'])
    ts = sub[vars].dropna().reset_index(drop=True)
    return ts

def sector_level_data_raw(df):
    """Aggregate MCF to sector level using all data (no regime filter)."""
    mcf_sector = df.groupby(['Sector', 'period_end'])['MCF_t'].mean().reset_index()
    mcf_sector.rename(columns={'MCF_t': 'MCF_sector'}, inplace=True)
    rssi_sector = df[['Sector', 'period_end', 'RSSI']].drop_duplicates()
    merged = mcf_sector.merge(rssi_sector, on=['Sector', 'period_end'])
    merged = merged.sort_values(['Sector', 'period_end'])
    return merged

# =============================================================================
# REPORT GENERATION
# =============================================================================
def write_report(content_lines, filename):
    """Write report lines to a UTF-8 text file and also print to console."""
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        for line in content_lines:
            f.write(line + '\n')
    for line in content_lines:
        print(line)

def main():
    print("=" * 70)
    print("REFLEXIVITY THEORY – CAUSALITY TESTS (Reproducible, Seed=42)")
    print("=" * 70)
    
    # Load and prepare data
    panel = load_full_panel()
    df = prepare_panel(panel)
    
    report_lines = []
    report_lines.append("ACADEMIC CAUSALITY REPORT")
    report_lines.append(f"Random seed: {RANDOM_SEED}")
    report_lines.append(f"Total speculative observations (C2,C3,C4): {len(df)}")
    report_lines.append("")
    
    # ========== TEST 1: MRF → MCF (Reflexive regime) ==========
    cond_ref = (df['Phi_t'] == 1) & (df['B_high']) & (df['Configuration'].isin(['C2','C3']))
    ts_ref = prepare_pooled_ts(df, cond_ref, ['MCF_t', 'MRF_t_placeholder'])
    report_lines.append("--- TEST 1: MRF → MCF (Firm-level Granger, Reflexive regime) ---")
    report_lines.append(f"Condition: Phi_t=1 & B_high & Config in [C2,C3]")
    report_lines.append(f"Observations: {cond_ref.sum()}")
    report_lines.append(f"Pooled time-series length: {len(ts_ref)}")
    
    if len(ts_ref) > 20:
        mcf = ts_ref['MCF_t'].values
        mrf = ts_ref['MRF_t_placeholder'].values
        for lag in [1,2,4]:
            f_stat, p = granger_test_robust(mcf, mrf, lag)
            report_lines.append(f"  Lag {lag}: F = {f_stat:.4f}, p = {p:.4f}")
    else:
        report_lines.append("  Insufficient data (<=20 observations). Test skipped.")
    report_lines.append("")
    
    # ========== TEST 2: MCF_sector → RSSI (All data, simple Granger) ==========
    sector_df = sector_level_data_raw(df)
    sector_df = sector_df.dropna().reset_index(drop=True)
    report_lines.append("--- TEST 2: MCF_sector → RSSI (Sector-level Granger, All data) ---")
    report_lines.append(f"Pooled sector-quarter observations: {len(sector_df)}")
    y = sector_df['RSSI'].values
    x = sector_df['MCF_sector'].values
    for lag in [1,2,4]:
        f_stat, p = granger_test_simple(y, x, lag)
        report_lines.append(f"  Lag {lag}: F = {f_stat:.4f}, p = {p:.4f}")
    report_lines.append("  Reverse causality (RSSI → MCF_sector):")
    for lag in [1,2,4]:
        f_stat, p = granger_test_simple(x, y, lag)
        report_lines.append(f"    Lag {lag}: F = {f_stat:.4f}, p = {p:.4f}")
    report_lines.append("")
    
    # ========== TEST 3: B → collapse (Permutation test, Phi=1 & RSSI=Mid) ==========
    env_cond = (df['Phi_t'] == 1) & (df['RSSI_cat'] == 'Mid')
    env = df[env_cond].copy()
    report_lines.append("--- TEST 3: B → collapse (Contemporaneous permutation test) ---")
    report_lines.append(f"Condition: Phi_t=1 & RSSI_cat='Mid'")
    report_lines.append(f"Observations: {len(env)}")
    
    if len(env) > 20:
        high_rate = env[env['B_high']]['collapse_4Q'].mean()
        low_rate = env[~env['B_high']]['collapse_4Q'].mean()
        diff = high_rate - low_rate
        # Permutation test with fixed seed
        np.random.seed(RANDOM_SEED)
        null_diffs = []
        y_perm = env['collapse_4Q'].values
        b_high = env['B_high'].values
        for _ in range(1000):
            perm_b = np.random.permutation(b_high)
            high_perm = y_perm[perm_b].mean()
            low_perm = y_perm[~perm_b].mean()
            null_diffs.append(high_perm - low_perm)
        p_val = np.mean(np.array(null_diffs) >= diff)
        report_lines.append(f"  Collapse rate (B_high): {high_rate:.4f}")
        report_lines.append(f"  Collapse rate (B_low):  {low_rate:.4f}")
        report_lines.append(f"  Difference: {diff:.4f}")
        report_lines.append(f"  Permutation p-value (one-sided, H1: diff>0): {p_val:.4f}")
    else:
        report_lines.append("  Insufficient data (<=20 observations). Test skipped.")
    
    report_lines.append("")
    report_lines.append("--- END OF REPORT ---")
    
    # Write report
    filename = f"T13_causality_report.txt"
    write_report(report_lines, filename)
    print("\n" + "=" * 70)
    print(f"Report saved to: {OUTPUT_DIR / filename}")

if __name__ == "__main__":
    main()