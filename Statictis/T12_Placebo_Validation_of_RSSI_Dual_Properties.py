"""
PLACEBO VALIDATION OF RSSI DUAL PROPERTIES (ACADEMIC STANDARD)
Methodology: shuffle independent variable(s) while keeping collapse_4Q unchanged.
- 12A: Shuffle RSSI_cat (Low/Mid/Extreme) within (Phi=1, C2/C3)
- 12B: Shuffle RSSI_direction (asc/desc) within (B_high, Phi=1, C2/C3, Mid)
- 12D: Shuffle B_high (True/False) within (Phi=1, RSSI_cat=Mid)
- 12E: Shuffle RSSI time series (within each firm) to break autocorrelation, only for B_high firms
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASSIFIED_DIR = Path("../data/classified")   # ADJUST THIS PATH
RANDOM_SEED = 42
N_PERM = 1000
SECTORS = ["Healthcare", "Technology", "Services"]

# =============================================================================
# DATA LOADING & PREPARATION (identical to original)
# =============================================================================
def load_full_panel() -> pd.DataFrame:
    required = [
        "Ticker", "period_end", "Configuration", "B", "RSSI", "Phi_t",
        "dRSSI_dt", "collapse_next"
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
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                continue
    if not all_dfs:
        raise ValueError("No valid data found. Check CLASSIFIED_DIR path.")
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(["Ticker", "period_end"]).reset_index(drop=True)
    return panel

def prepare_panel(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df = df.sort_values(["Ticker", "period_end"])

    # collapse_4Q: collapse within next 4 quarters
    df["collapse_4Q"] = 0
    for lag in range(1, 5):
        col = f"c_{lag}"
        df[col] = df.groupby("Ticker")["collapse_next"].shift(-lag)
        df["collapse_4Q"] = df["collapse_4Q"] | df[col].fillna(0).astype(int)
    df["collapse_4Q"] = df["collapse_4Q"].astype(int)

    # RSSI categories (terciles)
    try:
        df["RSSI_cat"] = pd.qcut(df["RSSI"], 3, labels=["Low", "Mid", "Extreme"], duplicates="drop")
    except ValueError:
        df["RSSI_cat"] = "Mid"

    # RSSI direction (sector-level)
    sector_rssi = (df.groupby(["Sector", "period_end"])["RSSI"]
                   .first().reset_index()
                   .sort_values(["Sector", "period_end"]))
    sector_rssi["dRSSI"] = sector_rssi.groupby("Sector")["RSSI"].diff()
    sector_rssi["RSSI_direction"] = np.where(
        sector_rssi["dRSSI"] > 0, "ascending",
        np.where(sector_rssi["dRSSI"] < 0, "descending", "flat")
    )
    df = df.merge(sector_rssi[["Sector", "period_end", "RSSI_direction"]],
                  on=["Sector", "period_end"], how="left")

    # B_high (top quartile)
    df["B_high"] = df["B"] > df["B"].quantile(0.75)

    # Keep only speculative configurations
    df = df[df["Configuration"].isin(["C2", "C3", "C4"])].copy()
    return df

# =============================================================================
# 12A – MODERATOR PROPERTY (shuffle RSSI_cat)
# =============================================================================
def test12A_shuffle_rssi(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """
    Permutation test for moderator property (interaction B × RSSI).
    Shuffle RSSI_cat (Low/Mid) within (Phi=1, C2/C3) while keeping collapse_4Q.
    Returns observed log-odds difference and one-sided p-value.
    """
    np.random.seed(seed)
    mask = (df['Phi_t'] == 1) & (df['Configuration'].isin(['C2','C3']))
    sub = df[mask].copy()
    sub = sub[sub['RSSI_cat'].isin(['Low','Mid'])].copy()
    if len(sub) < 30:
        return None

    def interaction_effect(data: pd.DataFrame) -> float:
        high = data[data['B_high']]
        low = data[~data['B_high']]
        def log_or(g):
            tab = pd.crosstab(g['RSSI_cat'] == 'Mid', g['collapse_4Q'])
            if tab.shape != (2,2) or (tab.values == 0).any():
                return np.nan
            or_val = (tab.iloc[0,0] * tab.iloc[1,1]) / (tab.iloc[0,1] * tab.iloc[1,0])
            return np.log(or_val)
        logOR_high = log_or(high)
        logOR_low = log_or(low)
        if np.isnan(logOR_high) or np.isnan(logOR_low):
            return np.nan
        return logOR_high - logOR_low

    obs_eff = interaction_effect(sub)
    if np.isnan(obs_eff):
        return None

    null_effs = []
    for _ in range(n_perm):
        sub_perm = sub.copy()
        sub_perm['RSSI_cat'] = np.random.permutation(sub_perm['RSSI_cat'].values)
        eff_perm = interaction_effect(sub_perm)
        if not np.isnan(eff_perm):
            null_effs.append(eff_perm)
    if not null_effs:
        return None
    p_val = np.mean(np.array(null_effs) >= obs_eff)
    return {'obs_effect': obs_eff, 'p_value': p_val, 'n_perm': len(null_effs)}

# =============================================================================
# 12B – CYCLE COORDINATE PROPERTY (shuffle RSSI_direction)
# =============================================================================
def test12B_shuffle_direction(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """
    Test whether ascending vs descending collapse rate difference is real.
    Shuffle RSSI_direction within (B_high, Phi=1, C2/C3, Mid) while keeping collapse_4Q.
    """
    np.random.seed(seed)
    mask = (df['B_high']) & (df['Phi_t']==1) & (df['Configuration'].isin(['C2','C3'])) & (df['RSSI_cat']=='Mid')
    sub = df[mask].copy()
    if len(sub) < 20:
        return None
    asc = sub[sub['RSSI_direction']=='ascending']['collapse_4Q']
    desc = sub[sub['RSSI_direction']=='descending']['collapse_4Q']
    if len(asc) < 5 or len(desc) < 5:
        return None
    obs_diff = asc.mean() - desc.mean()
    null_diffs = []
    for _ in range(n_perm):
        sub_perm = sub.copy()
        sub_perm['RSSI_direction'] = np.random.permutation(sub_perm['RSSI_direction'].values)
        asc_p = sub_perm[sub_perm['RSSI_direction']=='ascending']['collapse_4Q']
        desc_p = sub_perm[sub_perm['RSSI_direction']=='descending']['collapse_4Q']
        if len(asc_p) >= 5 and len(desc_p) >= 5:
            null_diffs.append(asc_p.mean() - desc_p.mean())
    if not null_diffs:
        return None
    p_val = np.mean(np.array(null_diffs) >= obs_diff)
    return {'obs_diff': obs_diff, 'asc_rate': asc.mean(), 'desc_rate': desc.mean(),
            'p_value': p_val, 'n_perm': len(null_diffs)}

# =============================================================================
# 12D – ENVIRONMENT MODERATOR (shuffle B_high within Phi=1, Mid)
# =============================================================================
def test12D_shuffle_B(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = RANDOM_SEED):
    """
    Within the environment (Phi=1, RSSI_cat=Mid), test whether B_high increases collapse.
    Shuffle B_high labels while keeping collapse_4Q.
    """
    np.random.seed(seed)
    mask = (df['Phi_t']==1) & (df['RSSI_cat']=='Mid')
    sub = df[mask].copy()
    if len(sub) < 30:
        return None
    obs_diff = sub[sub['B_high']]['collapse_4Q'].mean() - sub[~sub['B_high']]['collapse_4Q'].mean()
    null_diffs = []
    for _ in range(n_perm):
        sub_perm = sub.copy()
        sub_perm['B_high'] = np.random.permutation(sub_perm['B_high'].values)
        diff_perm = sub_perm[sub_perm['B_high']]['collapse_4Q'].mean() - sub_perm[~sub_perm['B_high']]['collapse_4Q'].mean()
        null_diffs.append(diff_perm)
    p_val = np.mean(np.array(null_diffs) >= obs_diff)
    return {'obs_diff': obs_diff, 'p_value': p_val, 'n_perm': n_perm}

# =============================================================================
# 12E – CYCLE PROPERTY OF RSSI ITSELF (autocorrelation, shuffle within firm)
# =============================================================================
def test12E_shuffle_rssi_cycle(df: pd.DataFrame, n_perm: int = 1000, seed: int = RANDOM_SEED):
    """
    For firms with B_high, compute lag-1 autocorrelation of RSSI.
    Null distribution: shuffle RSSI values within each firm (breaking time series).
    """
    np.random.seed(seed)
    high_firms = df[df['B_high']]['Ticker'].unique()
    if len(high_firms) < 10:
        return None
    obs_acf = []
    null_acf = []
    for ticker in high_firms:
        firm = df[df['Ticker']==ticker].sort_values('period_end')
        rssi = firm['RSSI'].dropna()
        if len(rssi) < 5:
            continue
        acf_obs = rssi.autocorr(lag=1)
        if np.isnan(acf_obs):
            continue
        obs_acf.append(acf_obs)
        for _ in range(n_perm):
            rssi_shuffled = rssi.sample(frac=1, replace=False).reset_index(drop=True)
            acf_perm = rssi_shuffled.autocorr(lag=1)
            if not np.isnan(acf_perm):
                null_acf.append(acf_perm)
    if not obs_acf or not null_acf:
        return None
    mean_obs = np.mean(obs_acf)
    p_val = np.mean(np.array(null_acf) >= mean_obs)
    return {'mean_obs_autocorr': mean_obs, 'p_value': p_val, 'n_firms': len(obs_acf)}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("Loading data...")
    panel = load_full_panel()
    df = prepare_panel(panel)
    print(f"Speculative observations: {len(df)}")

    results = {}

    print("\n12A - Moderator (shuffle RSSI_cat within Phi=1, C2/C3):")
    resA = test12A_shuffle_rssi(df)
    results['12A'] = resA
    if resA:
        print(f"  Observed effect (log-odds diff) = {resA['obs_effect']:.4f}, p = {resA['p_value']:.4f}")
    else:
        print("  Insufficient data")

    print("\n12B - Cycle coordinate (shuffle direction within B_high, Phi=1, C2/C3, Mid):")
    resB = test12B_shuffle_direction(df)
    results['12B'] = resB
    if resB:
        print(f"  Observed diff (asc - desc) = {resB['obs_diff']:.4f}, asc_rate={resB['asc_rate']:.3f}, desc_rate={resB['desc_rate']:.3f}, p={resB['p_value']:.4f}")
    else:
        print("  Insufficient data")

    print("\n12D - Environment moderator (shuffle B_high within Phi=1, Mid):")
    resD = test12D_shuffle_B(df)
    results['12D'] = resD
    if resD:
        print(f"  Observed diff (B_high - B_low) = {resD['obs_diff']:.4f}, p = {resD['p_value']:.4f}")
    else:
        print("  Insufficient data")

    print("\n12E - RSSI cycle (autocorrelation, shuffle within firm, B_high firms):")
    resE = test12E_shuffle_rssi_cycle(df)
    results['12E'] = resE
    if resE:
        print(f"  Mean autocorr = {resE['mean_obs_autocorr']:.4f}, p = {resE['p_value']:.4f}, n_firms={resE['n_firms']}")
    else:
        print("  Insufficient data")

    # Save results to a report file
    report_path = Path("results/reports") / "T12_Placebo_Validation_of_RSSI_dual_properties.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("PLACEBO VALIDATION (ACADEMIC STANDARD)\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"Settings: N_PERM = {N_PERM}, seed = {RANDOM_SEED}\n")
        f.write("="*80 + "\n\n")
        for name, res in results.items():
            f.write(f"{name}:\n")
            if res is None:
                f.write("  Insufficient data\n\n")
            else:
                for k, v in res.items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
    print(f"\nResults saved to {report_path}")

if __name__ == "__main__":
    main()