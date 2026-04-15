"""
Joint_collapse_condition_complete.py

TEST 4: JOINT COLLAPSE CONDITION – BREAKING POINT IDENTIFICATION
----------------------------------------------------------------
- Fixed datetime issue in survival analysis (using year_quarter).
- Logistic regression uses RSSI dummy variables.
- Survival analysis: Kaplan‑Meier, log‑rank, Cox PH (with fallback).
- Absence pattern tested with B = Q5.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# OPTIONAL LIBRARY CHECKS
# =============================================================================
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not installed; logistic p‑values will not be computed.")

try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    print("WARNING: lifelines not installed; survival analysis will be skipped.")

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASSIFIED_DIR = Path('../data/classified')
PROCESSED_DIR = Path('../data/processed')
OUTPUT_DIR = Path('results')
TABLE_DIR = OUTPUT_DIR / 'tables'
FIGURE_DIR = OUTPUT_DIR / 'figures'
REPORT_DIR = OUTPUT_DIR / 'reports'
for d in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SECTORS = ['Healthcare', 'Technology', 'Services']

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def load_full_panel():
    """Load panel data from classified CSV files, adding Sector and collapse_next if missing."""
    required = ['Ticker', 'period_end', 'Configuration',
                'D_t', 'B', 'RSSI', 'dRSSI_dt']
    all_dfs = []
    for sector in SECTORS:
        sector_path = CLASSIFIED_DIR / sector
        if not sector_path.exists():
            continue
        for fpath in sector_path.glob('*_classified.csv'):
            try:
                df = pd.read_csv(fpath, parse_dates=['period_end'])
                if 'Sector' not in df.columns:
                    df['Sector'] = sector
                if 'collapse_next' not in df.columns:
                    if 'Configuration' in df.columns:
                        df = df.sort_values(['Ticker', 'period_end'])
                        df['next_config'] = df.groupby('Ticker')['Configuration'].shift(-1)
                        df['collapse_next'] = df['next_config'].isin(['C1', 'C6']).astype(int)
                        df.drop(columns=['next_config'], inplace=True)
                    else:
                        continue
                missing = [c for c in required if c not in df.columns]
                if missing:
                    continue
                df = df[required + ['Sector', 'collapse_next']].copy()
                all_dfs.append(df)
            except Exception:
                continue
    if not all_dfs:
        raise FileNotFoundError("No valid classified files found.")
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    return panel


def add_collapse_windows(df):
    """Add collapse_4Q and collapse_6Q indicators."""
    df = df.sort_values(['Ticker', 'period_end'])
    df['collapse_4Q'] = 0
    for lag in range(1, 5):
        col = f'collapse_t{lag}'
        df[col] = df.groupby('Ticker')['Configuration'].shift(-lag).isin(['C1', 'C6']).astype(int)
        df['collapse_4Q'] = df['collapse_4Q'] | df[col]
    df['collapse_4Q'] = df['collapse_4Q'].astype(int)
    df['collapse_6Q'] = df['collapse_4Q']
    for lag in range(5, 7):
        col = f'collapse_t{lag}'
        df[col] = df.groupby('Ticker')['Configuration'].shift(-lag).isin(['C1', 'C6']).astype(int)
        df['collapse_6Q'] = df['collapse_6Q'] | df[col]
    df['collapse_6Q'] = df['collapse_6Q'].astype(int)
    min_periods = df.groupby('Ticker')['period_end'].transform('count')
    df = df[min_periods >= 7].copy()
    return df


def assign_levels(df):
    """Create categorical levels for RSSI and B."""
    df['RSSI_q5'] = pd.qcut(df['RSSI'], 5, labels=False, duplicates='drop') + 1

    def rssi_group(q):
        if q <= 2: return 'Low'
        elif q == 3: return 'Mid'
        else: return 'Extreme'
    df['RSSI_level'] = df['RSSI_q5'].apply(lambda x: rssi_group(x) if pd.notna(x) else np.nan)

    df['B_level'] = np.where(df['B'] > df['B'].quantile(0.75), 'High', 'Low')
    df['B_q5'] = pd.qcut(df['B'], 5, labels=False, duplicates='drop') + 1
    return df


def bootstrap_ci(data, k=1000, alpha=0.05):
    """Return bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return np.nan, np.nan
    means = np.random.choice(data, (k, len(data)), replace=True).mean(axis=1)
    return np.percentile(means, 100*alpha/2), np.percentile(means, 100*(1-alpha/2))


# =============================================================================
# BLOCK 4A & 4B: HEATMAPS
# =============================================================================
def block4A_2x3_matrix(df):
    """2×3 matrix: B_level (Low/High) × RSSI_level (Low/Mid/Extreme)."""
    data = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].dropna(
        subset=['B_level', 'RSSI_level', 'collapse_4Q']
    )
    results = []
    for b in ['Low', 'High']:
        for r in ['Low', 'Mid', 'Extreme']:
            sub = data[(data['B_level'] == b) & (data['RSSI_level'] == r)]
            if len(sub) == 0:
                continue
            rate = sub['collapse_4Q'].mean()
            ci_low, ci_high = bootstrap_ci(sub['collapse_4Q'].values)
            results.append({
                'B_level': b, 'RSSI_level': r, 'N': len(sub),
                'collapse_rate': rate, 'CI_low': ci_low, 'CI_high': ci_high
            })
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T4_4A_2x3_matrix.csv', index=False)
    pivot = df_res.pivot(index='B_level', columns='RSSI_level', values='collapse_rate')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='Reds')
    plt.title('P(collapse_4Q) by B_level × RSSI_level')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'T4_heatmap_2x3.png', dpi=150)
    plt.close()
    return df_res


def block4B_5x3_matrix(df):
    """5×3 matrix: B_quantile (1‑5) × RSSI_level (Low/Mid/Extreme)."""
    data = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].dropna(
        subset=['B_q5', 'RSSI_level', 'collapse_4Q']
    )
    results = []
    for b in range(1, 6):
        for r in ['Low', 'Mid', 'Extreme']:
            sub = data[(data['B_q5'] == b) & (data['RSSI_level'] == r)]
            if len(sub) == 0:
                continue
            rate = sub['collapse_4Q'].mean()
            results.append({
                'B_quantile': b, 'RSSI_level': r, 'N': len(sub), 'collapse_rate': rate
            })
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T4_4B_5x3_matrix.csv', index=False)
    pivot = df_res.pivot(index='B_quantile', columns='RSSI_level', values='collapse_rate')
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='Reds')
    plt.title('P(collapse_4Q) by B_quantile × RSSI_level')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'T4_heatmap_5x3.png', dpi=150)
    plt.close()
    return df_res


# =============================================================================
# BLOCK 4C: LOGISTIC REGRESSION WITH RSSI DUMMIES
# =============================================================================
def block4C_logistic(df):
    """Logistic regression using RSSI phase dummies and interactions."""
    data = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].dropna(
        subset=['B', 'RSSI', 'collapse_4Q']
    ).copy()
    if len(data) == 0:
        return None

    data['B_std'] = (data['B'] - data['B'].mean()) / data['B'].std()
    data['RSSI_Mid'] = (data['RSSI_level'] == 'Mid').astype(int)
    data['RSSI_Extreme'] = (data['RSSI_level'] == 'Extreme').astype(int)
    data['BxMid'] = data['B_std'] * data['RSSI_Mid']
    data['BxExt'] = data['B_std'] * data['RSSI_Extreme']

    X = data[['B_std', 'RSSI_Mid', 'RSSI_Extreme', 'BxMid', 'BxExt']]
    y = data['collapse_4Q']

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    results = {}
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    model.fit(X, y)
    y_prob = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_prob)
    results['AUC'] = auc
    results['coef'] = dict(zip(X.columns, model.coef_[0]))

    if STATSMODELS_AVAILABLE:
        X_sm = sm.add_constant(X)
        logit = sm.Logit(y, X_sm).fit(disp=0)
        results['pvalues'] = logit.pvalues.to_dict()
        results['pseudo_r2'] = logit.prsquared

    with open(TABLE_DIR / 'T4_4C_logistic.txt', 'w', encoding='utf-8') as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    return results


# =============================================================================
# BLOCK 4D: SURVIVAL ANALYSIS (DURATION FIXED)
# =============================================================================
def block4D_survival(df):
    """Kaplan‑Meier, log‑rank, and Cox PH (if lifelines available)."""
    if not LIFELINES_AVAILABLE:
        with open(TABLE_DIR / 'T4_4D_survival.txt', 'w', encoding='utf-8') as f:
            f.write("Survival analysis requires lifelines.\n")
        return None

    data = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].dropna(
        subset=['B_level', 'RSSI_level']
    ).copy()
    if len(data) == 0:
        return None

    data = data.sort_values(['Ticker', 'period_end'])

    # Compute duration (quarters)
    data['yq'] = data['period_end'].dt.year * 4 + data['period_end'].dt.quarter
    data['collapse_yq'] = np.nan
    mask_collapse = data['collapse_next'] == 1
    data.loc[mask_collapse, 'collapse_yq'] = data.loc[mask_collapse, 'yq']
    data['collapse_yq'] = data.groupby('Ticker')['collapse_yq'].bfill()
    data['duration'] = data['collapse_yq'] - data['yq']
    data['duration'] = data['duration'].fillna(12).clip(lower=1)
    data['event'] = data['collapse_next']

    # Define risk groups
    data['group'] = 'Other'
    data.loc[(data['B_level'] == 'High') & (data['RSSI_level'] == 'Mid'), 'group'] = 'B=High,RSSI=Mid'
    data.loc[(data['B_level'] == 'High') & (data['RSSI_level'] == 'Low'), 'group'] = 'B=High,RSSI=Low'
    data.loc[(data['B_level'] == 'High') & (data['RSSI_level'] == 'Extreme'), 'group'] = 'B=High,RSSI=Ext'
    data.loc[(data['B_level'] == 'Low') & (data['RSSI_level'] == 'Mid'), 'group'] = 'B=Low,RSSI=Mid'
    main_groups = ['B=High,RSSI=Mid', 'B=High,RSSI=Low', 'B=High,RSSI=Ext', 'B=Low,RSSI=Mid']
    data = data[data['group'].isin(main_groups)].copy()

    results = {}
    # Kaplan‑Meier
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    for g in main_groups:
        sub = data[data['group'] == g]
        if len(sub) == 0:
            continue
        kmf.fit(sub['duration'], sub['event'], label=g)
        kmf.plot_survival_function()
    plt.title('Kaplan‑Meier Survival Curves (actual duration)')
    plt.xlabel('Time to collapse (quarters)')
    plt.ylabel('Survival probability')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'T4_survival_curves.png', dpi=150)
    plt.close()

    # Log‑rank tests
    ref = data[data['group'] == 'B=High,RSSI=Mid']
    logrank_results = {}
    for g in main_groups:
        if g == 'B=High,RSSI=Mid':
            continue
        comp = data[data['group'] == g]
        if len(ref) == 0 or len(comp) == 0:
            continue
        lr = logrank_test(ref['duration'], comp['duration'], ref['event'], comp['event'])
        logrank_results[f'Mid vs {g}'] = {'p': lr.p_value}
    results['logrank'] = logrank_results

    # Cox PH (main effects only to avoid singularity)
    cox_data = data.copy()
    cox_data['High_B'] = (cox_data['B_level'] == 'High').astype(int)
    cox_data['RSSI_Mid'] = (cox_data['RSSI_level'] == 'Mid').astype(int)
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data[['duration', 'event', 'High_B', 'RSSI_Mid']],
                duration_col='duration', event_col='event')
        results['cox_summary'] = cph.summary.to_dict()
    except Exception as e:
        results['cox_error'] = str(e)

    with open(TABLE_DIR / 'T4_4D_survival.txt', 'w', encoding='utf-8') as f:
        f.write("SURVIVAL ANALYSIS RESULTS (DURATION FIXED)\n")
        f.write("-" * 50 + "\n")
        f.write("Log‑rank tests vs. B=High,RSSI=Mid:\n")
        for k, v in logrank_results.items():
            f.write(f"  {k}: p = {v['p']:.4f}\n")
        f.write("\nCox PH (High_B + RSSI_Mid):\n")
        if 'cox_summary' in results:
            f.write(pd.DataFrame(results['cox_summary']).to_string())
        elif 'cox_error' in results:
            f.write(f"Error: {results['cox_error']}\n")
    return results


# =============================================================================
# BLOCK 4E: BY CONFIGURATION
# =============================================================================
def block4E_by_config(df):
    """2×3 matrices stratified by configuration (C2, C3, C4)."""
    results = {}
    for cfg in ['C2', 'C3', 'C4']:
        sub = df[df['Configuration'] == cfg].dropna(subset=['B_level', 'RSSI_level', 'collapse_4Q'])
        if len(sub) == 0:
            continue
        pivot = sub.groupby(['B_level', 'RSSI_level'])['collapse_4Q'].mean().unstack()
        results[cfg] = pivot
        pivot.to_csv(TABLE_DIR / f'T4_4E_{cfg}_matrix.csv')
    return results


# =============================================================================
# BLOCK 4F: ABSENCE PATTERN (B = Q5)
# =============================================================================
def block4F_absence(df):
    """Test whether firms can remain stable in the (B=Q5, RSSI=Mid) region."""
    data = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].copy()
    data['B_Q5'] = (data['B_q5'] == 5).astype(int)
    q33, q67 = data['RSSI'].quantile(0.33), data['RSSI'].quantile(0.67)
    data['RSSI_Mid'] = (data['RSSI'] >= q33) & (data['RSSI'] <= q67)
    data = data.sort_values(['Ticker', 'period_end'])
    data['future_collapse'] = 0
    for lag in range(1, 7):
        data[f'c_{lag}'] = data.groupby('Ticker')['Configuration'].shift(-lag).isin(['C1', 'C6']).astype(int)
        data['future_collapse'] = data['future_collapse'] | data[f'c_{lag}']
    data['stable'] = (data['future_collapse'] == 0) & data['Configuration'].isin(['C2', 'C3', 'C4'])
    mask = data['B_Q5'] & data['RSSI_Mid'] & data['stable']
    obs = mask.sum()
    n = len(data)
    p_BQ5 = data['B_Q5'].mean()
    p_RM  = data['RSSI_Mid'].mean()
    p_stable = data['stable'].mean()
    exp = n * p_BQ5 * p_RM * p_stable
    try:
        from scipy.stats import binomtest
        p_val = binomtest(obs, n=n, p=p_BQ5 * p_RM * p_stable, alternative='less').pvalue
    except ImportError:
        p_val = stats.binom_test(obs, n=n, p=p_BQ5 * p_RM * p_stable, alternative='less')
    res = {'observed': obs, 'expected': exp, 'p_value': p_val}
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T4_4F_absence.csv', index=False)
    return res


# =============================================================================
# ACADEMIC REPORT GENERATION
# =============================================================================
def generate_report(results):
    """Write a comprehensive English report."""
    report_path = REPORT_DIR / 'T4_Joint_Collapse_Report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("TEST 4: JOINT COLLAPSE CONDITION – COMPLETE ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        # 4A
        f.write("4A. 2×3 MATRIX (B_level × RSSI_level)\n")
        f.write("-" * 50 + "\n")
        if results.get('4A') is not None:
            f.write(results['4A'].to_string(index=False) + "\n\n")
        else:
            f.write("No data available.\n\n")

        # 4B
        f.write("4B. 5×3 MATRIX (B_quantile × RSSI_level)\n")
        f.write("-" * 50 + "\n")
        if results.get('4B') is not None:
            f.write(results['4B'].to_string(index=False) + "\n\n")
        else:
            f.write("No data available.\n\n")

        # 4C
        f.write("4C. LOGISTIC REGRESSION (DUMMY VARIABLES)\n")
        f.write("-" * 50 + "\n")
        if results.get('4C'):
            for k, v in results['4C'].items():
                f.write(f"{k}: {v}\n")
        else:
            f.write("No data available.\n")
        f.write("\n")

        # 4D
        f.write("4D. SURVIVAL ANALYSIS (ACTUAL DURATION)\n")
        f.write("-" * 50 + "\n")
        if results.get('4D'):
            r4d = results['4D']
            if 'logrank' in r4d:
                f.write("Log‑rank tests vs. B=High,RSSI=Mid:\n")
                for k, v in r4d['logrank'].items():
                    f.write(f"  {k}: p = {v['p']:.4f}\n")
            if 'cox_summary' in r4d:
                f.write("\nCox PH summary:\n")
                f.write(pd.DataFrame(r4d['cox_summary']).to_string())
            elif 'cox_error' in r4d:
                f.write(f"Cox PH error: {r4d['cox_error']}\n")
        else:
            f.write("No data available.\n")
        f.write("\n")

        # 4E
        f.write("4E. STRATIFIED BY CONFIGURATION\n")
        f.write("-" * 50 + "\n")
        if results.get('4E'):
            for cfg, pivot in results['4E'].items():
                f.write(f"\n--- Configuration {cfg} ---\n")
                f.write(pivot.to_string() + "\n")
        else:
            f.write("No data available.\n")
        f.write("\n")

        # 4F
        f.write("4F. ABSENCE PATTERN (B = Q5)\n")
        f.write("-" * 50 + "\n")
        if results.get('4F'):
            r = results['4F']
            f.write(f"Observed stable observations in (B=Q5, RSSI=Mid): {r['observed']}\n")
            f.write(f"Expected under independence: {r['expected']:.2f}\n")
            f.write(f"One‑sided binomial p‑value: {r['p_value']:.4e}\n")
            if r['p_value'] < 0.05:
                f.write("=> Significantly fewer than expected; the region cannot remain stable.\n")
            else:
                f.write("=> No statistical evidence of absence.\n")
        else:
            f.write("No data available.\n")
    print(f"Report saved to: {report_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("TEST 4: JOINT COLLAPSE CONDITION (COMPLETE)")
    print("=" * 80)

    panel = load_full_panel()
    panel = add_collapse_windows(panel)
    panel_spec = panel[panel['Configuration'].isin(['C2', 'C3', 'C4'])].copy()
    print(f"Speculative observations: {len(panel_spec)}")
    panel_spec = assign_levels(panel_spec)

    results = {}
    print("\nBlock 4A..."); results['4A'] = block4A_2x3_matrix(panel_spec)
    print("Block 4B..."); results['4B'] = block4B_5x3_matrix(panel_spec)
    print("Block 4C..."); results['4C'] = block4C_logistic(panel_spec)
    print("Block 4D..."); results['4D'] = block4D_survival(panel_spec)
    print("Block 4E..."); results['4E'] = block4E_by_config(panel_spec)
    print("Block 4F..."); results['4F'] = block4F_absence(panel_spec)

    print("\nGenerating report...")
    generate_report(results)

    print("\nTest 4 complete.")


if __name__ == "__main__":
    main()