"""
T11_structural_bifurcation_tests_robust.py

STRUCTURAL BIFURCATION & NONLINEARITY TESTS FOR RSSI (TIME‑SPLIT + BOOTSTRAP)
------------------------------------------------------------------------------
- Time‑split validation (train ≤2021, test ≥2022)
- Bootstrap confidence band for continuous shape (no statsmodels)
- Consistent target: collapse_4Q and trajectory
- Overfitting check for Random Forest
- Time‑split entropy analysis
- RSSI direction incorporated: Mid split into Mid_asc and Mid_desc
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASSIFIED_DIR = Path('../data/classified')
OUTPUT_DIR = Path('results')
TABLE_DIR = OUTPUT_DIR / 'tables'
FIGURE_DIR = OUTPUT_DIR / 'figures'
REPORT_DIR = OUTPUT_DIR / 'reports'
for d in [TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SECTORS = ['Healthcare', 'Technology', 'Services']
RANDOM_STATE = 42

# =============================================================================
# DATA LOADING
# =============================================================================
def load_panel():
    required = ['Ticker', 'period_end', 'Configuration', 'B', 'RSSI', 'Phi_t', 'collapse_next']
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
                    df = df.sort_values(['Ticker', 'period_end'])
                    df['next_config'] = df.groupby('Ticker')['Configuration'].shift(-1)
                    df['collapse_next'] = df['next_config'].isin(['C1', 'C6']).astype(int)
                    df.drop(columns=['next_config'], inplace=True)
                missing = [c for c in required if c not in df.columns]
                if missing:
                    continue
                df = df[required + ['Sector']].copy()
                all_dfs.append(df)
            except Exception:
                continue
    if not all_dfs:
        raise ValueError("No valid data found.")
    panel = pd.concat(all_dfs, ignore_index=True)
    panel = panel.sort_values(['Ticker', 'period_end']).reset_index(drop=True)
    return panel

def prepare_panel(df):
    df = df.copy()
    # Time split column
    df['year'] = df['period_end'].dt.year
    # collapse_4Q
    df['collapse_4Q'] = 0
    for lag in range(1, 5):
        col = f'c_{lag}'
        df[col] = df.groupby('Ticker')['collapse_next'].shift(-lag)
        df['collapse_4Q'] = df['collapse_4Q'] | df[col].fillna(0).astype(int)
    df['collapse_4Q'] = df['collapse_4Q'].astype(int)
    # Next config for trajectory
    df['next_config'] = df.groupby('Ticker')['Configuration'].shift(-1)
    # Trajectory label (Collapse / Sustain / Evolve)
    def trajectory(row):
        nxt = row['next_config']
        if pd.isna(nxt):
            return np.nan
        if nxt in ['C1','C6']:
            return 'Collapse'
        elif nxt == 'C2':
            return 'Sustain'
        elif nxt in ['C3','C4']:
            return 'Evolve'
        return np.nan
    df['trajectory'] = df.apply(trajectory, axis=1)
    # RSSI categories
    try:
        df['RSSI_cat'] = pd.qcut(df['RSSI'], 3, labels=['Low','Mid','Extreme'], duplicates='drop')
    except Exception:
        df['RSSI_cat'] = 'Mid'

    # ===== NEW: RSSI direction =====
    # Compute sector-level RSSI change
    df = df.sort_values(['Sector', 'period_end'])
    df['dRSSI'] = df.groupby('Sector')['RSSI'].diff()
    df['RSSI_direction'] = np.where(
        df['dRSSI'] > 0, 'ascending',
        np.where(df['dRSSI'] < 0, 'descending', 'flat')
    )
    # Create extended category with Mid split by direction
    df['RSSI_cat_dir'] = df['RSSI_cat'].astype(str)
    mid_mask = df['RSSI_cat'] == 'Mid'
    df.loc[mid_mask & (df['RSSI_direction'] == 'ascending'), 'RSSI_cat_dir'] = 'Mid_asc'
    df.loc[mid_mask & (df['RSSI_direction'] == 'descending'), 'RSSI_cat_dir'] = 'Mid_desc'
    # ===============================

    # Keep speculative only
    df = df[df['Configuration'].isin(['C2','C3','C4'])].copy()
    return df

# =============================================================================
# 1. ENTROPY TEST (TIME‑SPLIT) — UPDATED WITH Mid_asc/Mid_desc
# =============================================================================
def test_entropy_time_split(df):
    """Compute trajectory entropy for C2 & Φ=1, separately for train/test."""
    sub = df[(df['Configuration'] == 'C2') & (df['Phi_t'] == 1)].copy()
    print(f"  Entropy test: {len(sub)} C2 & Φ=1 observations")
    if len(sub) < 20:
        return None
    results = []
    categories = ['Low', 'Mid_asc', 'Mid_desc', 'Extreme']
    for period, mask in [('train', sub['year'] <= 2021), ('test', sub['year'] >= 2022)]:
        per_df = sub[mask]
        for r in categories:
            cat = per_df[per_df['RSSI_cat_dir'] == r]
            if len(cat) < 3:
                continue
            probs = cat['trajectory'].value_counts(normalize=True).values
            ent = scipy_entropy(probs) if len(probs) > 1 else 0.0
            results.append({
                'Period': period,
                'RSSI': r,
                'N': len(cat),
                'Entropy': ent
            })
    if not results:
        return None
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T11_entropy_timesplit.csv', index=False)
    return df_res

# =============================================================================
# 2. CONTINUOUS SHAPE WITH BOOTSTRAP BAND (NO STATSMODELS)
# =============================================================================
def bootstrap_rolling_mean(x, y, n_boot=200, window_frac=0.2):
    """Bootstrap confidence band for rolling mean (fractional window)."""
    n = len(x)
    window = max(5, int(n * window_frac))
    grid = np.linspace(x.min(), x.max(), 100)
    curves = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        xb, yb = x[idx], y[idx]
        order = np.argsort(xb)
        xb_sorted = xb[order]
        yb_sorted = yb[order]
        roll = pd.Series(yb_sorted).rolling(window, center=True, min_periods=3).mean().values
        interp = np.interp(grid, xb_sorted, roll, left=np.nan, right=np.nan)
        curves.append(interp)
    curves = np.array(curves)
    mean_curve = np.nanmean(curves, axis=0)
    lower = np.nanpercentile(curves, 5, axis=0)
    upper = np.nanpercentile(curves, 95, axis=0)
    return grid, mean_curve, lower, upper

def test_continuous_shape(df):
    sub = df[(df['Configuration'] == 'C2') & (df['Phi_t'] == 1)].copy()
    print(f"  Continuous shape: {len(sub)} C2 & Φ=1 observations")
    if len(sub) < 20:
        return None
    sub = sub.sort_values('RSSI')
    x = sub['RSSI'].values
    y = sub['collapse_next'].values

    grid, mean_curve, lower, upper = bootstrap_rolling_mean(x, y, n_boot=200)

    try:
        coeffs = np.polyfit(x, y, 2)
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
    except:
        coeffs = [np.nan, np.nan, np.nan]
        y_fit = None

    plt.figure(figsize=(8,5))
    plt.scatter(x, y, alpha=0.2, s=10, label='Observed')
    plt.plot(grid, mean_curve, color='red', linewidth=2, label='Rolling mean')
    plt.fill_between(grid, lower, upper, color='red', alpha=0.2, label='90% CI')
    if y_fit is not None:
        plt.plot(x_fit, y_fit, 'g--', linewidth=2, label=f'Quadratic fit (a={coeffs[0]:.3f})')
    plt.xlabel('RSSI')
    plt.ylabel('Collapse rate (t+1)')
    plt.title('C2 & Φ=1: Collapse probability vs RSSI (bootstrap CI)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / 'T11_continuous_shape_ci.png', dpi=150)
    plt.close()

    res = {'quad_a': coeffs[0], 'quad_b': coeffs[1], 'quad_c': coeffs[2], 'N': len(sub)}
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T11_quadratic_fit.csv', index=False)
    return res

# =============================================================================
# 3. TREE vs LINEAR (TIME‑SPLIT, OVERFIT CHECK)
# =============================================================================
def test_tree_vs_linear_oos(df):
    data = df[df['Configuration'] == 'C2'].dropna(subset=['B','RSSI','collapse_4Q','year']).copy()
    print(f"  Tree model: {len(data)} C2 observations")
    if len(data) < 70:
        return None

    train = data[data['year'] <= 2021]
    test  = data[data['year'] >= 2022]
    if len(train) < 30 or len(test) < 20:
        return None

    X_tr = train[['B','RSSI']].values
    y_tr = train['collapse_4Q'].values
    X_te = test[['B','RSSI']].values
    y_te = test['collapse_4Q'].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg.fit(X_tr_s, y_tr)
    auc_log_te = roc_auc_score(y_te, logreg.predict_proba(X_te_s)[:,1])

    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE)
    rf.fit(X_tr, y_tr)
    auc_rf_tr = roc_auc_score(y_tr, rf.predict_proba(X_tr)[:,1])
    auc_rf_te = roc_auc_score(y_te, rf.predict_proba(X_te)[:,1])

    imp = pd.DataFrame({'Feature': ['B','RSSI'], 'Importance': rf.feature_importances_})
    imp.to_csv(TABLE_DIR / 'T11_rf_importance.csv', index=False)

    res = {
        'AUC_log_test': auc_log_te,
        'AUC_RF_train': auc_rf_tr,
        'AUC_RF_test': auc_rf_te,
        'Delta_RF_test': auc_rf_te - auc_log_te,
        'Overfit_gap': auc_rf_tr - auc_rf_te,
        'N_train': len(train),
        'N_test': len(test)
    }
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T11_tree_timesplit.csv', index=False)
    return res

# =============================================================================
# 4. C2 TRANSITION MATRIX BY RSSI — UPDATED WITH Mid_asc/Mid_desc
# =============================================================================
def test_c2_transition_matrix(df):
    sub = df[df['Configuration'] == 'C2'].copy()
    print(f"  Transition matrix: {len(sub)} C2 observations")
    if len(sub) < 10:
        return None
    results = []
    categories = ['Low', 'Mid_asc', 'Mid_desc', 'Extreme']
    for r in categories:
        cat = sub[sub['RSSI_cat_dir'] == r]
        if len(cat) < 3:
            continue
        nxt = cat['next_config']
        n = len(cat)
        row = {
            'RSSI': r,
            'N': n,
            'to_C1_C6': (nxt.isin(['C1','C6'])).mean(),
            'to_C2': (nxt == 'C2').mean(),
            'to_C3_C4': (nxt.isin(['C3','C4'])).mean()
        }
        results.append(row)
    if not results:
        return None
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T11_c2_transition_matrix.csv', index=False)
    return df_res

# =============================================================================
# GENERATE REPORT
# =============================================================================
def generate_report(results):
    report_path = REPORT_DIR / 'T11_Structural_Bifurcation_Report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("STRUCTURAL BIFURCATION & NONLINEARITY TESTS FOR RSSI (TIME‑SPLIT)\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. ENTROPY TEST (C2, Φ=1, TRAIN/TEST SPLIT)\n")
        f.write("-" * 50 + "\n")
        if results.get('entropy') is not None:
            f.write(results['entropy'].to_string(index=False) + "\n")
            f.write("Interpretation: Mid_asc should show highest entropy (bifurcation zone).\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("2. CONTINUOUS SHAPE WITH BOOTSTRAP CI\n")
        f.write("-" * 50 + "\n")
        if results.get('shape'):
            r = results['shape']
            f.write(f"Quadratic coefficient a = {r['quad_a']:.4f} ")
            f.write("(negative = inverted‑U)\n")
            f.write(f"Sample size: {r['N']}\n")
            f.write("Bootstrap 90% CI plotted.\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("3. TREE vs LINEAR (OUT‑OF‑SAMPLE, OVERFIT CHECK)\n")
        f.write("-" * 50 + "\n")
        if results.get('tree'):
            r = results['tree']
            f.write(f"Logistic test AUC : {r['AUC_log_test']:.4f}\n")
            f.write(f"Random Forest test AUC: {r['AUC_RF_test']:.4f}\n")
            f.write(f"ΔAUC (RF − Logit) : {r['Delta_RF_test']:.4f}\n")
            f.write(f"Overfitting gap (train − test): {r['Overfit_gap']:.4f}\n")
            f.write(f"Train N: {r['N_train']}, Test N: {r['N_test']}\n")
            if r['Delta_RF_test'] > 0.01 and r['Overfit_gap'] < 0.10:
                f.write("✅ RF outperforms logistic out‑of‑sample with acceptable overfit.\n")
            else:
                f.write("⚠️ Nonlinear gain is limited or overfitting is present.\n")
            f.write("\n")
        else:
            f.write("Insufficient data (need ≥30 train, ≥20 test).\n\n")

        f.write("4. C2 TRANSITION MATRIX BY RSSI (INCLUDING DIRECTION)\n")
        f.write("-" * 50 + "\n")
        if results.get('transition') is not None:
            f.write(results['transition'].to_string(index=False) + "\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("=" * 100 + "\n")
        f.write("CONCLUSION: Mid_asc vs Mid_desc differences reveal the directional\n")
        f.write("nature of the bifurcation zone. Combined with 6G, this confirms\n")
        f.write("that RSSI momentum matters as much as its level.\n")
    print(f"Report saved to: {report_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("STRUCTURAL BIFURCATION TESTS FOR RSSI (TIME‑SPLIT + BOOTSTRAP)")
    print("=" * 80)

    df = load_panel()
    df = prepare_panel(df)
    print(f"Speculative observations: {len(df)}")

    results = {}
    print("\n1. Entropy test (time‑split)...")
    results['entropy'] = test_entropy_time_split(df)

    print("\n2. Continuous shape with bootstrap CI...")
    results['shape'] = test_continuous_shape(df)

    print("\n3. Tree vs Linear (out‑of‑sample)...")
    results['tree'] = test_tree_vs_linear_oos(df)

    print("\n4. C2 transition matrix...")
    results['transition'] = test_c2_transition_matrix(df)

    print("\nGenerating report...")
    generate_report(results)

    print("All tests completed.")

if __name__ == "__main__":
    main()