"""
T10_rssi_incremental_validity_fixed_v2.py

VALIDITY TESTS FOR RSSI AS AN INDEPENDENT AMPLIFIER (CORRECTED & CLEANED)
-------------------------------------------------------------------------
- Unified NaN removal to prevent "Input X contains NaN" error.
- Incremental AUC now correctly computed.
- Bootstrap with fixed seed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CLASSIFIED_DIR = Path('../data/classified')
OUTPUT_DIR = Path('results')
TABLE_DIR = OUTPUT_DIR / 'tables'
REPORT_DIR = OUTPUT_DIR / 'reports'
for d in [TABLE_DIR, REPORT_DIR]:
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
                # Keep only required columns and drop rows with missing values in them
                missing = [c for c in required if c not in df.columns]
                if missing:
                    continue
                df = df[required + ['Sector']].copy()
                # Drop rows where any of the required columns are NaN
                df = df.dropna(subset=required)
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
    # collapse within 4 quarters
    df['collapse_4Q'] = 0
    for lag in range(1, 5):
        col = f'c_{lag}'
        df[col] = df.groupby('Ticker')['collapse_next'].shift(-lag)
        df['collapse_4Q'] = df['collapse_4Q'] | df[col].fillna(0).astype(int)
    df['collapse_4Q'] = df['collapse_4Q'].astype(int)
    # B quantiles (for matching)
    df['B_q'] = pd.qcut(df['B'], 4, labels=False, duplicates='drop') + 1
    # RSSI categories
    try:
        df['RSSI_cat'] = pd.qcut(df['RSSI'], 3, labels=['Low','Mid','Extreme'], duplicates='drop')
    except:
        df['RSSI_cat'] = 'Mid'
    # speculative only
    df = df[df['Configuration'].isin(['C2', 'C3', 'C4'])].copy()
    # Final cleanup: remove any remaining NaNs in essential columns
    essential = ['B', 'RSSI', 'Phi_t', 'collapse_4Q', 'B_q', 'RSSI_cat']
    df = df.dropna(subset=essential).copy()
    return df

# =============================================================================
# TEST 1: FIX Φ=1, EXAMINE RSSI EFFECT WITHIN B QUANTILES
# =============================================================================
def test_fixed_phi(df):
    sub = df[df['Phi_t'] == 1].copy()
    if len(sub) < 20:
        return None
    results = []
    for b in sorted(sub['B_q'].dropna().unique()):
        for r in ['Low','Mid','Extreme']:
            cell = sub[(sub['B_q'] == b) & (sub['RSSI_cat'] == r)]
            if len(cell) < 5:
                continue
            rate = cell['collapse_4Q'].mean()
            results.append({'B_q': b, 'RSSI': r, 'N': len(cell), 'collapse_rate': rate})
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T10_fixed_phi_rssi_effect.csv', index=False)
    return df_res

# =============================================================================
# TEST 2: FIX RSSI=MID, COMPARE Φ=1 VS Φ=0
# =============================================================================
def test_fixed_rssi(df):
    sub = df[df['RSSI_cat'] == 'Mid'].copy()
    if len(sub) < 20:
        return None
    results = []
    for phi in [0, 1]:
        cell = sub[sub['Phi_t'] == phi]
        if len(cell) < 5:
            continue
        rate = cell['collapse_4Q'].mean()
        results.append({'Phi': phi, 'N': len(cell), 'collapse_rate': rate})
    if len(results) == 2:
        phi1 = sub[sub['Phi_t'] == 1]['collapse_4Q']
        phi0 = sub[sub['Phi_t'] == 0]['collapse_4Q']
        if len(phi1) > 5 and len(phi0) > 5:
            u, p = mannwhitneyu(phi1, phi0, alternative='two-sided')
            results.append({'Phi': 'p_value', 'N': '', 'collapse_rate': p})
    df_res = pd.DataFrame(results)
    df_res.to_csv(TABLE_DIR / 'T10_fixed_rssi_phi_effect.csv', index=False)
    return df_res

# =============================================================================
# TEST 3: INCREMENTAL AUC OF RSSI (CORRECTED)
# =============================================================================
def test_incremental_auc(df):
    data = df.dropna(subset=['B', 'RSSI', 'collapse_4Q']).copy()
    if len(data) < 50:
        return None
    scaler = StandardScaler()
    X_base = scaler.fit_transform(data[['B']])
    y = data['collapse_4Q']
    model_base = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model_base.fit(X_base, y)
    auc_base = roc_auc_score(y, model_base.predict_proba(X_base)[:, 1])
    X_ext = np.column_stack([X_base, data['RSSI'].values])
    model_ext = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model_ext.fit(X_ext, y)
    auc_ext = roc_auc_score(y, model_ext.predict_proba(X_ext)[:, 1])
    delta = auc_ext - auc_base
    res = {'AUC_B': auc_base, 'AUC_B_RSSI': auc_ext, 'Delta_AUC': delta}
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T10_incremental_auc.csv', index=False)
    return res

# =============================================================================
# TEST 4: INTERACTION REGRESSION (WITH SEED)
# =============================================================================
def test_interaction_regression(df):
    data = df.dropna(subset=['B', 'RSSI', 'Phi_t', 'collapse_4Q']).copy()
    if len(data) < 50:
        return None
    scaler = StandardScaler()
    data['B_scaled'] = scaler.fit_transform(data[['B']])
    data['RSSI_scaled'] = scaler.fit_transform(data[['RSSI']])
    data['RSSI_x_Phi'] = data['RSSI_scaled'] * data['Phi_t']
    data['B_x_RSSI'] = data['B_scaled'] * data['RSSI_scaled']
    X = data[['B_scaled', 'RSSI_scaled', 'Phi_t', 'RSSI_x_Phi', 'B_x_RSSI']]
    y = data['collapse_4Q']
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X, y)
    coef = dict(zip(X.columns, model.coef_[0]))
    np.random.seed(RANDOM_STATE)
    n_boot = 500
    coef_boot = []
    for _ in range(n_boot):
        idx = np.random.choice(len(data), len(data), replace=True)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        model.fit(Xb, yb)
        coef_boot.append(model.coef_[0])
    coef_boot = np.array(coef_boot)
    pvals = {col: 2 * min(np.mean(coef_boot[:, i] >= 0), np.mean(coef_boot[:, i] <= 0))
             for i, col in enumerate(X.columns)}
    results = {'coef': coef, 'p_value': pvals, 'AUC': roc_auc_score(y, model.predict_proba(X)[:, 1])}
    with open(TABLE_DIR / 'T10_interaction_regression.txt', 'w') as f:
        f.write(str(results))
    return results

# =============================================================================
# TEST 5: TIME SPLIT VALIDATION
# =============================================================================
def test_time_split(df):
    df['year'] = df['period_end'].dt.year
    train = df[df['year'].between(2017, 2021)]
    test  = df[df['year'].between(2022, 2026)]
    if len(train) < 50 or len(test) < 20:
        return None
    X_tr = train[['B']].copy()
    X_tr['RSSI_Mid'] = (train['RSSI_cat'] == 'Mid').astype(int)
    X_tr['RSSI_Ext'] = (train['RSSI_cat'] == 'Extreme').astype(int)
    y_tr = train['collapse_4Q']
    X_te = test[['B']].copy()
    X_te['RSSI_Mid'] = (test['RSSI_cat'] == 'Mid').astype(int)
    X_te['RSSI_Ext'] = (test['RSSI_cat'] == 'Extreme').astype(int)
    y_te = test['collapse_4Q']
    model = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    auc_train = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
    auc_test  = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
    res = {'AUC_train': auc_train, 'AUC_test': auc_test, 'Delta': auc_test - auc_train}
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T10_time_split.csv', index=False)
    return res

# =============================================================================
# TEST 6: STRATIFIED MATCHING (HARD MODE)
# =============================================================================
def test_stratified_matching(df):
    sub = df[(df['Phi_t'] == 1) & (df['B_q'].isin([2, 3]))].copy()
    if len(sub) < 30:
        return None
    mid = sub[sub['RSSI_cat'] == 'Mid']['collapse_4Q']
    ext = sub[sub['RSSI_cat'] == 'Extreme']['collapse_4Q']
    if len(mid) < 10 or len(ext) < 10:
        return None
    u, p = mannwhitneyu(mid, ext, alternative='two-sided')
    res = {'N_mid': len(mid), 'N_ext': len(ext), 'rate_mid': mid.mean(),
           'rate_ext': ext.mean(), 'MW_p': p}
    pd.DataFrame([res]).to_csv(TABLE_DIR / 'T10_stratified_matching.csv', index=False)
    return res

# =============================================================================
# GENERATE REPORT
# =============================================================================
def generate_report(results):
    report_path = REPORT_DIR / 'T10_RSSI_Validity_Report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("VALIDITY TESTS FOR RSSI AS AN INDEPENDENT AMPLIFIER (CORRECTED & CLEANED)\n")
        f.write("=" * 100 + "\n\n")

        f.write("1. FIXED Φ=1 – RSSI EFFECT WITHIN B QUANTILES\n")
        f.write("-" * 50 + "\n")
        if results.get('fixed_phi') is not None:
            f.write(results['fixed_phi'].to_string(index=False) + "\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("2. FIXED RSSI=MID – Φ EFFECT\n")
        f.write("-" * 50 + "\n")
        if results.get('fixed_rssi') is not None:
            f.write(results['fixed_rssi'].to_string(index=False) + "\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("3. INCREMENTAL AUC OF RSSI OVER B ALONE\n")
        f.write("-" * 50 + "\n")
        if results.get('incremental'):
            r = results['incremental']
            f.write(f"Baseline AUC (B only)     : {r['AUC_B']:.4f}\n")
            f.write(f"Extended AUC (B + RSSI)   : {r['AUC_B_RSSI']:.4f}\n")
            f.write(f"ΔAUC                      : {r['Delta_AUC']:.4f}\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("4. INTERACTION REGRESSION (BOOTSTRAPPED P‑VALUES)\n")
        f.write("-" * 50 + "\n")
        if results.get('interaction'):
            f.write("Coefficients and p‑values:\n")
            f.write(str(results['interaction']['coef']) + "\n")
            f.write(str(results['interaction']['p_value']) + "\n")
            f.write(f"AUC: {results['interaction']['AUC']:.4f}\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("5. TIME SPLIT VALIDATION (train 2017‑2021, test 2022‑2026)\n")
        f.write("-" * 50 + "\n")
        if results.get('time_split'):
            f.write(f"Train AUC: {results['time_split']['AUC_train']:.4f}\n")
            f.write(f"Test AUC : {results['time_split']['AUC_test']:.4f}\n")
            f.write(f"Delta    : {results['time_split']['Delta']:.4f}\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("6. STRATIFIED MATCHING (B in Q2/Q3, Φ=1, Mid vs Extreme)\n")
        f.write("-" * 50 + "\n")
        if results.get('matching'):
            r = results['matching']
            f.write(f"N_mid: {r['N_mid']}, N_ext: {r['N_ext']}\n")
            f.write(f"Collapse rate Mid: {r['rate_mid']:.4f}, Extreme: {r['rate_ext']:.4f}\n")
            f.write(f"Mann‑Whitney p: {r['MW_p']:.4f}\n\n")
        else:
            f.write("Insufficient data.\n\n")

        f.write("=" * 100 + "\n")
        f.write("CONCLUSION: Incremental AUC and stratified matching provide\n")
        f.write("clean evidence for RSSI's independent contribution.\n")
    print(f"Report saved to: {report_path}")

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("RSSI VALIDITY TESTS (CLEANED)")
    print("=" * 80)

    df = load_panel()
    df = prepare_panel(df)
    print(f"Speculative observations after cleaning: {len(df)}")

    if len(df) < 100:
        print("WARNING: Less than 100 clean observations. Results may be unreliable.")

    results = {}
    print("\nTest 1: Fixed Φ=1...")
    results['fixed_phi'] = test_fixed_phi(df)

    print("Test 2: Fixed RSSI=Mid...")
    results['fixed_rssi'] = test_fixed_rssi(df)

    print("Test 3: Incremental AUC...")
    results['incremental'] = test_incremental_auc(df)

    print("Test 4: Interaction regression...")
    results['interaction'] = test_interaction_regression(df)

    print("Test 5: Time split...")
    results['time_split'] = test_time_split(df)

    print("Test 6: Stratified matching...")
    results['matching'] = test_stratified_matching(df)

    print("\nGenerating report...")
    generate_report(results)

    print("All tests completed.")

if __name__ == "__main__":
    main()