"""
STEP 8: SPECULATIVE FRAMEWORK CALCULATION (PURE MATH)
Focus: Strictly calculates core LTV metrics (V_Prod_base, E*, K_Pi', R_t, PDI_t) and Formal Gates.
No classification or labeling happens here.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('../Survey_config.yaml')
RAW_DATA_FOLDER = Path('../data/raw')
PROCESS_FOLDER = Path('../data/process')
BENCHMARK_FOLDER = Path('../data/processed')

def get_q_period(date):
    if pd.isna(date): return None
    return (pd.to_datetime(date) - pd.Timedelta(days=15)).to_period('Q')

def load_benchmark_lookup():
    lookup = {}
    if not BENCHMARK_FOLDER.exists(): return lookup
    for file_path in BENCHMARK_FOLDER.glob("*_benchmark_median.xlsx"):
        sector = file_path.stem.replace("_benchmark_median", "")
        try:
            df = pd.read_excel(file_path)
            if 's_baseline' in df.columns and 'period_end' in df.columns:
                df['period_end'] = pd.to_datetime(df['period_end'])
                for _, row in df.iterrows():
                    if 'Operating_Margin_median' in df.columns and 'period_end' in df.columns:
                        lookup[(sector, get_q_period(row['period_end']))] = row['Operating_Margin_median']
        except Exception as e:
            logger.error(f"  ⚠️ Error loading benchmark for {sector}: {e}")
    return lookup

def get_benchmark_margin(sector, period_end, lookup):
    target_q = get_q_period(period_end)
    if (sector, target_q) in lookup: return lookup[(sector, target_q)]
    candidates = [v for k, v in lookup.items() if k[0] == sector and k[1] <= target_q]
    return candidates[-1] if candidates else np.nan

def calculate_framework_metrics(df, sector, benchmark_lookup):
    df = df.copy()
    df['period_end'] = pd.to_datetime(df['period_end'])
    df = df.sort_values('period_end').reset_index(drop=True)

    rev_col = next((c for c in df.columns if str(c).lower() == 'revenue'), None)
    if rev_col and rev_col != 'Revenue':
        df.rename(columns={rev_col: 'Revenue'}, inplace=True)

    required_cols = ['Revenue', 'market_cap', 'KBrand']
    if not all(col in df.columns for col in required_cols): return None

    # Giữ nguyên Doanh thu gốc, không đắp số ảo
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0)
    is_productive = df['Revenue'] > 0  # Dấu hiệu vòng M-C-M'

    # Tìm cột Lợi nhuận (Operating Income) để làm Thặng dư cho nhóm M-M'
    op_inc_col = next((c for c in df.columns if 'operating' in str(c).lower() and 'income' in str(c).lower()), None)
    if not op_inc_col:
        op_inc_col = next((c for c in df.columns if 'operatingincome' in str(c).lower()), None)
    
    if op_inc_col:
        df[op_inc_col] = pd.to_numeric(df[op_inc_col], errors='coerce').fillna(0)
    else:
        df['Fallback_OpInc'] = 0.0
        op_inc_col = 'Fallback_OpInc'

    # 1. Margin Selection (Chỉ tính cho nhóm M-C-M')
    rnd = next((c for c in df.columns if 'research' in str(c).lower()), None)
    sga = next((c for c in df.columns if 'selling' in str(c).lower()), None)
    df['Opex_Ratio'] = np.where(is_productive, (df[rnd].fillna(0) + df[sga].fillna(0)) / df['Revenue'], 0.0) if rnd and sga else 0.0
    
    if 'Operating_Margin' not in df.columns:
        df['Operating_Margin'] = np.nan
        
    cost = df['CostOfRevenue'].fillna(0) if 'CostOfRevenue' in df.columns else 0
    fallback_margin = np.where(is_productive, (df['Revenue'] - cost) / df['Revenue'] - 0.2, np.nan)
    df['Operating_Margin'] = df['Operating_Margin'].fillna(pd.Series(fallback_margin))

    if 'Gross_Margin' not in df.columns:
        df['Gross_Margin'] = np.nan

    use_gross = (df['Operating_Margin'] < 0) & (df['Opex_Ratio'] > 0.30) & (df.get('Gross_Margin', -1) > 0)
    df['Selected_Margin'] = np.where(use_gross, df.get('Gross_Margin', df['Operating_Margin']), df['Operating_Margin'])
    
    # Cap an toàn cho nhóm sản xuất
    safe_margin = df['Selected_Margin'].clip(lower=None, upper=0.9999)

    # ==============================================================
    # 2. Base & Surplus Components (Tôn trọng tuyệt đối LTV)
    # ==============================================================
    df['Benchmark_Margin'] = [get_benchmark_margin(sector, d, benchmark_lookup) for d in df['period_end']]
    
    # M-M' không có sản xuất -> V_Prod_base = 0
    df['V_Prod_base'] = np.where(is_productive, df['Revenue'] * (1 - safe_margin.clip(lower=0)), 0.0)
    
    s_base_calc = np.where(df['Benchmark_Margin'].isna(), 
                           df['Revenue'] * safe_margin.clip(lower=0),
                           df['Revenue'] * np.minimum(safe_margin.clip(lower=0), df['Benchmark_Margin']))
    df['s_baseline_value'] = np.where(is_productive, s_base_calc, 0.0)
    
    # CHIẾM ĐOẠT THẶNG DƯ: Nhóm M-C-M' tính qua Margin. Nhóm M-M' lấy thẳng Lợi nhuận gán vào S_Surplus!
    s_surp_calc = np.where(df['Benchmark_Margin'].isna(), 0.0,
                           df['Revenue'] * np.maximum(0, safe_margin - df['Benchmark_Margin']))
    df['S_Surplus'] = np.where(is_productive, s_surp_calc, df[op_inc_col].clip(lower=0))
    
    df['s_total'] = df['s_baseline_value'] + df['S_Surplus']

    # 3. K_Pi' Extraction
    df['K_Pi_prime'] = df['market_cap'] - (df['V_Prod_base'] + df['s_total'] + df['KBrand'])
    
    # 4. E* Ratios (Tự động ra NaN đối với nhóm M-M' vì V_Prod_base = 0)
    vpb = df['V_Prod_base']
    df['E_star'] = np.where(vpb > 0, (df['market_cap'] - vpb) / vpb, np.nan)
    df['E_0'] = np.where(vpb > 0, df['s_baseline_value'] / vpb, np.nan)
    df['E_1'] = np.where(vpb > 0, df['S_Surplus'] / vpb, np.nan)
    df['E_2'] = np.where(vpb > 0, df['KBrand'] / vpb, np.nan)
    df['E_3'] = np.where(vpb > 0, df['K_Pi_prime'] / vpb, np.nan)
    
    # 5. Dynamics & Momentum
    df['K_Pi_prime_lag'] = df['K_Pi_prime'].shift(1)
    df['R_t'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'] != 0), 
        df['s_total'] / df['K_Pi_prime_lag'], 
        0.0  
    )
    df['dK_Pi_prime'] = df['K_Pi_prime'].diff().fillna(0.0)
    df['dK_Pi_prime_pct'] = np.where(
        df['K_Pi_prime_lag'].notna() & (df['K_Pi_prime_lag'].abs() > 0), 
        df['dK_Pi_prime'] / df['K_Pi_prime_lag'].abs(), 
        0.0
    )
    df['PGR_t'] = np.where(is_productive, df['V_Prod_base'].pct_change().fillna(0.0), 0.0)
    
    denominator = df['dK_Pi_prime'].abs() + df['s_total']
    df['PDI_t'] = np.where(
        (denominator != 0) & denominator.notna(), 
        df['s_total'] / denominator, 
        0.0
    )

    # 6. Formal Gates
    df['Gate_C1'] = np.where(df['V_Prod_base'] > 0, (df['K_Pi_prime'] / df['V_Prod_base']) > 0, False)
    df['Gate_C2'] = (df['E_3'] > (df['E_0'] + df['E_1'] + df['E_2'])).astype(bool)
    df['Gate_C3'] = (df['R_t'] < 1.0).astype(bool)
    
    standard_speculative = df['Gate_C1'] & df['Gate_C2'] & df['Gate_C3']
    
    # NHẬN DIỆN TƯ BẢN GIẢ: Công ty không có sản xuất (Revenue=0) tự động mở cửa vào Framework Đầu cơ
    extreme_speculation = (df['Revenue'] <= 0) & (df['market_cap'] > 0)
    df['Speculative_Regime'] = np.where(extreme_speculation, True, standard_speculative)
    
    return df

def main():
    logger.info("="*70)
    logger.info("FRAMEWORK METRICS CALCULATION (THEORETICALLY PURE)")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Config missing: {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    benchmark_lookup = load_benchmark_lookup()
    stats = {'processed': 0, 'skipped_invalid': 0, 'file_not_found': 0}

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nProcessing Sector: {sector_name}")
        sector_process_dir = PROCESS_FOLDER / sector_name
        sector_process_dir.mkdir(parents=True, exist_ok=True)
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active': continue
            
            ticker = company['ticker']
            fpath = RAW_DATA_FOLDER / sector_name / f"{ticker}_raw.csv"
            out_fpath = sector_process_dir / f"{ticker}_processed.csv"
            
            if fpath.exists():
                try:
                    df_raw = pd.read_csv(fpath)
                    df_out = calculate_framework_metrics(df_raw, sector_name, benchmark_lookup)
                    
                    if df_out is not None:
                        df_out.to_csv(out_fpath, index=False, encoding='utf-8-sig')
                        stats['processed'] += 1
                        logger.debug(f"  ✅ {ticker:6} | Math computed.")
                    else:
                        if out_fpath.exists(): out_fpath.unlink()
                        logger.warning(f"  ⚠️ {ticker:6} | Skipped.")
                        stats['skipped_invalid'] += 1
                except Exception as e:
                     logger.error(f"  ❌ {ticker:6} | Error: {e}")
                     stats['skipped_invalid'] += 1
            else:
                stats['file_not_found'] += 1
                
    logger.info("\n" + "=" * 70)
    logger.info("CALCULATION COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
