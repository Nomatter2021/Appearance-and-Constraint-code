"""
STEP 9: THEORETICAL CLASSIFICATION
Focus: Applies Proposition 7 rules to classify structural states (Normal, C1-C6) 
and assigns linguistic phases based on the math computed in Step 08.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path('../Survey_config.yaml')
PROCESS_FOLDER = Path('../data/process')
CLASSIFIED_FOLDER = Path('../data/classified')

def classify_state(row):
    rt = row.get('R_t')
    dk = row.get('dK_Pi_prime')
    s = row.get('s_total')
    dkp = row.get('dK_Pi_prime_pct')
    k_pi_prime = row.get('K_Pi_prime')
    k_pi_prime_lag = row.get('K_Pi_prime_lag')
    
    # Doanh thu gốc là thước đo tuyệt đối của nền sản xuất thực (M-C-M')
    rev = row.get('Revenue', 0.0)

    c1 = row.get('Gate_C1', False)
    c2 = row.get('Gate_C2', False)
    speculative_step8 = row.get('Speculative_Regime', False)

    if pd.isna(k_pi_prime_lag):
        return 'N/A'

    if pd.notna(k_pi_prime) and k_pi_prime <= 0:
        return 'Normal'

    # GATE KEEPER: Bỏ chặn Gate_C3
    if not ((c1 and c2) or speculative_step8):
        return 'Normal'

    # BRANCH A: Không có thặng dư (M-M' lỗ hoặc M-C-M' lỗ)
    if s <= 0:
        if dk <= 0: 
            return 'C1' if dkp <= -0.15 else 'C6' 
        else:
            return 'C2'                           
            
    # BRANCH B: Có thặng dư 
    else:
        if dk > 0:
            return 'C3'                           
        else:
            # ==========================================================
            # CHỐT CHẶN C5: Phải hoàn thành vòng M-C-M' (Revenue > 0)
            # Tư bản giả (M-M') vĩnh viễn bị giam ở C4, bất kể thặng dư lớn cỡ nào!
            # ==========================================================
            if rt >= 0.999 and rev > 0:
                return 'C5'                       
            else:
                return 'C4'                       
                
    return 'Other'

def classify_company_data(df, ticker):
    df = df.copy()
    df['Ticker'] = ticker

    core_cols = ['E_3', 'R_t', 'PDI_t', 'PGR_t']
    for col in core_cols:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
    if 'R_t' in df.columns and 's_total' in df.columns:
        df['R_t'] = np.where(df['s_total'] <= 0, 0.0, df['R_t'])

    df['Configuration'] = df.apply(classify_state, axis=1)

    # Phân biệt chế độ Sản xuất và Thu tô dựa vào biến số Doanh thu
    df['Regime_Label'] = np.where(
        df['Configuration'] == 'Normal', 
        np.where(df.get('Revenue', 0) <= 0, 'Rentier_Regime', 'Normal_Regime'),   
        np.where(df['Configuration'] == 'N/A', 'Unknown', 'Speculative_Regime')
    )

    conditions = [
        (df['Configuration'] == 'Normal') & (df.get('Revenue', 0) > 0),   
        (df['Configuration'] == 'Normal') & (df.get('Revenue', 0) <= 0),  
        df['Configuration'].isin(['C2', 'C3']),   
        df['Configuration'] == 'C4',              
        df['Configuration'] == 'C1',              
        df['Configuration'] == 'C6',              
        df['Configuration'] == 'C5'               
    ]
    choices = [
        'Normal_Production',           
        'Steady_Rent_Seeking',         
        'Accumulation_Phase',          
        'Partial_Absorption',          
        'Redistributive_Collapse',     
        'Silent_Redistribution',       
        'Productive_Discharge'         
    ]
    df['Phase'] = np.select(conditions, choices, default='Other')

    df_clean = df[df['Configuration'] != 'N/A'].copy()
    df_clean = df_clean.reset_index(drop=True)

    return df_clean

def main():
    logger.info("="*70)
    logger.info("THEORETICAL CLASSIFICATION (C1-C6 & PHASES)")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Input config missing: {CONFIG_FILE}")
        return
        
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {'classified': 0, 'errors': 0}

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nSector: {sector_name}")
        sector_classified_dir = CLASSIFIED_FOLDER / sector_name
        sector_classified_dir.mkdir(parents=True, exist_ok=True)
        
        for company in sector_info.get('companies', []):
            if company.get('status') != 'active': continue
            
            ticker = company['ticker']
            fpath = PROCESS_FOLDER / sector_name / f"{ticker}_processed.csv"
            
            if fpath.exists():
                try:
                    df_processed = pd.read_csv(fpath)
                    df_classified = classify_company_data(df_processed, ticker)
                    
                    out_path = sector_classified_dir / f"{ticker}_classified.csv"
                    df_classified.to_csv(out_path, index=False, encoding='utf-8-sig')
                    stats['classified'] += 1
                    logger.debug(f"  ✅ {ticker:6} | Classification applied.")
                except Exception as e:
                    logger.error(f"  ❌ {ticker:6} | Error: {e}")
                    stats['errors'] += 1

    logger.info("\n" + "="*70)
    logger.info(f"🎯 COMPLETE! Classified {stats['classified']} files.")
    logger.info("="*70)

if __name__ == "__main__":
    main()
