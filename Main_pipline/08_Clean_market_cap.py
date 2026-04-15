"""
STEP 6: MARKET CAP DATA CLEANING & CONTINUITY FILTER
Logic: 
1. Inflation Bias Filter: Detects if historical shares are flat (fallback used) and deletes them.
2. Continuity Filter: Scans time series to find the longest continuous sequence of quarters.
   Compliance with Paper (Step 6, Pg 39): Minimum 6 consecutive quarters required.
CRITICAL: Updates 'Survey_config.yaml' status so downstream scripts skip deleted files.
"""

import pandas as pd
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_FILE = Path("../Survey_config.yaml")
RAW_DATA_DIR = Path("../data/raw")
MIN_CONSECUTIVE_QUARTERS = 6  # Fixed to align strictly with Paper Step 6

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe: remove duplicate columns and standardize names."""
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    # Standardize price column
    price_cols = [c for c in df.columns if 'price_at_period' in c.lower()]
    if len(price_cols) > 1:
        best_price = max(price_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in price_cols if c != best_price])
        df.rename(columns={best_price: 'price_at_period_end'}, inplace=True)
    elif len(price_cols) == 1:
        df.rename(columns={price_cols[0]: 'price_at_period_end'}, inplace=True)
    
    # Standardize shares column
    shares_cols = [c for c in df.columns if 'shares_outstanding' in c.lower()]
    if len(shares_cols) > 1:
        best_shares = max(shares_cols, key=lambda c: df[c].notna().sum())
        df = df.drop(columns=[c for c in shares_cols if c != best_shares])
        df.rename(columns={best_shares: 'shares_outstanding'}, inplace=True)
    elif len(shares_cols) == 1:
        df.rename(columns={shares_cols[0]: 'shares_outstanding'}, inplace=True)
        
    # Ensure Market Cap is updated
    if 'price_at_period_end' in df.columns and 'shares_outstanding' in df.columns:
        df['market_cap'] = (df['price_at_period_end'] * df['shares_outstanding']).round(2)
        
    return df

def extract_longest_streak(df: pd.DataFrame) -> pd.DataFrame:
    """Finds the longest continuous streak of quarters (80-105 days gap)."""
    if 'market_cap' not in df.columns:
        return pd.DataFrame()
        
    df_valid = df.dropna(subset=['market_cap']).copy()
    if df_valid.empty:
        return pd.DataFrame()

    df_valid['period_end'] = pd.to_datetime(df_valid['period_end'])
    df_valid = df_valid.sort_values('period_end').reset_index(drop=True)
    
    longest_streak = []
    current_streak = [0]
    
    for i in range(1, len(df_valid)):
        days_diff = (df_valid.loc[i, 'period_end'] - df_valid.loc[i-1, 'period_end']).days
        if 80 <= days_diff <= 105:
            current_streak.append(i)
        else:
            if len(current_streak) > len(longest_streak):
                longest_streak = current_streak
            current_streak = [i]  # Reset streak
            
    if len(current_streak) > len(longest_streak):
        longest_streak = current_streak
        
    return df_valid.iloc[longest_streak].copy()

def main():
    logger.info("=" * 70)
    logger.info("MARKET CAP CLEANING & CONTINUITY FILTER (PAPER STEP 6)")
    logger.info("=" * 70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Configuration file {CONFIG_FILE} not found.")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {
        'checked': 0, 
        'kept': 0, 
        'deleted_no_cap': 0, 
        'deleted_inflation_bias': 0,
        'deleted_continuity': 0
    }

    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            ticker = company['ticker']
            
            if company.get('status') != 'active':
                continue

            stats['checked'] += 1
            csv_path = RAW_DATA_DIR / sector_name / f"{ticker}_raw.csv"

            if not csv_path.exists():
                company['status'] = 'delete'
                company['reason'] = 'Missing CSV file'
                continue

            try:
                df = pd.read_csv(csv_path)
                df_clean = clean_dataframe(df)
                
                # === NEW RULE: INFLATION BIAS FILTER ===
                # If 'shares_outstanding' has only 1 unique value across all quarters, 
                # it means the crawler used the current shares fallback. We must delete it.
                if 'shares_outstanding' not in df_clean.columns or df_clean['shares_outstanding'].nunique() <= 1:
                    logger.warning(f"  ✗ {ticker}: Constant shares detected. Deleting due to Inflation Bias risk.")
                    csv_path.unlink()
                    company['status'] = 'delete'
                    company['reason'] = 'Inflation Bias risk (Used constant/fallback shares)'
                    stats['deleted_inflation_bias'] += 1
                    continue
                # ========================================

                df_streak = extract_longest_streak(df_clean)
                streak_len = len(df_streak)

                if streak_len == 0:
                    logger.warning(f"  ✗ {ticker}: No valid Market Cap data. Deleting.")
                    csv_path.unlink()
                    company['status'] = 'delete'
                    company['reason'] = 'No valid Market Cap calculated'
                    stats['deleted_no_cap'] += 1
                    
                elif streak_len < MIN_CONSECUTIVE_QUARTERS:
                    logger.warning(f"  ✗ {ticker}: Max continuity is {streak_len} quarters (Needs >= {MIN_CONSECUTIVE_QUARTERS}). Deleting.")
                    csv_path.unlink()
                    company['status'] = 'delete'
                    company['reason'] = f'Failed continuity filter ({streak_len} < {MIN_CONSECUTIVE_QUARTERS} qtrs)'
                    stats['deleted_continuity'] += 1
                    
                else:
                    logger.info(f"  ✓ {ticker}: Kept. Longest streak = {streak_len} quarters.")
                    # TRUNCATE: Save only the continuous streak back to CSV
                    df_streak.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    company['status'] = 'active'
                    stats['kept'] += 1

            except Exception as e:
                logger.error(f"  ✗ {ticker}: Error processing file ({e})")
                if csv_path.exists():
                    csv_path.unlink()
                company['status'] = 'delete'
                company['reason'] = f'Error during continuity filter: {e}'

    # Update YAML Config
    logger.info("\nWriting updated statuses back to YAML config...")
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        f.write("# FOCAL OBJECTS CONFIGURATION (Updated after Continuity Filter)\n\n")
        yaml.dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    logger.info("\n" + "=" * 70)
    logger.info("CONTINUITY FILTER COMPLETE: SUMMARY")
    logger.info(f"  - Total active checked      : {stats['checked']}")
    logger.info(f"  - Companies KEPT            : {stats['kept']}")
    logger.info(f"  - Deleted (No Cap Data)     : {stats['deleted_no_cap']}")
    logger.info(f"  - Deleted (Inflation Bias)  : {stats['deleted_inflation_bias']}")
    logger.info(f"  - Deleted (Continuity < 6)  : {stats['deleted_continuity']}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
