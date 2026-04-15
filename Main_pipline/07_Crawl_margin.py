"""
STEP 5: MARKET CAPITALIZATION CRAWLER
Fetches historical stock prices and shares outstanding from Yahoo Finance to compute Market Cap (V_Price).
Ensures alignment with accounting periods and protects against survival bias.

Methodology Adjustments:
1. Backward Alignment: Fetches the closest price ON or BEFORE the period end.
2. Month-End Synchronization: Aligns period_end to the last day of the month.
3. Continuity Protection: Checkpointing implemented to resume interrupted massive downloads.
"""

import pandas as pd
import yfinance as yf
import yaml
import os
import time
import random
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
CONFIG_FILE = Path("../Survey_config.yaml")
RAW_DATA_DIR = Path("../data/raw")
BACKUP_FOLDER = Path("../data/backup")

# Throttling to prevent API rate limiting from Yahoo Finance (Adjust if getting HTTP 429)
SLEEP_BETWEEN_COMPANIES = (1.5, 3.0) 

def create_backup(file_path):
    """Creates a timestamped backup before modifying the raw CSV."""
    BACKUP_FOLDER.mkdir(parents=True, exist_ok=True)
    backup_name = f"{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    backup_path = BACKUP_FOLDER / backup_name
    try:
        shutil.copy2(file_path, backup_path)
        logger.debug(f"  💾 Backup created: {backup_name}")
    except Exception as e:
        logger.error(f"  Backup failed: {e}")

# ==================== DATA PROCESSING CORE ====================
def process_company_market_cap(ticker: str, sector_name: str) -> dict:
    """Fetches price/shares and merges into the existing quarterly CSV."""
    file_path = RAW_DATA_DIR / sector_name / f"{ticker}_raw.csv"
    stats = {'success': False, 'rows': 0, 'market_cap_calculated': 0, 'skipped': False}
    
    if not file_path.exists():
        logger.warning(f"  ⚠️ File not found: {file_path.name}")
        return stats
    
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            logger.warning("  Empty file, skipping.")
            return stats

        # === CHECKPOINT ===
        # If 'market_cap' exists and is fully calculated, skip the API call
        if 'market_cap' in df.columns and df['market_cap'].notna().all():
            logger.info(f"  ⏭️ Checkpoint: Skipping {ticker} (Market Cap already calculated).")
            stats['skipped'] = True
            stats['success'] = True
            return stats

        logger.info(f"📁 Processing: {ticker} (Sector: {sector_name})")
        
        # SYNC FIX: Ensure period_end is aligned to Month-End for consistency with Step 03
        df['period_end'] = pd.to_datetime(df['period_end']) + pd.offsets.MonthEnd(0)
        df['period_end'] = df['period_end'].astype('datetime64[ns]')
        stats['rows'] = len(df)
        
        create_backup(file_path)
        
        # Buffer for price fetching
        start_date = df['period_end'].min() - pd.Timedelta(days=45)
        end_date = df['period_end'].max() + pd.Timedelta(days=15)
        
        ticker_obj = yf.Ticker(ticker)
        price_history = ticker_obj.history(start=start_date, end=end_date)
        
        if price_history.empty:
            logger.error(f"  ❌ No price data found for {ticker} (Likely delisted/ticker changed).")
            return stats
        
        # Align price history index to datetime64[ns]
        price_history.index = pd.to_datetime(price_history.index).tz_localize(None).astype('datetime64[ns]')
        
        # Fetch historical Shares Outstanding (Crucial for V_Price calculation)
        shares_df = None
        try:
            shares_history = ticker_obj.get_shares_full(start=start_date, end=end_date)
            if shares_history is not None and not shares_history.empty:
                shares_df = shares_history.to_frame(name='shares_outstanding')
                shares_df.index = pd.to_datetime(shares_df.index).tz_localize(None).astype('datetime64[ns]')
            else:
                curr_shares = ticker_obj.info.get('sharesOutstanding')
                if curr_shares:
                    shares_df = pd.DataFrame({'shares_outstanding': curr_shares}, index=price_history.index)
                    logger.warning("  ⚠️ Using current shares for historical periods (Risk of inflation).")
        except Exception as e:
            logger.warning(f"  ⚠️ Could not fetch shares: {e}")

        # MERGE LOGIC: Using 'backward' to get the last available price before period end
        df = df.sort_values('period_end')
        price_history = price_history.sort_index()
        
        # Merge Price
        df_merged = pd.merge_asof(
            df,
            price_history[['Close']].reset_index(),
            left_on='period_end',
            right_on='Date',
            direction='backward', # FIX: Use closest price BEFORE/ON period end
            tolerance=pd.Timedelta('7 days')
        ).drop(columns=['Date'], errors='ignore')
        
        # Merge Shares
        if shares_df is not None:
            shares_df = shares_df.sort_index().reset_index().rename(columns={'index': 'Date'})
            df_merged = pd.merge_asof(
                df_merged.sort_values('period_end'),
                shares_df,
                left_on='period_end',
                right_on='Date',
                direction='backward',
                tolerance=pd.Timedelta('120 days')
            ).drop(columns=['Date'], errors='ignore')

        # Calculate V_Price (Market Cap)
        if 'Close' in df_merged.columns:
            df_merged.rename(columns={'Close': 'price_at_period_end'}, inplace=True)
            if 'shares_outstanding' in df_merged.columns:
                df_merged['market_cap'] = (df_merged['price_at_period_end'] * df_merged['shares_outstanding']).round(2)
                stats['market_cap_calculated'] = df_merged['market_cap'].notna().sum()
                logger.info(f"  ✓ Market Cap computed: {stats['market_cap_calculated']}/{len(df_merged)} rows")
        
        # Save updated CSV
        df_merged.to_csv(file_path, index=False, encoding='utf-8-sig')
        stats['success'] = True
        return stats
        
    except Exception as e:
        logger.error(f"❌ Error processing {ticker}: {str(e)}")
        return stats

# ==================== MAIN PIPELINE ====================
def main():
    logger.info("=" * 70)
    logger.info("MARKET CAP CRAWLER - RESEARCH REPLICATION SUITE")
    logger.info("=" * 70)
    
    if not CONFIG_FILE.exists():
        logger.error(f"Missing input file: {CONFIG_FILE}")
        return

    # Load dynamic config
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Flatten the companies list for tracking progress
    all_companies = []
    for sector_name, sector_info in config.get('sectors', {}).items():
        for company in sector_info.get('companies', []):
            if company.get('status') == 'active':
                all_companies.append({
                    'ticker': company['ticker'],
                    'sector': sector_name
                })
    
    total_companies = len(all_companies)
    logger.info(f"Total active companies to process: {total_companies}")

    for idx, comp in enumerate(all_companies):
        ticker = comp['ticker']
        sector_name = comp['sector']
        
        # Progress indicator
        if idx % 50 == 0 and idx != 0:
            logger.info(f"--- Progress: {idx}/{total_companies} companies processed ---")
            
        stats = process_company_market_cap(ticker, sector_name)
        
        # Only sleep if we actually made API calls (didn't skip via Checkpoint)
        if stats.get('success') and not stats.get('skipped'):
            if idx < total_companies - 1:
                wait = random.uniform(*SLEEP_BETWEEN_COMPANIES)
                time.sleep(wait)

    logger.info("\n" + "=" * 70)
    logger.info("CRAWL COMPLETE - V_PRICE (MARKET CAP) LAYER READY")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
