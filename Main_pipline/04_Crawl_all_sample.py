"""
SEC XBRL DATA CRAWLER FOR SPECULATIVE REGIME RESEARCH
Reads 'Survey_config.yaml' and fetches quarterly financial data.
Applies strict discrete quarter filtering to prevent Q4 restatement noise.
"""

import requests
import pandas as pd
import yaml
import time
import logging
import numpy as np
from pathlib import Path

# Configure logging for reproducible pipeline tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("../Survey_config.yaml")
OUTPUT_DIR = Path("../data/raw")

# IMPORTANT: Update this to your institutional or personal email
SEC_USER_AGENT = "Independent Researcher dramainmylife@gmail.com"
SEC_RATE_LIMIT = 0.15 # SEC allows 10 requests per second, 0.15s delay is safe

XBRL_FALLBACKS = {
    "Revenue": [
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "TotalRevenue",
        "NetSales"
    ],
    "OperatingIncome": [
        "OperatingIncomeLoss",
        "ProfitLoss",
        "OperatingIncome",
        "IncomeLossFromContinuingOperations",
        "NetIncomeLoss"
    ],
    "CostOfRevenue": [
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfSales",
        "CostOfRevenueAndCostOfGoodsSold"
    ]
}

def fetch_sec_metric(cik: str, metric_name: str, fallback_chain: list) -> list:
    """Fetch financial metric from SEC Company Facts API with standard fallbacks."""
    # CRITICAL FIX: Ensure CIK is exactly 10 digits padded with leading zeros
    cik_padded = str(cik).strip().zfill(10)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 404:
            return []
        
        response.raise_for_status()
        data = response.json()
        
        facts = data.get('facts', {}).get('us-gaap', {})
        for tag in fallback_chain:
            if tag in facts:
                units = facts[tag].get('units', {}).get('USD', [])
                if units:
                    logger.debug(f"Found {metric_name} via '{tag}'")
                    return units
        return []
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            logger.warning(f"Rate limited on CIK {cik_padded}. Waiting 10s...")
            time.sleep(10)
            return fetch_sec_metric(cik, metric_name, fallback_chain)
        logger.error(f"HTTP error for {cik_padded}: {e}")
    except Exception as e:
        logger.error(f"Error fetching {cik_padded}: {e}")
        
    return []

def parse_quarterly_records(records: list, metric_name: str) -> pd.DataFrame:
    """
    Parse SEC records strictly filtering for discrete quarters (80-105 days).
    Prevents FY - 9M subtractions that inject restatement noise into Q4 data.
    """
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    required = ['start', 'end', 'val', 'fy', 'fp']
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
        
    df = df[required].copy()
    df.columns = ['start_date', 'period_end', 'val', 'fy', 'fp']
    
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['period_end'] = pd.to_datetime(df['period_end'])
    df['days'] = (df['period_end'] - df['start_date']).dt.days
    
    # STRICT FILTER: Only keep discrete quarters to maintain the integrity of the R_t metric
    df_discrete = df[(df['days'] >= 80) & (df['days'] <= 105)].copy()
    
    # ALIGNMENT FIX: Standardize dates to the end of the month to fix SEC reporting mismatches
    df_discrete['period_end'] = df_discrete['period_end'] + pd.offsets.MonthEnd(0)
    
    # Deduplicate: Keep the latest reported value for a given period end
    df_discrete = df_discrete.sort_values(['period_end', 'val']).drop_duplicates('period_end', keep='last')
    df_discrete = df_discrete.rename(columns={'val': metric_name})
    
    return df_discrete.set_index('period_end')[[metric_name, 'fy', 'fp']]

def calculate_margin(df_rev: pd.DataFrame, df_op: pd.DataFrame, df_cogs: pd.DataFrame) -> pd.Series:
    """
    Calculate operating margin with fallback to gross margin.
    Safely handles zero revenue to prevent ZeroDivisionError inf/NaN cascades.
    """
    margin = pd.Series(index=df_rev.index, dtype=float)
    
    # SAFE DIVISION FIX
    safe_revenue = df_rev['Revenue'].replace(0, np.nan)
    
    if not df_op.empty and 'OperatingIncome' in df_op.columns:
        margin = df_op['OperatingIncome'] / safe_revenue
    elif not df_cogs.empty and 'CostOfRevenue' in df_cogs.columns:
        margin = (safe_revenue - df_cogs['CostOfRevenue']) / safe_revenue
        
    return margin.clip(-1, 1).astype(float)

def crawl_sec_data():
    """Main crawler function reading from YAML config with checkpointing."""
    if not CONFIG_FILE.exists():
        logger.error(f"❌ Config file {CONFIG_FILE} not found.")
        return
        
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger.info("="*60)
    logger.info("STARTING SEC DATA CRAWL WITH CHECKPOINT")
    logger.info("="*60)

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nProcessing Sector: {sector_name}")
        
        # Create sub-directories per sector
        sector_dir = OUTPUT_DIR / sector_name
        sector_dir.mkdir(parents=True, exist_ok=True)
        
        for company in sector_info.get('companies', []):
            ticker = company['ticker']
            cik = company['cik']
            
            # === CHECKPOINT ===
            # Define the file path first
            file_path = sector_dir / f"{ticker}_raw.csv"
            
            # If the file already exists, automatically skip to save time
            if file_path.exists():
                logger.info(f"⏭️ Checkpoint: Skipping {ticker} (Data already exists)")
                continue
            # ==================

            if company.get('status') != 'active':
                logger.info(f"⏭️ Skipping {ticker} (Status: {company.get('status')})")
                continue
                
            logger.info(f"  Fetching {ticker} (CIK: {cik})...")
            
            # Fetch data
            rev_records = fetch_sec_metric(cik, "Revenue", XBRL_FALLBACKS["Revenue"])
            op_records = fetch_sec_metric(cik, "OperatingIncome", XBRL_FALLBACKS["OperatingIncome"])
            cogs_records = fetch_sec_metric(cik, "CostOfRevenue", XBRL_FALLBACKS["CostOfRevenue"])
            
            time.sleep(SEC_RATE_LIMIT) # Respect SEC guidelines
            
            if not rev_records:
                logger.warning(f"  ⊘ {ticker}: No revenue records found.")
                continue
                
            # Parse data
            df_rev = parse_quarterly_records(rev_records, "Revenue")
            df_op = parse_quarterly_records(op_records, "OperatingIncome")
            df_cogs = parse_quarterly_records(cogs_records, "CostOfRevenue")
            
            if df_rev.empty:
                logger.warning(f"  ⊘ {ticker}: No discrete quarterly records found after parsing.")
                continue
                
            # Merge datasets relying on the MonthEnd alignment
            df_company = df_rev[['Revenue']].copy()
            if not df_op.empty:
                df_company = df_company.join(df_op[['OperatingIncome']], how='left')
            if not df_cogs.empty:
                df_company = df_company.join(df_cogs[['CostOfRevenue']], how='left')
                
            # Calculate Margin safely
            df_company['Operating_Margin'] = calculate_margin(df_rev, df_op, df_cogs)
            
            # Save to CSV inside sector sub-directory
            df_company.to_csv(file_path)
            
            valid_margins = df_company['Operating_Margin'].notna().sum()
            logger.info(f"  ✓ Saved {ticker}: {len(df_company)} quarters, {valid_margins} valid margins.")

    logger.info("\n" + "="*60)
    logger.info(f"CRAWL COMPLETE. Data saved to {OUTPUT_DIR.absolute()}")
    logger.info("="*60)

if __name__ == "__main__":
    crawl_sec_data()
