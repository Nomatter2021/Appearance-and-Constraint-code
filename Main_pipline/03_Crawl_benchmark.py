"""
STEP 2: CRAWL SEC DATA FOR BENCHMARK COHORT
Fetches historical financials for the benchmark groups dynamically loaded from Benchmark_config.yaml.
Output: data/benchmark/raw/{Sector}_benchmark_companies.xlsx
"""

import requests
import pandas as pd
import time
import logging
import yaml
from pathlib import Path

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('benchmark_crawl.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# SEC API Requirements
SEC_USER_AGENT = "Independent Researcher data_admin@yourdomain.com" # PLEASE UPDATE
SEC_RATE_LIMIT = 0.2 # SEC limit is 10 requests/sec; 0.2s is safe
RAW_DATA_DIR = Path("../data/benchmark/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = '../Benchmark_config.yaml'

# XBRL Tags with prioritized fallbacks for robust cross-sector matching
XBRL_FALLBACKS = {
    "Revenue": [
        "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "TotalRevenue", "NetSales"
    ],
    "OperatingIncome": [
        "OperatingIncomeLoss", "ProfitLoss", "OperatingIncome",
        "IncomeLossFromContinuingOperations", "NetIncomeLoss"
    ],
    "CostOfRevenue": [
        "CostOfRevenue", "CostOfGoodsSold", "CostOfSales", "CostOfRevenueAndCostOfGoodsSold"
    ]
}

# =============================================================================
# DATA PROCESSING CORE
# =============================================================================

def format_cik(cik):
    """Normalize CIK to 10-digit string for SEC API compatibility."""
    return str(cik).strip().zfill(10)

def load_benchmark_config(filepath: str) -> dict:
    """Load benchmark sectors and companies from the YAML config file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config.get('sectors', {})
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {filepath}. Please run Step 1 first.")
        return {}
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        return {}

def fetch_sec_metric(cik: str, metric_name: str, fallback_chain: list) -> list:
    """Fetch financial facts from SEC API with retry on rate limit."""
    cik_padded = format_cik(cik)
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    headers = {"User-Agent": SEC_USER_AGENT}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 429:
            logger.warning("Rate limit hit. Sleeping for 10s...")
            time.sleep(10)
            return fetch_sec_metric(cik, metric_name, fallback_chain)
        
        response.raise_for_status()
        data = response.json()
        facts = data.get('facts', {}).get('us-gaap', {})
        
        for tag in fallback_chain:
            if tag in facts:
                units = facts[tag].get('units', {}).get('USD', [])
                if units: return units
        return []
    except requests.exceptions.HTTPError as he:
        # 404 means the SEC doesn't have structured XBRL data for this CIK
        if response.status_code == 404:
             logger.warning(f"No structured XBRL data found for CIK {cik_padded} (HTTP 404).")
        else:
             logger.error(f"HTTP Error fetching CIK {cik_padded}: {he}")
        return []
    except Exception as e:
        logger.error(f"Error fetching CIK {cik_padded}: {e}")
        return []

def parse_to_dataframe(records: list, metric_name: str) -> pd.DataFrame:
    """Clean SEC records into quarterly dataframes, filtering for discrete quarters."""
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    
    # Filter for discrete 3-month periods (approx 80-105 days)
    if 'start' not in df.columns or 'end' not in df.columns:
        return pd.DataFrame()
        
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df['days'] = (df['end'] - df['start']).dt.days
    df = df[(df['days'] >= 80) & (df['days'] <= 105)].copy()
    
    if df.empty: return pd.DataFrame()
    
    # Align to MonthEnd to fix SEC date mismatches (e.g., 12-30 vs 12-31)
    df['end'] = df['end'] + pd.offsets.MonthEnd(0)
    df = df.rename(columns={'val': metric_name, 'end': 'period_end'})
    df = df.sort_values(['period_end', metric_name]).drop_duplicates('period_end', keep='last')
    
    return df.set_index('period_end')[[metric_name]]

def crawl_benchmark_data():
    """Execute the crawl dynamically based on the YAML config."""
    logger.info("="*60)
    logger.info("STARTING SEC BENCHMARK DATA CRAWL")
    logger.info("="*60)

    # 1. Load targets from YAML
    sectors_data = load_benchmark_config(CONFIG_FILE)
    if not sectors_data:
        logger.error("No sectors loaded. Exiting.")
        return

    # 2. Iterate through dynamic sectors
    for sector_name, sector_info in sectors_data.items():
        companies = sector_info.get('companies', [])
        logger.info(f"\n>>> Processing Sector: {sector_name} ({len(companies)} companies)")
        
        excel_file = RAW_DATA_DIR / f"{sector_name}_benchmark_companies.xlsx"
        sector_payload = {}

        for comp in companies:
            ticker = comp.get('ticker')
            cik = comp.get('cik')
            
            if not ticker or not cik:
                continue
                
            logger.info(f"  Fetching {ticker} (CIK: {cik})...")
            
            # Data Acquisition
            rev = parse_to_dataframe(fetch_sec_metric(cik, "Revenue", XBRL_FALLBACKS["Revenue"]), "Revenue")
            op = parse_to_dataframe(fetch_sec_metric(cik, "OperatingIncome", XBRL_FALLBACKS["OperatingIncome"]), "OperatingIncome")
            cogs = parse_to_dataframe(fetch_sec_metric(cik, "CostOfRevenue", XBRL_FALLBACKS["CostOfRevenue"]), "CostOfRevenue")
            
            if rev.empty: 
                logger.warning(f"  -> Skipping {ticker}: Insufficient Revenue data.")
                time.sleep(SEC_RATE_LIMIT)
                continue
            
            # Merging and Margin Calculation (SAFE DIVISION)
            df = rev.join([op, cogs], how='left')
            safe_rev = df['Revenue'].replace(0, pd.NA)
            
            if 'OperatingIncome' in df.columns:
                df['Operating_Margin'] = (df['OperatingIncome'] / safe_rev).clip(-1, 1)
            elif 'CostOfRevenue' in df.columns:
                df['Operating_Margin'] = ((safe_rev - df['CostOfRevenue']) / safe_rev).clip(-1, 1)
            else:
                df['Operating_Margin'] = pd.NA
                
            sector_payload[ticker] = df
            time.sleep(SEC_RATE_LIMIT)

        # Save sector-specific Excel with ticker-sheets
        if sector_payload:
            logger.info(f"Writing {sector_name} data to Excel...")
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                for t, data in sector_payload.items():
                    data.to_excel(writer, sheet_name=t)
            logger.info(f"✓ Saved {sector_name} data to {excel_file}")
        else:
            logger.warning(f"No valid data retrieved for sector: {sector_name}")

if __name__ == "__main__":
    crawl_benchmark_data()
