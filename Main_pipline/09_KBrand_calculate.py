"""
STEP 7: TIME-ADJUSTED KBRAND ESTIMATION (PROXY METHOD)
Calculates KBrand based on Brand scores and sector multipliers.
Methodology: KBrand = Revenue * Base_Multiplier * (Brand_Score / 100)
Compliance: Paper Table 1 (Page 17) & Tier 3 Data Quality (Page 38).
"""

import pandas as pd
import yaml
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

CONFIG_FILE = Path("../Survey_config.yaml")
RAW_DATA_DIR = Path("../data/raw")

# ==================== BRAND SCORES OVER TIME ====================
# Source: Interbrand Best Global Brands / Brand Finance proxies
# Score 0-100. Companies not in this list get a default low score (15).
BRAND_SCORES = {
    'AAPL': {2015: 92, 2016: 94, 2017: 96, 2018: 98, 2019: 99, 2020: 100, 2021: 100, 2022: 100, 2023: 100},
    'MSFT': {2015: 78, 2016: 82, 2017: 85, 2018: 87, 2019: 89, 2020: 91, 2021: 93, 2022: 95, 2023: 97},
    # Add major pharma/finance/tech companies here if needed
    'JNJ': {2020: 75, 2021: 76, 2022: 77, 2023: 78},
    'JPM': {2020: 80, 2021: 82, 2022: 83, 2023: 85},
}

# Base multipliers: Brand intensity as % of Revenue
# UPDATED to match the 3 new high-volume dynamic sectors
BASE_MULTIPLIERS = {
    'Technology': 0.35,                  # High brand reliance in tech
    'Healthcare_Pharma': 0.20,           # Patents matter more than consumer brand
    'Financials_and_Real_Estate': 0.15   # Trust matters, but structurally different from consumer tech
}

def get_brand_score(ticker: str, year: int) -> float:
    """Gets brand score with linear interpolation/extrapolation."""
    scores = BRAND_SCORES.get(ticker, {})
    if not scores:
        return 15.0  # Default score for non-top-100 companies
    
    years = sorted(scores.keys())
    if year in scores:
        return float(scores[year])
    if year < years[0]:
        return float(scores[years[0]])
    if year > years[-1]:
        if len(years) >= 2:
            last_growth = (scores[years[-1]] - scores[years[-2]]) / (years[-1] - years[-2])
        else:
            last_growth = 0
        return max(0.0, min(100.0, scores[years[-1]] + last_growth * (year - years[-1])))
    
    before = max(y for y in years if y < year)
    after = min(y for y in years if y > year)
    interpolated = scores[before] + (scores[after] - scores[before]) * (year - before) / (after - before)
    return round(interpolated, 1)

def calculate_kbrand(row: pd.Series, ticker: str, sector: str) -> float:
    """Formula: KBrand_t = Revenue_t * Base_Multiplier * (Brand_Score_t / 100)"""
    if 'Revenue' not in row.index or pd.isna(row['Revenue']) or pd.isna(row['period_end']):
        return None
        
    revenue = row['Revenue']
    try:
        year = pd.to_datetime(row['period_end']).year
    except:
        return None
    
    base_mult = BASE_MULTIPLIERS.get(sector, 0.20) # Default fallback multiplier
    brand_factor = get_brand_score(ticker, year) / 100.0
    
    kbrand = revenue * base_mult * brand_factor
    return round(kbrand, 2)

def main():
    logger.info("="*70)
    logger.info("STARTING K_BRAND ESTIMATION (PROXY METHOD)")
    logger.info("="*70)

    if not CONFIG_FILE.exists():
        logger.error(f"❌ Configuration file {CONFIG_FILE} not found.")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {'processed': 0, 'total_rows': 0, 'kbrand_computed': 0}

    for sector_name, sector_info in config.get('sectors', {}).items():
        logger.info(f"\nProcessing Sector: {sector_name}")
        
        for company in sector_info.get('companies', []):
            ticker = company['ticker']
            
            # CRITICAL: Skip deleted companies
            if company.get('status') != 'active':
                continue

            csv_path = RAW_DATA_DIR / sector_name / f"{ticker}_raw.csv"
            
            if not csv_path.exists():
                logger.warning(f"  ✗ {ticker}: CSV missing.")
                continue

            try:
                df = pd.read_csv(csv_path)
                
                # Checkpoint: Skip if KBrand is already calculated completely
                if 'KBrand' in df.columns and df['KBrand'].notna().all():
                    logger.debug(f"  ⏭️ {ticker}: KBrand already exists. Skipping.")
                    stats['processed'] += 1
                    continue
                
                # Calculate KBrand across the dataframe
                df['KBrand'] = df.apply(lambda row: calculate_kbrand(row, ticker, sector_name), axis=1)
                
                # Save
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                computed_count = df['KBrand'].notna().sum()
                stats['total_rows'] += len(df)
                stats['kbrand_computed'] += computed_count
                stats['processed'] += 1
                
                logger.debug(f"  ✓ {ticker}: KBrand computed for {computed_count}/{len(df)} quarters.")

            except Exception as e:
                logger.error(f"  ✗ {ticker}: Error computing KBrand - {e}")

    logger.info("\n" + "="*70)
    logger.info("K_BRAND ESTIMATION COMPLETE")
    logger.info(f"  - Active companies processed : {stats['processed']}")
    logger.info(f"  - Total KBrand rows computed : {stats['kbrand_computed']}/{stats['total_rows']}")
    logger.info("="*70)

if __name__ == "__main__":
    main()
