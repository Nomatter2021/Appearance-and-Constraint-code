"""
STEP 3: POST-PROCESSING BENCHMARK DATA
Operationalizes Section 4.2 of the paper:
1. Aligns fiscal quarters to calendar quarters.
2. Applies a 12-quarter (3-year) rolling mean of cross-sectional medians for S_baseline.
3. Implements Quality Tiers (N >= 5).

Output:
- data/processed/{sector}_benchmark_median.xlsx
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# --- GLOBAL CONFIGURATION ---
CONFIG_FILE = Path("../Benchmark_config.yaml")
RAW_DATA_DIR = Path("../data/benchmark/raw")
PROCESSED_DIR = Path("../data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Methodology Parameters from Paper Section 4.2
ROLLING_WINDOW = 12             # 3 Years = 12 Quarters 
MIN_COMPANIES_FOR_MEDIAN = 5    # Tier A threshold
# ----------------------------

def load_dynamic_sectors(config_path: Path) -> list:
    """Reads the configuration file to dynamically get the list of processed sectors."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return list(config.get('sectors', {}).keys())
    except Exception as e:
        logger.error(f"Failed to load sectors from {config_path}: {e}")
        return []

def load_and_align_data(sector: str) -> dict:
    excel_file = RAW_DATA_DIR / f"{sector}_benchmark_companies.xlsx"
    if not excel_file.exists():
        logger.error(f"Source file missing: {excel_file}")
        return {}

    company_data = {}
    xl = pd.ExcelFile(excel_file)
    for ticker in xl.sheet_names:
        df = pd.read_excel(excel_file, sheet_name=ticker, index_col=0)
        if df.empty: continue
        df.index = pd.to_datetime(df.index)
        # Align to calendar quarter end
        df.index = df.index.to_period('Q').to_timestamp('Q')
        df = df[~df.index.duplicated(keep='last')]
        company_data[ticker] = df.sort_index()
    return company_data

def calculate_rolling_S_baseline(company_data: dict, metric: str = 'Operating_Margin'):
    if not company_data: return pd.DataFrame()

    # Merge peers into wide format
    combined_metrics = pd.concat(
        [df[[metric]].rename(columns={metric: ticker}) for ticker, df in company_data.items()],
        axis=1
    ).sort_index()

    # Cross-sectional Median per quarter
    period_median = combined_metrics.median(axis=1, skipna=True)
    period_count = combined_metrics.notna().sum(axis=1)

    # 3-Year Smoothing for Socially Necessary Conditions (S_baseline)
    s_baseline_smooth = period_median.rolling(window=ROLLING_WINDOW, min_periods=4).mean()

    # Build result
    benchmark_df = pd.DataFrame({
        'raw_median': period_median,
        's_baseline': s_baseline_smooth,
        'n_peers': period_count
    })

    # Quality Tiers per Section 4.4
    benchmark_df['quality_tier'] = np.where(benchmark_df['n_peers'] >= MIN_COMPANIES_FOR_MEDIAN, 'A', 
                                   np.where(benchmark_df['n_peers'] >= 3, 'B', 'C'))
    
    benchmark_df['status'] = np.where(benchmark_df['n_peers'] >= 3, 'active', 'insufficient_data')

    return benchmark_df

def main():
    logger.info("="*60)
    logger.info("STARTING BENCHMARK POST-PROCESSING (DYNAMIC LOGIC)")
    logger.info("="*60)

    # DYNAMIC SECTOR LOADING: Replaces the hardcoded list
    sectors = load_dynamic_sectors(CONFIG_FILE)
    if not sectors:
        logger.error("No sectors found to process. Stopping.")
        return

    for sector in sectors:
        logger.info(f"Processing Sector: {sector}")
        data = load_and_align_data(sector)
        if not data: continue

        results = calculate_rolling_S_baseline(data, 'Operating_Margin')
        if not results.empty:
            output_path = PROCESSED_DIR / f"{sector}_benchmark_median.xlsx"
            results.to_excel(output_path)
            logger.info(f"  ✓ {sector}: Saved S_baseline to {output_path}")

    logger.info("="*60)
    logger.info("POST-PROCESSING COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
