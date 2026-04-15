"""
12_calculate_RSSI.py

TÍNH TOÁN RSSI THEO HISTORICAL ROLLING BASELINE (FINAL)
-------------------------------------------------------
- Đọc toàn bộ file *_classified.csv từ Healthcare, Technology, Services.
- Tính trung bình K_Pi_prime theo quý cho từng sector.
- RSSI_t = (mean_KPi_t - rolling_mean_historical) / rolling_std_historical
- Rolling window: 8 quý, min_periods=4, shift(1) để tránh look-ahead bias.
- Lưu: data/processed/{sector}_RSSI_historical.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CẤU HÌNH
# ============================================================================
CLASSIFIED_DIR = Path('../data/classified')
OUTPUT_DIR = Path('../data/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ['Healthcare', 'Technology', 'Services']
ROLLING_WINDOW = 8   # 8 quý (2 năm)
MIN_PERIODS = 4

# ============================================================================
# 1. ĐỌC DỮ LIỆU VÀ TÍNH MEAN K_PI_PRIME THEO QUÝ
# ============================================================================
def load_sector_panel(sector_name):
    sector_path = CLASSIFIED_DIR / sector_name
    if not sector_path.exists():
        return pd.DataFrame()
    
    dfs = []
    for fpath in sector_path.glob('*_classified.csv'):
        try:
            df = pd.read_csv(fpath, parse_dates=['period_end'])
            if 'K_Pi_prime' not in df.columns:
                continue
            df = df[['period_end', 'K_Pi_prime']].copy()
            ticker = fpath.stem.replace('_classified', '')
            df['Ticker'] = ticker
            dfs.append(df)
        except Exception:
            pass
    
    if not dfs:
        return pd.DataFrame()
    
    panel = pd.concat(dfs, ignore_index=True)
    panel['period_end'] = pd.to_datetime(panel['period_end'])
    panel['period_end'] = panel['period_end'].dt.to_period('Q').dt.to_timestamp('Q')
    return panel

def compute_quarterly_mean(panel):
    quarterly = panel.groupby('period_end')['K_Pi_prime'].mean().reset_index()
    quarterly.columns = ['period_end', 'mean_KPi']
    quarterly = quarterly.sort_values('period_end').reset_index(drop=True)
    return quarterly

# ============================================================================
# 2. TÍNH RSSI HISTORICAL
# ============================================================================
def compute_rssi_historical(quarterly):
    df = quarterly.copy()
    
    df['hist_mean'] = df['mean_KPi'].rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).mean().shift(1)
    df['hist_std']  = df['mean_KPi'].rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS).std().shift(1)
    
    df['RSSI_hist'] = np.where(
        df['hist_std'] > 0,
        (df['mean_KPi'] - df['hist_mean']) / df['hist_std'],
        np.nan
    )
    
    df['dRSSI_hist_dt'] = df['RSSI_hist'].diff()
    
    # Winsorized (99th percentile) để dùng cho thống kê nếu cần
    cap = df['RSSI_hist'].quantile(0.99)
    df['RSSI_hist_winsor'] = df['RSSI_hist'].clip(upper=cap)
    df['dRSSI_hist_winsor_dt'] = df['RSSI_hist_winsor'].diff()
    
    return df

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("TÍNH TOÁN RSSI THEO HISTORICAL BASELINE (FINAL)")
    print("=" * 70)
    
    for sector in SECTORS:
        print(f"\n📁 Đang xử lý: {sector}")
        panel = load_sector_panel(sector)
        if panel.empty:
            print(f"  ⚠️ Không có dữ liệu cho {sector}, bỏ qua.")
            continue
        
        print(f"  - Số quan sát: {len(panel)}")
        print(f"  - Số ticker: {panel['Ticker'].nunique()}")
        
        quarterly = compute_quarterly_mean(panel)
        print(f"  - Số quý: {len(quarterly)}")
        
        if len(quarterly) < MIN_PERIODS + 1:
            print(f"  ❌ Không đủ số quý để tính RSSI (cần ít nhất {MIN_PERIODS+1}).")
            continue
        
        rssi_df = compute_rssi_historical(quarterly)
        valid = rssi_df['RSSI_hist'].notna().sum()
        print(f"  - Số quý có RSSI hợp lệ: {valid} / {len(rssi_df)}")
        
        if valid > 0:
            print(f"    RSSI_hist range: [{rssi_df['RSSI_hist'].min():.2f}, {rssi_df['RSSI_hist'].max():.2f}]")
            print(f"    RSSI_hist median: {rssi_df['RSSI_hist'].median():.2f}")
        
        out_path = OUTPUT_DIR / f"{sector}_RSSI_historical.csv"
        rssi_df.to_csv(out_path, index=False)
        print(f"  ✅ Đã lưu: {out_path}")
    
    print("\n" + "=" * 70)
    print("HOÀN THÀNH")
    print("=" * 70)

if __name__ == "__main__":
    main()