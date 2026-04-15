"""
13_adding_variables.py

BỔ SUNG TOÀN BỘ BIẾN PAPER 2 VÀO CÁC FILE CLASSIFIED
------------------------------------------------------
- Đọc RSSI historical từ data/processed/{sector}_RSSI_historical.csv
- Merge vào từng file *_classified.csv
- Tính các biến firm-level: D_t, A, B, Phi_t, Phi_drop, Psi_t
- Tính MCF_t và MRF_t_placeholder (dùng RSSI_hist_winsor)
- Tính dRSSI_dt, d²RSSI/dt², dRSSI_negative
- Ghi đè file (có backup tùy chọn)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CẤU HÌNH
# ============================================================================
CLASSIFIED_DIR = Path('../data/classified')
PROCESSED_DIR = Path('../data/processed')
BACKUP = True  # Đặt True để tạo file .bak trước khi ghi đè

SECTORS = ['Healthcare', 'Technology', 'Services']

# ============================================================================
# HÀM TÍNH CÁC BIẾN FIRM-LEVEL (từ dữ liệu đã có)
# ============================================================================
def add_firm_level_variables(df):
    df = df.copy()
    df = df.sort_values('period_end').reset_index(drop=True)

    # D_t: log distance
    df['D_t'] = np.where(
        df['V_Prod_base'] > 0,
        np.log(df['market_cap'] / df['V_Prod_base']),
        np.nan
    )

    # A và B
    df['A'] = 1 + df['PGR_t']
    df['B'] = df['E_3'] - df['A']

    # Phi_t
    df['Phi_t'] = np.where(df['dK_Pi_prime'] > 0, 1, 0)

    # Phi_drop: transition từ 1 → 0
    df['Phi_prev'] = df['Phi_t'].shift(1)
    df['Phi_drop'] = ((df['Phi_prev'] == 1) & (df['Phi_t'] == 0)).astype(int)
    df.drop(columns=['Phi_prev'], inplace=True)

    # Psi_t
    psi_raw = df['dK_Pi_prime_pct'] * (1 - df['PDI_t'])
    df['Psi_t'] = np.where(df['Phi_t'] == 1, psi_raw.clip(lower=0), 0.0)

    return df

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("BỔ SUNG BIẾN PAPER 2 VÀO CÁC FILE CLASSIFIED")
    print("=" * 70)

    for sector in SECTORS:
        rssi_path = PROCESSED_DIR / f"{sector}_RSSI_historical.csv"
        if not rssi_path.exists():
            print(f"⚠️ Không tìm thấy {rssi_path}, bỏ qua {sector}.")
            continue

        # Đọc RSSI
        rssi_df = pd.read_csv(rssi_path, parse_dates=['period_end'])
        # Chọn cột cần merge (ưu tiên bản winsorized)
        cols_to_merge = ['period_end', 'RSSI_hist_winsor', 'dRSSI_hist_winsor_dt']
        # Đổi tên cho gọn
        rssi_df = rssi_df[cols_to_merge].copy()
        rssi_df.rename(columns={
            'RSSI_hist_winsor': 'RSSI',
            'dRSSI_hist_winsor_dt': 'dRSSI_dt'
        }, inplace=True)

        # Đọc từng file classified
        sector_dir = CLASSIFIED_DIR / sector
        csv_files = list(sector_dir.glob('*_classified.csv'))
        print(f"\n📁 {sector}: {len(csv_files)} files")

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, parse_dates=['period_end'])

                # Xóa cột RSSI cũ nếu có
                old_cols = ['RSSI', 'dRSSI_dt', 'd2RSSI_dt2', 'dRSSI_negative',
                            'MCF_t', 'MRF_t_placeholder']
                for col in old_cols:
                    if col in df.columns:
                        df.drop(columns=[col], inplace=True)

                # Merge RSSI mới
                df = df.merge(rssi_df, on='period_end', how='left')

                # Tính d²RSSI/dt²
                df = df.sort_values('period_end')
                df['d2RSSI_dt2'] = df['dRSSI_dt'].diff()

                # dRSSI_negative
                df['dRSSI_negative'] = (df['dRSSI_dt'] < 0).astype(int)

                # Tính các biến firm-level (nếu có đủ cột)
                required = ['V_Prod_base', 'market_cap', 'PGR_t', 'E_3',
                            'dK_Pi_prime', 'dK_Pi_prime_pct', 'PDI_t']
                if all(c in df.columns for c in required):
                    df = add_firm_level_variables(df)
                else:
                    print(f"   ⚠️ {fpath.name} thiếu cột cần thiết, bỏ qua tính firm-level.")
                    continue

                # Tính MCF_t
                if 'D_t' in df.columns and 'Phi_t' in df.columns:
                    df['MCF_t'] = df['D_t'] * df['RSSI'] * df['Phi_t']

                # MRF_t placeholder
                if 'MCF_t' in df.columns and 'market_cap' in df.columns and 'Psi_t' in df.columns:
                    df['V_Price_lag1'] = df['market_cap'].shift(1)
                    df['MRF_t_placeholder'] = 1.0 * df['MCF_t'] * df['V_Price_lag1'] * df['Psi_t']
                    df.drop(columns=['V_Price_lag1'], inplace=True)

                # Backup nếu cần
                if BACKUP:
                    backup_path = fpath.with_suffix('.bak')
                    shutil.copy(fpath, backup_path)

                # Ghi đè
                df.to_csv(fpath, index=False)
                print(f"   ✅ {fpath.name}")

            except Exception as e:
                print(f"   ❌ Lỗi {fpath.name}: {e}")

    print("\n" + "=" * 70)
    print("HOÀN THÀNH BỔ SUNG BIẾN PAPER 2")
    print("=" * 70)

if __name__ == "__main__":
    main()