"""
00_plot_RSSI_timeseries.py

VẼ BIỂU ĐỒ CHUỖI THỜI GIAN RSSI THEO NGÀNH
-------------------------------------------
- Đọc dữ liệu từ data/processed/{sector}_RSSI_historical.csv
- Vẽ biểu đồ riêng cho từng ngành (Healthcare, Technology, Services)
- Vẽ biểu đồ tổng hợp so sánh ba ngành
- Lưu ảnh vào results/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CẤU HÌNH
# ============================================================================
PROCESSED_DIR = Path('../data/processed')
OUTPUT_DIR = Path('results/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ['Healthcare', 'Technology', 'Services']
COLORS = {'Healthcare': 'tab:blue', 'Technology': 'tab:green', 'Services': 'tab:orange'}

# Chọn cột RSSI để vẽ (có thể đổi thành 'RSSI_hist' nếu muốn)
RSSI_COL = 'RSSI_hist_winsor'
ROLL_COL = None  # Nếu muốn vẽ thêm đường rolling thì đặt tên cột, ví dụ: 'RSSI_hist'

FIG_SIZE = (12, 6)

# ============================================================================
# HÀM VẼ CHO MỘT NGÀNH
# ============================================================================
def plot_single_sector(sector):
    file_path = PROCESSED_DIR / f"{sector}_RSSI_historical.csv"
    if not file_path.exists():
        print(f"⚠️ Không tìm thấy file: {file_path}")
        return

    df = pd.read_csv(file_path, parse_dates=['period_end'])
    df = df.sort_values('period_end')

    plt.figure(figsize=FIG_SIZE)
    plt.plot(df['period_end'], df[RSSI_COL],
             marker='o', linestyle='-', linewidth=1.5, markersize=3,
             color=COLORS[sector], label=f'{sector} ({RSSI_COL})')

    # Vẽ thêm đường rolling nếu có
    if ROLL_COL and ROLL_COL in df.columns:
        plt.plot(df['period_end'], df[ROLL_COL],
                 linestyle='--', linewidth=2, color='darkred', alpha=0.7,
                 label=f'{sector} ({ROLL_COL})')

    plt.title(f'RSSI Historical – {sector}', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('RSSI', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / f'RSSI_{sector}.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu: {out_path}")

# ============================================================================
# HÀM VẼ TỔNG HỢP BA NGÀNH
# ============================================================================
def plot_all_sectors():
    plt.figure(figsize=FIG_SIZE)

    for sector in SECTORS:
        file_path = PROCESSED_DIR / f"{sector}_RSSI_historical.csv"
        if not file_path.exists():
            continue
        df = pd.read_csv(file_path, parse_dates=['period_end'])
        df = df.sort_values('period_end')
        plt.plot(df['period_end'], df[RSSI_COL],
                 marker='o', linestyle='-', linewidth=1.5, markersize=3,
                 color=COLORS[sector], label=sector, alpha=0.8)

    plt.title(f'RSSI Historical – So sánh các ngành', fontsize=14)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('RSSI', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = OUTPUT_DIR / 'RSSI_all_sectors.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✅ Đã lưu biểu đồ tổng hợp: {out_path}")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("VẼ BIỂU ĐỒ CHUỖI THỜI GIAN RSSI")
    print("=" * 70)

    for sector in SECTORS:
        plot_single_sector(sector)

    plot_all_sectors()

    print("\n✅ Hoàn tất. Các biểu đồ được lưu trong:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
