import requests
from bs4 import BeautifulSoup
import yaml
import time
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def load_local_sec_mapping(yaml_file):
    mapping = {}
    try:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if not data or 'sectors' not in data:
                return mapping
            
            for sector_key, sector_info in data['sectors'].items():
                for company in sector_info.get('companies', []):
                    ticker = company.get('ticker')
                    if ticker:
                        mapping[ticker] = {
                            'cik': company.get('cik'),
                            'name': company.get('name')
                        }
    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy file {yaml_file}.")
    
    return mapping

def auto_generate_sme_benchmark():
    sec_yaml_file = '../SEC_Ticker_config.yaml'
    logging.info(f"1. Đang tải dữ liệu từ file: {sec_yaml_file}...")
    sec_mapping = load_local_sec_mapping(sec_yaml_file)
    
    if not sec_mapping:
        logging.error("Không tải được dữ liệu SEC. Dừng chạy script.")
        return

    logging.info(f"-> Đã tải thành công {len(sec_mapping)} tickers để đối chiếu.")

    # Cập nhật mapping 4 nhóm ngành tương ứng với Finviz filter
    FINVIZ_SECTOR_MAPPING = {
        'Healthcare': ['sec_healthcare'],
        'Financial': ['sec_financial'],
        'Technology': ['sec_technology'],
        # Finviz không có nhóm "Services" chung, nên gộp 2 nhóm có dịch vụ nhiều nhất
        'Services': ['sec_communicationservices', 'sec_industrials'] 
    }
    
    benchmark_config = {"sectors": {}}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
    }

    logging.info("\n2. Bắt đầu quét Finviz & Đối chiếu SEC...")
    
    for bucket_name, finviz_codes in FINVIZ_SECTOR_MAPPING.items():
        logging.info(f"\n--- Xử lý nhóm: {bucket_name} ---")
        
        benchmark_config["sectors"][bucket_name] = {
            "description": f"SME Companies (<$2B) extracted from Finviz ({', '.join(finviz_codes)})",
            "companies": []
        }
        
        matched_count = 0
        
        for finviz_code in finviz_codes:
            if matched_count >= 200:
                break 
                
            page_index = 1
            logging.info(f"-> Đang quét bộ lọc Finviz: {finviz_code}...")
            
            while matched_count < 200:
                url = f"https://finviz.com/screener.ashx?v=111&f=cap_smallunder,{finviz_code}&r={page_index}"
                
                try:
                    resp = requests.get(url, headers=headers, timeout=10)
                    resp.raise_for_status()
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    
                    links = soup.find_all('a', href=re.compile(r'^quote\.ashx\?t='))
                    
                    page_tickers = []
                    for link in links:
                        t = link.text.strip()
                        if t.isupper() and t.isalpha() and 1 <= len(t) <= 5:
                            if t not in page_tickers:
                                page_tickers.append(t)
                    
                    if not page_tickers:
                        logging.warning(f"Hết dữ liệu ở trang {page_index} cho {finviz_code}.")
                        break 
                        
                    for t in page_tickers:
                        if t in sec_mapping and matched_count < 200:
                            company_data = {
                                'ticker': t,
                                'cik': sec_mapping[t]['cik'],
                                'name': sec_mapping[t]['name'] 
                            }
                            
                            if company_data not in benchmark_config["sectors"][bucket_name]["companies"]:
                                benchmark_config["sectors"][bucket_name]["companies"].append(company_data)
                                matched_count += 1
                                
                    logging.info(f"Trang {page_index} ({finviz_code}): Tìm thấy {len(page_tickers)} tickers -> Lọc khớp SEC: {matched_count}/200")
                    
                    page_index += 20 
                    time.sleep(2.0)  
                    
                except Exception as e:
                    logging.error(f"Lỗi khi quét {finviz_code} tại trang {page_index}: {e}")
                    break

    output_file = '../Benchmark_config.yaml'
    logging.info(f"\n3. Đang xuất ra file {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as file:
        yaml.dump(benchmark_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
    logging.info("="*50)
    logging.info(f"✓ HOÀN TẤT: File {output_file} đã được tạo.")
    logging.info("Thống kê:")
    for sector, data in benchmark_config["sectors"].items():
        logging.info(f"- {sector}: {len(data['companies'])} companies")
    logging.info("="*50)

if __name__ == "__main__":
    auto_generate_sme_benchmark()