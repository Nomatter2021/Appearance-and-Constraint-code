import yaml
import random
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def categorize_and_sample(input_file, output_file, sample_size=2000):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file {input_file}.")
        return

    # Khởi tạo 4 bucket chính xác với tên mà file crawl đã gán
    buckets = {
        "Healthcare": [],
        "Financial": [],
        "Technology": [],
        "Services": []
    }

    logging.info("Đang đưa các công ty vào 4 nhóm ngành chính...")

    for sic_key, content in data.get('sectors', {}).items():
        # Lấy description ra (Lúc này desc chính là giá trị "Healthcare", "Technology",...)
        desc = content.get('description', '').strip()
        companies = content.get('companies', [])

        # Nếu mô tả khớp với 1 trong 4 bucket thì tống hết công ty vào đó
        if desc in buckets:
            buckets[desc].extend(companies)

    survey_config = {"sectors": {}}
    
    # Metadata mô tả cho YAML cuối
    sector_meta = {
        "Healthcare": "Healthcare & Life Sciences (Pharma, Biotech, Medical Devices)",
        "Financial": "Financial Services (Banks, Insurance, Real Estate)",
        "Technology": "Technology (Hardware, Software, Semiconductors)",
        "Services": "General Services (excluding Software & Healthcare)"
    }

    logging.info("Bắt đầu lấy mẫu ngẫu nhiên...")

    for sector, all_companies in buckets.items():
        available_count = len(all_companies)
        actual_sample_size = min(sample_size, available_count)
        
        if actual_sample_size > 0:
            random_sample = random.sample(all_companies, actual_sample_size)
        else:
            random_sample = []
        
        # Thêm 'status: active' 
        for co in random_sample:
            co['status'] = 'active'

        survey_config["sectors"][sector] = {
            "description": sector_meta[sector],
            "companies": random_sample
        }
        
        logging.info(f"Nhóm {sector}: Đã lấy mẫu {actual_sample_size}/{available_count} công ty có sẵn.")

    logging.info(f"Đang ghi kết quả ra {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(survey_config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logging.info("Hoàn tất! File Survey_config.yaml đã được tạo thành công.")

if __name__ == "__main__":
    INPUT_FILE = '../SEC_Ticker_config.yaml'
    OUTPUT_FILE = '../Survey_config.yaml'
    SAMPLE_TARGET = 2000
    
    categorize_and_sample(INPUT_FILE, OUTPUT_FILE, SAMPLE_TARGET)