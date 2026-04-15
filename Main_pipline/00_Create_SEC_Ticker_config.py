import pandas as pd
import yaml

def convert_excel_to_yaml(excel_file, yaml_file):
    """
    Đọc dữ liệu công ty từ file Excel (được tạo bởi script crawl),
    nhóm theo mã SIC, và xuất ra file cấu hình YAML.
    """
    config_data = {"sectors": {}}

    print(f"Đang đọc dữ liệu từ {excel_file}...")
    
    try:
        # Đọc file Excel bằng pandas
        df = pd.read_excel(excel_file)
        
        # Xử lý các ô trống (NaN/NaT) thành chuỗi rỗng để tránh lỗi
        df = df.fillna('')
        
        for index, row in df.iterrows():
            # Sử dụng đúng tên cột từ script crawl: 'Ticker', 'Company Name', 'CIK', 'SIC', 'Industry'
            raw_ticker = str(row.get('Ticker', '')).strip()
            
            # Làm sạch ticker
            clean_ticker = raw_ticker.replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()
            
            # Bỏ qua nếu dòng này không có Ticker
            if not clean_ticker:
                continue 

            name = str(row.get('Company Name', '')).strip()
            
            # Format CIK chuẩn 10 số. split('.')[0] để phòng trường hợp pandas hiểu nhầm thành số thập phân (VD: 1234.0)
            raw_cik = str(row.get('CIK', '')).strip().split('.')[0]
            cik = raw_cik.zfill(10)
            
            # Format SIC code (Ép về số nguyên rồi thành chuỗi để bỏ .0 nếu có)
            try:
                sic_code = str(int(row.get('SIC', 0))).strip()
            except ValueError:
                sic_code = str(row.get('SIC', '')).strip()
                
            # Mô tả ngành (Industry)
            sic_desc = str(row.get('Industry', '')).strip().replace('"', '')

            # Nhóm theo SIC
            sector_key = f"SIC_{sic_code}"
            if sector_key not in config_data["sectors"]:
                config_data["sectors"][sector_key] = {
                    "description": sic_desc,
                    "companies": []
                }

            # Thêm công ty vào nhóm tương ứng
            config_data["sectors"][sector_key]["companies"].append({
                "ticker": clean_ticker,
                "cik": cik,
                "name": name
            })
            
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{excel_file}'. Vui lòng kiểm tra lại đường dẫn.")
        return
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý file: {e}")
        return

    print("Đang xuất dữ liệu ra file YAML...")
    with open(yaml_file, 'w', encoding='utf-8') as file:
        yaml.dump(config_data, file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Hoàn tất! Đã tạo thành công file: {yaml_file}")

if __name__ == "__main__":
    # Trỏ đến file Excel vừa crawl được ở bước trước
    INPUT_EXCEL = '../sec_companies.xlsx'
    OUTPUT_YAML = '../SEC_Ticker_config.yaml'
    
    convert_excel_to_yaml(INPUT_EXCEL, OUTPUT_YAML)
