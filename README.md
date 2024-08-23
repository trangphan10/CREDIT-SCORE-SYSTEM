# Dự Án: PHÁT TRIỂN HỆ THỐNG CHẤM ĐIỂM TÍN DỤNG VÀ PHÊ DUYỆT KHOẢN VAY TỰ ĐỘNG
**Họ và tên**: Phan Thị Thu Trang

## Mô Tả Dự Án
Đề tài này tập trung vào việc xây dựng và triển khai một hệ thống tự động nhằm đánh giá điểm tín dụng của khách hàng và một lowcode web phê duyệt khoản vay sau khi có dữ liệu khách hàng (đã tính điểm tín dụng).

## Các bước xây dựng và công nghệ sử dụng
1. Xây dựng bộ quy tắc cho quy trình tự động: Notebook/experiment.ipynb
2. Xây dựng quy trình tự động và lập lịch: Airflow + Docker
3. Xây dựng ứng dụng web để trực quan hoạt động mô hình: Streamlit + SQLite
4. Xây dựng ứng dụng lowcode web để phê duyệt khoản vay khách hàng: Oracle Apex

## Hướng Dẫn Chạy Dự Án

### 1. Clone Repository
Clone repository về máy:
```bash
git clone https://github.com/trangphan10/CREDIT-SCORE-SYSTEM.git
```
### 2. Chạy quy trình tự động bằng Airflow

- Đầu tiên cài đặt Docker và sau đó build image Apache Airflow trên Docker.
```bash
docker build -t apache_airflow .
```
- Chạy lệnh sau để khởi động Airflow:
```bash
docker-compose up
```

### 3. Chạy ứng dụng Streamlit
```bash
cd streamlit
streamlit run ./credit_score.py
```
### 4. Chạy ứng dụng Apex 
Truy cập vào thư mục Apex và làm theo hướng dẫn

