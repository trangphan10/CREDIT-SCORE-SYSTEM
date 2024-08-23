# Dự Án: Xây Dựng Quy Trình Tự Động cho Hệ Thống Chấm Điểm Tín Dụng

**Họ và tên**: Phan Thị Thu Trang

## Mô Tả Dự Án

Dự án này tập trung vào việc phát triển một quy trình tự động cho hệ thống chấm điểm tín dụng sử dụng Apache Airflow để quản lý và triển khai quy trình tự động, đồng thời áp dụng Streamlit để trực quan hóa và giám sát hoạt động của hệ thống.

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
streamlit run ./credit_score.py
```

