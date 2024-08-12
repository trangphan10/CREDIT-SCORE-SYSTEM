# Dự Án: Xây Dựng Quy Trình Tự Động cho Hệ Thống Chấm Điểm Tín Dụng

**Họ và tên**: Phan Thị Thu Trang

## Mô Tả Dự Án

Dự án này nhằm mục đích xây dựng một quy trình tự động cho hệ thống chấm điểm tín dụng, sử dụng Apache Airflow để quản lý và triển khai quy trình, cùng với ứng dụng Streamlit để hiển thị kết quả và giao diện người dùng.

## Hướng Dẫn Chạy Dự Án

### 1. Clone Repository
Đầu tiên, clone repository về máy:
```bash
git clone [[URL repository]](https://github.com/trangphan10/CREDIT-SCORE-SYSTEM.git)
```
### 2. Chạy quy trình tự động bằng Airflow

- Đảm bảo rằng bạn đã cài đặt Docker và Apache Airflow trên Docker.
- Chạy lệnh sau để khởi động Airflow:

  ```bash
  docker-compose up
  ```

### 3. Chạy ứng dụng Streamlit
  ```bash
streamlit run ./credit_score.py
 ```

