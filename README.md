# Dự Án: PHÁT TRIỂN HỆ THỐNG CHẤM ĐIỂM TÍN DỤNG VÀ PHÊ DUYỆT KHOẢN VAY TỰ ĐỘNG
**Họ và tên**: Phan Thị Thu Trang

## Mô Tả Dự Án
Đề tài này tập trung vào việc xây dựng và triển khai một hệ thống tự động nhằm đánh giá điểm tín dụng của khách hàng và một lowcode web phê duyệt khoản vay sau khi có dữ liệu khách hàng (đã tính điểm tín dụng).

## Bộ dữ liệu : Data/hmeq.csv 
- **Mô tả**: Bộ dữ liệu gồm 5960 bản ghi với 13 trường dữ liệu. Mỗi bản ghi đại diện cho thông tin khách hàng có thẻ tín dụng ở ngân hàng. Mỗi người được phân loại tín dụng tốt hay xấu dựa trên một tập thuộc tính.

- **Các trường dữ liệu**:
  - **BAD**:
    - 1: Người nộp đơn đã vỡ nợ hoặc nợ quá hạn nghiêm trọng
    - 0: Người nộp đơn đã thanh toán khoản vay
  - **LOAN**: Số tiền yêu cầu vay
  - **MORTDUE**: Số tiền nợ còn lại trên khoản thế chấp hiện tại
  - **VALUE**: Giá trị tài sản hiện tại
  - **REASON**: Lý do vay
    - DebtCon: Debt consolidation (ghép nợ)
    - HomeImp: Home improvement (sửa sang nhà cửa)
  - **JOB**: Loại công việc, bao gồm các nhóm:
    - "Office"
    - "Sales"
    - "Manager"
    - "Professional Executive"
    - "Other"
  - **YOJ**: Số năm làm việc tại công việc hiện tại
  - **DEROG**: Số lượng báo cáo vỡ nợ
  - **DELINQ**: Số hạn mức tín dụng quá hạn
  - **CLAGE**: Tuổi của hạn mức tín dụng lâu nhất tính theo tháng
  - **NINQ**: Số lượng yêu cầu tín dụng gần đây
  - **CLNO**: Số lượng hạn mức tín dụng
  - **DEBTINC**: Tỷ lệ nợ/thu nhập

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

