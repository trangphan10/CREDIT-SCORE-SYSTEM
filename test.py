import requests
from bs4 import BeautifulSoup

# Tải nội dung từ URL
response = requests.get('https://diem10cong.edu.vn/tom-tat-ly-thuyet-hoc-ky-2-mon-toan-lop-6-chuong-5-6')

# Kiểm tra xem yêu cầu có thành công không
if response.status_code == 200:
    # Phân tích cú pháp HTML bằng BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Bây giờ bạn có thể sử dụng soup để trích xuất thông tin từ trang
    print(soup.prettify())  # In ra HTML đã được phân tích cú pháp
else:
    print(f"Failed to retrieve content. Status code: {response.status_code}")
