**Ứng dụng lowcode web phê duyệt khoản vay** 
# Cách chạy thủ công 
1. Truy cập vào link sau để yêu cầu 1 workspace: https://apex.oracle.com/pls/apex/r/apex/sign-up/request-workspace?session=10859138611131\
2. Truy cập vào workspace
3. Tạo tài khoản: Truy cập vào biểu tượng người bên góc phải trên cùng, chọn Manages User and Group, tạo 2 tài khoản người dùng 
- MANAGER - Nhập mail và mật khẩu 
- STAFF - Nhập mail và mật khẩu
3. Truy cập AppBuilder và chọn Import. Thêm file app.sql vào phần Drag and Drop, chọn Next liên tiếp để tạo ứng dụng
4. Truy cập SQL Workshop, vào phần SQL Scripts. Chọn Upload để thêm schema.sql, sau khi đã tạo thành công, nhấn RUN để tạo bảng và trigger.
5. Vào AppBuilder, truy cập vào ứng dụng credit_score vừa load. Truy cập vào Application Access Control, nhấn Add User Role Assignment và nhập vào các username vừa tạo với các vai trò: 
- MANAGER: Administrator
- STAFF: Contributor, Reader
# Cách chạy trực tuyến
1. Truy cập theo đường link: https://apex.oracle.com/pls/apex/r/chanchan10/credit-score/login?session=14642550382489
2. Đăng nhập vào tài khoản: 
- Quản trị: Phê duyệt yêu cầu cho vay và xem dữ liệu lịch sử
+ Tài khoản: MANAGER
+ Mật khẩu: thunghiem
- Nhân viên: Upload tệp tin chứa dữ liệu tín dụng khách hàng và xem dữ liệu lịch sử 
+ Tài khoản: STAFF
+ Mật khẩu: Ph@ntr@n9
