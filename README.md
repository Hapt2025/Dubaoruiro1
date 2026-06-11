# 🛡️ Ứng dụng Phát hiện Giao dịch Gian lận (Fraud Detection Web App)

Ứng dụng web được phát triển bằng thư viện **Streamlit**, chuyển đổi toàn bộ quy trình phân tích và huấn luyện từ Notebook Học máy mã nguồn mở sang giao diện trực quan thân thiện với người vận hành.

## 🎯 Chức năng chính của ứng dụng
1. **Tổng quan dữ liệu**: Xem nhanh cấu trúc phân phối, kiểm tra chất lượng dữ liệu thô và các chỉ số thống kê mô tả.
2. **Trực quan hóa**: Biểu đồ phân phối chi tiết các biến đặc trưng đầu vào (`X_1` đến `X_14`) phân tách rõ nét theo nhãn giao dịch để phát hiện hành vi bất thường.
3. **Huấn luyện mô hình tối ưu**: Cho phép tùy chỉnh các siêu tham số linh hoạt của 3 thuật toán phổ biến:
   - *Random Forest Classifier*
   - *Decision Tree Classifier*
   - *Logistic Regression*
4. **Đánh giá kiểm định**: Hiển thị báo cáo chi tiết trực quan về chất lượng mô hình (Accuracy, Precision, Recall, F1-Score, Ma trận nhầm lẫn Confusion Matrix, Feature Importance).
5. **Dự báo thông minh**: Hỗ trợ đồng thời 2 chế độ: Nhập thông số trực tiếp hoặc Tải file dữ liệu kiểm tra mới để dự báo hàng loạt (Batch Prediction) và tải về báo cáo kết quả.

---

## 🛠️ Hướng dẫn cài đặt và vận hành

### 1. Yêu cầu môi trường
* Máy tính đã cài đặt Python phiên bản từ `3.9` đến `3.12`.

### 2. Các bước triển khai cụ thể

**Bước 2.1: Di chuyển vào thư mục chứa mã nguồn ứng dụng và cài đặt các thư viện bổ trợ**
```bash
pip install -r requirements.txt
