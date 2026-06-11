import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ==========================================
# 1. CẤU HÌNH TRANG (BẮT BUỘC ĐẦU TIÊN)
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát hiện Gian lận Giao dịch",
    page_icon="🛡️"
)

# ==========================================
# 2. HÀM CACHE DÙNG CHUNG
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ bytes để tối ưu hóa bộ nhớ cache"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            return None
        return df
    except Exception as e:
        st.error(import_error)
        return None

# ==========================================
# 3. SIDEBAR - VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # Tải dữ liệu mẫu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên dữ liệu huấn luyện (.csv, .xlsx)", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các đặc trưng từ X_1 đến X_14 và cột nhãn 'default'"
    )
    
    st.divider()
    
    # Lựa chọn mô hình AI (Notebook dùng 3 mô hình)
    model_option = st.selectbox(
        "Chọn thuật toán huấn luyện",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        help="Chọn thuật toán để xây dựng mô hình phân loại giao dịch gian lận"
    )
    
    st.subheader("Tham số mô hình AI")
    
    # Cấu hình tham số động theo mô hình được lựa chọn
    params = {}
    if model_option == "Random Forest":
        params['n_estimators'] = st.slider("Số lượng cây (n_estimators)", min_value=10, max_value=200, value=100, step=10, help="Số lượng cây quyết định trong rừng")
        params['criterion'] = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0)
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=10, help="Độ sâu tối đa của cây")
        params['random_state'] = st.number_input("Mã ngẫu nhiên (random_state)", value=42, step=1)
        
    elif model_option == "Decision Tree":
        params['criterion'] = st.selectbox("Tiêu chí đánh giá (criterion)", options=["gini", "entropy", "log_loss"], index=0)
        params['max_depth'] = st.slider("Độ sâu tối đa (max_depth)", min_value=1, max_value=30, value=5, help="Độ sâu tối đa của cây")
        params['random_state'] = st.number_input("Mã ngẫu nhiên (random_state)", value=42, step=1)
        
    elif model_option == "Logistic Regression":
        params['penalty'] = st.selectbox("Hình phạt l2 (penalty)", options=["l2", None], index=0)
        params['C'] = st.slider("Hệ số nghịch đảo điều hòa (C)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        params['max_iter'] = st.number_input("Số vòng lặp tối đa (max_iter)", value=100, step=10)
        params['random_state'] = st.number_input("Mã ngẫu nhiên (random_state)", value=42, step=1)

    # Gom tham số kiểm định vào expander nâng cao
    with st.expander("Tỷ lệ phân chia dữ liệu"):
        test_size = st.slider("Tỷ lệ tập kiểm tra (Test size)", min_value=0.1, max_value=0.5, value=0.3, step=0.05, help="Tỷ lệ dữ liệu dùng để đánh giá mô hình")

    st.divider()
    
    # Nút kích hoạt huấn luyện mô hình duy nhất
    train_clicked = st.button("🚀 Huấn luyện mô hình", type="primary", use_container_width=True)

# ==========================================
# 4. HEADER - VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Hệ thống Phát hiện Giao dịch Gian lận")
st.caption("Ứng dụng phân tích rủi ro và nhận diện tự động các giao dịch bất thường dựa trên mô hình học máy Scikit-Learn.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải file dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để bắt đầu.")
    st.stop()

# Đọc dữ liệu khi file đã được tải lên thành công
file_bytes = uploaded_file.read()
df = load_data(file_bytes, uploaded_file.name)

if df is None:
    st.error("❌ Không thể đọc file dữ liệu. Vui lòng kiểm tra lại định dạng tệp tin.")
    st.stop()

# Kiểm tra schema dữ liệu cơ bản
required_cols = [f'X_{i}' for i in range(1, 15)] + ['default']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"❌ Thiếu các cột bắt buộc sau trong file dữ liệu: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang sử dụng tệp: **{uploaded_file.name}** | Quy mô: {df.shape[0]:,} dòng và {df.shape[1]} cột.")
st.divider()

# ==========================================
# 5. KHỐI XỬ LÝ VÀ HUẤN LUYỆN (SESSION STATE)
# ==========================================
features = [f'X_{i}' for i in range(1, 15)]
target = 'default'

if train_clicked:
    with st.spinner("🔄 Đang chuẩn hóa dữ liệu và huấn luyện mô hình..."):
        # Phân tách X, y
        X = df[features]
        y = df[target]
        
        # Chia tập dữ liệu giống quy trình trong notebook
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=params.get('random_state', 42), stratify=y
        )
        
        # Khởi tạo và áp dụng StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Khởi tạo mô hình tương ứng tuyển chọn
        if model_option == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                random_state=params['random_state']
            )
        elif model_option == "Decision Tree":
            model = DecisionTreeClassifier(
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                random_state=params['random_state']
            )
        else:
            model = LogisticRegression(
                penalty=params['penalty'],
                C=params['C'],
                max_iter=params['max_iter'],
                random_state=params['random_state']
            )
            
        # Fit mô hình
        model.fit(X_train_scaled, y_train)
        
        # Đánh giá và lưu kết quả
        y_pred = model.predict(X_test_scaled)
        
        metrics_results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "cm": confusion_matrix(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True),
            "y_test": y_test.values,
            "y_pred": y_pred
        }
        
        # Lưu trữ 3 thành phần cốt lõi vào session_state
        st.session_state['trained_model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['metrics'] = metrics_results
        st.session_state['model_name'] = model_option
        
    st.success(f"🎉 Đã huấn luyện xong mô hình **{model_option}** thành công!")

# ==========================================
# 6. PHÂN VÙNG GIAO DIỆN - TABS CONTENT
# ==========================================
tabs = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa dữ liệu", 
    "🎯 Kết quả huấn luyện", 
    "🔮 Dự báo thực tế"
])

# --- TAB 1: TỔNG QUAN DỮ LIỆU ---
with tabs[0]:
    st.subheader("Phân tích cấu trúc dữ liệu")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Tổng số dòng ghi nhận", f"{df.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số đặc trưng đầu vào", f"{len(features)} biến (X_1 -> X_14)")
    with col_m3:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.metric("Dung lượng tệp tin", f"{file_size_mb:.2f} MB")
        
    st.write("##### Xem trước 5 hàng dữ liệu đầu tiên")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("##### Thống kê mô tả các đặc trưng toán học")
    # Chỉ hiển thị mô tả cho các cột tính toán mô hình nhằm tránh thừa thông tin
    st.dataframe(df[features + [target]].describe().T, use_container_width=True)

# --- TAB 2: TRỰC QUAN HÓA DỮ LIỆU ---
with tabs[1]:
    st.subheader("Phân tích phân phối và tương quan")
    
    # Biến mục tiêu ưu tiên hàng đầu
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.write("**Phân phối của biến mục tiêu (default)**")
        target_counts = df[target].value_counts().reset_index()
        target_counts.columns = ['Trạng thái', 'Số lượng']
        target_counts['Trạng thái'] = target_counts['Trạng thái'].map({0: "0 (Bình thường)", 1: "1 (Gian lận)"})
        fig_target = px.bar(target_counts, x='Trạng thái', y='Số lượng', color='Trạng thái',
                            color_discrete_map={"0 (Bình thường)": "#2ecc71", "1 (Gian lận)": "#e74c3c"},
                            height=350)
        st.plotly_chart(fig_target, use_container_width=True)
        
    with col_g2:
        st.write("**Tỷ lệ phần trăm giao dịch gian lận**")
        fig_pie = px.pie(target_counts, names='Trạng thái', values='Số lượng',
                         color='Trạng thái', color_discrete_map={"0 (Bình thường)": "#2ecc71", "1 (Gian lận)": "#e74c3c"},
                         hole=0.4, height=350)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.divider()
    
    # Lựa chọn hiển thị linh hoạt các đặc trưng X tự động
    st.write("**Biểu đồ phân phối chi tiết các đặc trưng X**")
    selected_features = st.multiselect(
        "Chọn các đặc trưng để xem phân phối:",
        options=features,
        default=['X_1', 'X_2', 'X_3', 'X_4'],
        max_selections=8
    )
    
    if selected_features:
        # Chia lưới hiển thị 2 cột cân đối
        cols_grid = st.columns(2)
        for idx, feat in enumerate(selected_features):
            col_target = cols_grid[idx % 2]
            with col_target:
                fig_hist = px.histogram(df, x=feat, color=target, barmode='overlay',
                                        title=f"Phân phối biến {feat} theo nhãn giao dịch",
                                        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                                        labels={'default': 'Nhãn'}, height=300)
                st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Vui lòng chọn ít nhất một biến X để hiển thị biểu đồ phân phối.")

# --- TAB 3: KẾT QUẢ HUÂN LUYỆN & KIỂM ĐỊNH ---
with tabs[2]:
    st.subheader("Đánh giá chất lượng mô hình phân loại")
    
    # Kiểm tra trạng thái huấn luyện từ session_state
    if 'trained_model' not in st.session_state:
        st.info("💡 Chưa có mô hình nào được huấn luyện. Vui lòng bấm nút **'Huấn luyện mô hình'** tại thanh Sidebar bên trái.")
    else:
        metrics = st.session_state['metrics']
        model_name = st.session_state['model_name']
        
        st.success(f"Mô hình đang hiển thị kết quả: **{model_name}**")
        
        # Khối hiển thị chỉ số chính bằng Metric
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Độ chính xác (Accuracy)", f"{metrics['accuracy']:.4f}")
        with m_col2:
            st.metric("Precision (Dự đoán đúng gian lận)", f"{metrics['precision']:.4f}")
        with m_col3:
            st.metric("Recall (Bỏ sót ít gian lận)", f"{metrics['recall']:.4f}")
        with m_col4:
            st.metric("F1-Score", f"{metrics['f1']:.4f}")
            
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.write("**Ma trận nhầm lẫn (Confusion Matrix):**")
            cm = metrics['cm']
            # Vẽ ma trận nhầm lẫn trực quan bằng Plotly
            fig_cm = px.imshow(
                cm, text_auto=True,
                labels=dict(x="Nhãn Dự Đoán", y="Nhãn Thực Tế", color="Số lượng"),
                x=['Bình thường (0)', 'Gian lận (1)'],
                y=['Bình thường (0)', 'Gian lận (1)'],
                color_continuous_scale='Blues',
                height=350
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_res2:
            st.write("**Báo cáo phân loại chi tiết (Classification Report):**")
            report_df = pd.DataFrame(metrics['report']).transpose()
            st.dataframe(report_df.style.format(precision=4), use_container_width=True)
            
        # Biểu đồ tầm quan trọng đặc trưng (Feature Importance) chỉ áp dụng cho cây
        if model_name in ["Random Forest", "Decision Tree"]:
            st.divider()
            st.write("**Độ quan trọng của các đặc trưng đầu vào (Feature Importance):**")
            importance = st.session_state['trained_model'].feature_importances_
            feat_imp_df = pd.DataFrame({'Đặc trưng': features, 'Độ quan trọng': importance})
            feat_imp_df = feat_imp_df.sort_values(by='Độ quan trọng', ascending=True)
            
            fig_imp = px.bar(feat_imp_df, x='Độ quan trọng', y='Đặc trưng', orientation='h',
                             title=f"Mức độ đóng góp quyết định của các đặc trưng - {model_name}",
                             color='Độ quan trọng', color_continuous_scale='Viridis', height=400)
            st.plotly_chart(fig_imp, use_container_width=True)

# --- TAB 4: SỬ DỤNG MÔ HÌNH (DỰ BÁO THỰC TẾ) ---
with tabs[3]:
    st.subheader("Dự báo thời gian thực & Chấm điểm dữ liệu")
    
    if 'trained_model' not in st.session_state:
        st.info("💡 Chưa có mô hình trực tuyến. Vui lòng cấu hình và huấn luyện mô hình thành công trước khi dự báo.")
    else:
        model = st.session_state['trained_model']
        scaler = st.session_state['scaler']
        
        mode = st.radio("Chọn phương thức dự báo đầu vào:", options=["Nhập thông số trực tiếp", "Tải file dữ liệu kiểm tra mới (Batch Prediction)"])
        
        if mode == "Nhập thông số trực tiếp":
            st.write("👉 Vui lòng điền các giá trị thông số giao dịch để phân tích rủi ro:")
            
            # Sử dụng st.form để gom cụm xử lý tránh hiện tượng tự động reload trang khi thay đổi thông số
            with st.form("single_prediction_form"):
                form_cols = st.columns(4)
                input_data = {}
                
                # Tạo widget động cho 14 biến dựa theo phân phối thực của tập dữ liệu gốc
                for idx, feat in enumerate(features):
                    col_target = form_cols[idx % 4]
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    
                    with col_target:
                        input_data[feat] = st.number_input(
                            f"Giá trị {feat}",
                            min_value=min_val * 2, # Mở rộng biên an toàn cho dữ liệu mới
                            max_value=max_val * 2,
                            value=mean_val,
                            format="%.4f"
                        )
                
                submit_predict = st.form_submit_button("🛡️ Kiểm tra giao dịch", type="primary")
                
            if submit_predict:
                # Chuyển đổi dữ liệu input sang DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Áp dụng bộ tiền xử lý StandardScaler đã fit lúc huấn luyện
                input_scaled = scaler.transform(input_df)
                
                # Thực hiện dự báo nhãn và xác suất rủi ro
                pred_label = model.predict(input_scaled)[0]
                
                st.divider()
                st.write("### Kết quả đánh giá từ hệ thống AI:")
                
                if pred_label == 1:
                    st.error("🚨 **CẢNH BÁO: Giao dịch này có dấu hiệu GIAN LẬN nguy hiểm!**")
                else:
                    st.success("✅ **AN TOÀN: Giao dịch được đánh giá là Bình thường.**")
                    
                # Hiện xác suất dự báo nếu mô hình hỗ trợ
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(input_scaled)[0]
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.metric("Xác suất an toàn (Bình thường)", f"{pred_proba[0]*100:.2f}%")
                    with col_p2:
                        st.metric("Xác suất nguy cơ (Gian lận)", f"{pred_proba[1]*100:.2f}%", 
                                  delta=f"{pred_proba[1]*100:.1f}%" if pred_proba[1] > 0.5 else None, delta_color="inverse")
                                  
        elif mode == "Tải file dữ liệu kiểm tra mới (Batch Prediction)":
            st.write("👉 Tải lên tệp chứa các đặc trưng từ `X_1` đến `X_14` (Không cần cột nhãn `default`) để chấm điểm hàng loạt.")
            
            new_file = st.file_uploader("Tải tệp cần dự báo (.csv, .xlsx)", type=["csv", "xlsx"], key="batch_file")
            
            if new_file is not None:
                new_bytes = new_file.read()
                new_df = load_data(new_bytes, new_file.name)
                
                if new_df is not None:
                    # Kiểm tra sự đầy đủ của đặc trưng đầu vào
                    missing_batch_cols = [col for col in features if col not in new_df.columns]
                    
                    if missing_batch_cols:
                        st.error(f"❌ File tải lên thiếu các cột đặc trưng kỹ thuật sau: {', '.join(missing_batch_cols)}")
                    else:
                        # Chỉ lấy đúng 14 cột phục vụ chạy mô hình
                        X_new = new_df[features]
                        
                        # Chuẩn hóa dữ liệu theo chuẩn cũ
                        X_new_scaled = scaler.transform(X_new)
                        
                        # Dự báo hàng loạt
                        predictions = model.predict(X_new_scaled)
                        
                        # Tạo dataframe kết quả trả về cho khách hàng
                        result_df = new_df.copy()
                        result_df['Dự_Báo_Kết_Quả'] = predictions
                        result_df['Ý_Nghĩa'] = result_df['Dự_Báo_Kết_Quả'].map({0: "An toàn (Bình thường)", 1: "Nguy cơ (Gian lận)"})
                        
                        if hasattr(model, "predict_proba"):
                            prob = model.predict_proba(X_new_scaled)
                            result_df['Xác_Suất_Gian_Lận'] = prob[:, 1]
                        
                        st.divider()
                        st.write("### Kết quả dự báo hàng loạt")
                        
                        # Thống kê tổng quan số lượng rủi ro trong file mới
                        num_fraud = int((predictions == 1).sum())
                        total_rows = len(predictions)
                        st.warning(f"Hệ thống phát hiện **{num_fraud} / {total_rows}** giao dịch có dấu hiệu gian lận bất thường.")
                        
                        # Hiển thị bảng dữ liệu kết quả đã xử lý
                        st.dataframe(result_df, use_container_width=True)
                        
                        # Xuất file kết quả dạng CSV sang bộ nhớ tạm để download
                        csv_buffer = io.StringIO()
                        result_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Tải xuống tệp kết quả dự báo (.CSV)",
                            data=csv_data,
                            file_name=f"ket_qua_du_bao_gian_lan_{model_name}.csv",
                            mime="text/csv"
                        )
