import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Cấu hình trang Dashboard
st.set_page_config(page_title="XAI-NIDS SOC Dashboard", layout="wide")
st.title("🛡️ SOC Dashboard: Hệ thống Phát hiện Xâm nhập (XAI-NIDS)")

# 2. Tải Mô hình & Engine Giải thích (Cache để web chạy mượt)
@st.cache_resource
def load_artifacts():
    # Khởi tạo cái "vỏ" XGBoost và nhét "lõi" JSON vào
    model = xgb.XGBClassifier()
    model.load_model('xgboost_nids.json')
    
    # SHAP vẫn xài file PKL vì nó ổn định
    explainer = shap.TreeExplainer(model)
    
    return model, explainer

model, explainer = load_artifacts()

# 3. Khu vực Upload Dữ liệu (Sidebar bên trái)
st.sidebar.header("📁 Nhập Dữ liệu Luồng mạng")
uploaded_file = st.sidebar.file_uploader("Tải lên file CSV dữ liệu (VD: test_traffic.csv)", type=['csv'])

if uploaded_file:
    # Đọc dữ liệu
    df = pd.read_csv(uploaded_file)
    st.write("### 🔍 Giám sát luồng dữ liệu (Live Traffic Preview)")
    st.dataframe(df.head(10)) # Hiển thị 10 dòng đầu cho gọn

    st.write("---")
    
    # ================= BƯỚC TIỀN XỬ LÝ DỮ LIỆU =================
    # 1. Dùng pandas lọc lấy các cột số y chang bước làm sạch ở Colab
    df_processed = df.select_dtypes(include=[np.number])
    
    # 2. Tự động gỡ luôn cột nhãn (nếu mày lỡ up file data gốc thay vì file test)
    for col in ['Attack_type', 'Attack_label', 'label', 'Label']:
        if col in df_processed.columns:
            df_processed = df_processed.drop(columns=[col])
    # ==========================================================

    # 4. Engine Phân tích & Báo động
    if st.button("🚀 Quét Rủi Ro (Scan Traffic)"):
        with st.spinner('AI đang phân tích các gói tin...'):
            # CHÚ Ý: Dùng df_processed (bảng đã lọc) để dự đoán
            predictions = model.predict(df_processed)
            malicious_count = sum(predictions)
            total_count = len(predictions)
            
            # Hiển thị thống kê tổng quan
            col1, col2 = st.columns(2)
            col1.metric("Tổng số gói tin (Flows)", total_count)
            col2.metric("🚨 Phát hiện Tấn công (Attacks)", malicious_count, delta_color="inverse")

    st.write("---")
    
    # 5. Engine Giải thích XAI (Click-to-Explain)
    st.write("### 🕵️‍♂️ Module XAI: Giải thích nguyên nhân chặn")
    st.info("💡 Chọn một ID gói tin bị nghi ngờ để xuất Báo cáo SHAP Waterfall Plot.")
    
    # Cho phép người dùng chọn gói tin theo dòng (Row Index)
    packet_idx = st.number_input("Nhập ID gói tin (Row Index):", min_value=0, max_value=len(df)-1, value=0)
    
    # Lấy đúng định dạng gói tin đã xử lý cho AI giải thích
    packet_data = df_processed.iloc[[packet_idx]]
    pred = model.predict(packet_data)[0]
    
    if pred == 1:
        st.error(f"Gói tin số {packet_idx} được hệ thống AI chẩn đoán là: **MÃ ĐỘC (Attack)**")
    else:
        st.success(f"Gói tin số {packet_idx} được hệ thống AI chẩn đoán là: **BÌNH THƯỜNG (Benign)**")
    
    # Vẽ biểu đồ Waterfall giải thích
    shap_values = explainer(packet_data)
    
    # Khởi tạo khung hình mới và vẽ
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    
    # Ép Streamlit lấy cái hình vừa vẽ xong
    st.pyplot(fig)
    
    # Xóa bộ nhớ đệm để không bị chồng hình cho lần bấm sau
    plt.clf() 
    
else:
    st.info("Vui lòng tải lên một file dữ liệu lưu lượng mạng (CSV) ở thanh bên trái để bắt đầu mô phỏng hệ thống SOC.")