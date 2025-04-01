import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load mô hình một lần khi ứng dụng khởi động
model_path = 'best_model.h5'  # Thay đổi thành đường dẫn thực tế đến file best_model.h5
model = tf.keras.models.load_model(model_path)

# Hàm xử lý và dự đoán ảnh
def predict_image(image_file):
    try:
        # Đọc file ảnh từ bộ nhớ
        img_data = image_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize trực tiếp thành 800x800, bỏ bước crop
        img_resized = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LANCZOS4)

        # Chuẩn bị ảnh cho mô hình (bỏ sharpening để nhất quán với huấn luyện)
        img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
        img_array = np.array(img_processed)  # Chuyển thành mảng NumPy
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch

        # Dự đoán bằng mô hình
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        if predicted_class == 0:
            predicted_class = 'Viêm do tạp trùng hoặc tác nhân khác'
        elif predicted_class == 1:
            predicted_class = 'Viêm do clue cell'
        elif predicted_class == 2:
            predicted_class = 'Viêm do vi nấm'

        return predicted_class
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Thiết lập tựa đề và mô tả ứng dụng
st.title("Dự đoán hình ảnh soi tươi huyết trắng bằng MobileNetV2")
st.markdown("""
    Ứng dụng này sử dụng mô hình MobileNetV2 để phân tích và dự đoán các loại viêm từ hình ảnh soi tươi huyết trắng.  
    Vui lòng tải lên hình ảnh để nhận kết quả dự đoán nhanh chóng!
""")

# Sidebar chứa thông tin bổ sung
st.sidebar.header("Thông tin ứng dụng")
st.sidebar.markdown("""
    - **Tác giả**: Nguyễn Trương Công Minh
    - **Phiên bản**: 1.0
    - **Công nghệ**: MobileNetV2  
    - **Mục đích**: Hỗ trợ dự đoán viêm âm đạo từ hình ảnh soi tươi huyết trắng.
""")

# Phần tải lên hình ảnh
uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    st.image(uploaded_file, caption="Hình ảnh đã tải lên", use_column_width=True)
    st.write("")  # Thêm khoảng trống để giao diện thoáng hơn
    st.info("Đang dự đoán...")  # Thông báo trạng thái bằng hộp thông tin

    # Thực hiện dự đoán (giả định hàm predict_image đã được định nghĩa)
    result = predict_image(uploaded_file)
    st.success(f"Kết quả dự đoán: {result}")  # Hiển thị kết quả nổi bật

# Thêm CSS tùy chỉnh để cải thiện thẩm mỹ
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f2f6;  /* Màu nền nhẹ nhàng */
    }
    h1 {
        color: #1f77b4;  /* Màu xanh chuyên nghiệp cho tiêu đề */
        font-family: 'Arial', sans-serif;
    }
    .stMarkdown p {
        color: #333333;  /* Màu chữ tối cho mô tả */
        font-size: 16px;
    }
    .stImage {
        border: 2px solid #ddd;  /* Viền nhẹ cho hình ảnh */
        border-radius: 8px;  /* Bo góc hình ảnh */
        padding: 5px;
        background-color: #ffffff;
    }
    .stSuccess {
        font-size: 18px;  /* Tăng kích thước chữ kết quả */
        font-weight: bold;  /* In đậm kết quả */
    }
    .stSidebar .sidebar-content {
        background-color: #ffffff;  /* Sidebar màu trắng */
        border-right: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)
