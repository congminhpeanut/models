import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load mô hình một lần khi ứng dụng khởi động
def load_model():
	model_path = 'best_model.h5'
	return tf.keras.models.load_model(model_path)

model = load_model()

# Hàm xử lý và dự đoán ảnh
def predict_image(image_file):
    try:
        # Đọc file ảnh từ bộ nhớ
        img_data = image_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Bước 1: Crop hình vuông 1500x1500 ưu tiên vùng trung tâm
        height, width = img.shape[:2]
        target_size_1500 = 1500
        crop_size_1500 = min(height, width, target_size_1500)
        
        if width > height:
            left_1500 = (width - crop_size_1500) // 2
            right_1500 = left_1500 + crop_size_1500
            top_1500, bottom_1500 = 0, crop_size_1500
        else:
            top_1500 = (height - crop_size_1500) // 2
            bottom_1500 = top_1500 + crop_size_1500
            left_1500, right_1500 = 0, crop_size_1500
        
        img_cropped_1500 = img[top_1500:bottom_1500, left_1500:right_1500]

        # Bước 2: Crop hình vuông 800x800 từ ảnh đã crop 1500x1500
        height_1500, width_1500 = img_cropped_1500.shape[:2]
        target_size_800 = 800
        crop_size_800 = min(height_1500, width_1500, target_size_800)
        
        if width_1500 > height_1500:
            left_800 = (width_1500 - crop_size_800) // 2
            right_800 = left_800 + crop_size_800
            top_800, bottom_800 = 0, crop_size_800
        else:
            top_800 = (height_1500 - crop_size_800) // 2
            bottom_800 = top_800 + crop_size_800
            left_800, right_800 = 0, crop_size_800
        
        img_cropped_800 = img_cropped_1500[top_800:bottom_800, left_800:right_800]

        # Resize ảnh về kích thước 800x800
        img_resized = cv2.resize(img_cropped_800, (800, 800), interpolation=cv2.INTER_LANCZOS4)

        # Chuẩn bị ảnh cho mô hình (chuyển sang RGB và thêm chiều batch)
        img_processed = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = np.array(img_processed)
        img_array = np.expand_dims(img_array, axis=0)

        # Dự đoán bằng mô hình
        prediction = model.predict(img_array)
        probabilities = prediction[0]
        sorted_indices = np.argsort(probabilities)[::-1]  # Sắp xếp từ cao đến thấp

        # Lấy thông tin 3 lớp hàng đầu
        top_idx, second_idx, third_idx = sorted_indices[:3]
        confidence_top = round(probabilities[top_idx] * 100)
        confidence_second = round(probabilities[second_idx] * 100)

        # Ánh xạ tên lớp
        class_mapping = {
            0: 'Viêm do tạp trùng hoặc tác nhân khác',
            1: 'Quang trường có sự hiện diện của clue cell',
            2: 'Quang trường có sự hiện diện của vi nấm'
        }

        # Định dạng tên lớp thứ 2
        secondary_class = class_mapping[second_idx].replace("Quang trường ", "", 1)

        # Tạo kết quả chính
        result = f"{class_mapping[top_idx]} ({confidence_top}%)"

        # Thêm dự đoán thứ 2 nếu có điều kiện
        if probabilities[second_idx] > probabilities[third_idx]:
            result += f", {secondary_class} ({confidence_second}%)"

        return result
    except Exception as e:
        return f"Lỗi: {str(e)}"

# Thiết lập tựa đề và mô tả ứng dụng
st.title("Phân loại nhanh hình ảnh soi tươi huyết trắng bằng MobileNetV2")
st.markdown("""
    Ứng dụng này sử dụng mô hình MobileNetV2 để phân tích và dự đoán nhanh sự xuất hiện của một số tác nhân gây viêm từ hình ảnh soi tươi huyết trắng.  
    Vui lòng tải lên hình ảnh chụp ở X40 để nhận kết quả dự đoán nhanh chóng!
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
