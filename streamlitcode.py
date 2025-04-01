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

        # Giảm kích thước ảnh xuống 800x800, giữ chi tiết vùng trung tâm
        height, width = img.shape[:2]
        if width > height:
            left = (width - height) // 2
            right = left + height
            top, bottom = 0, height
        else:
            top = (height - width) // 2
            bottom = top + width
            left, right = 0, width
        img_cropped = img[top:bottom, left:right]
        img_resized = cv2.resize(img_cropped, (800, 800), interpolation=cv2.INTER_LANCZOS4)

        # Tăng độ rõ nét ảnh bằng bộ lọc
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Bộ lọc làm sắc nét
        img_sharpened = cv2.filter2D(img_resized, -1, kernel)

        # Chuẩn bị ảnh cho mô hình
        img_processed = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB
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

# Giao diện Streamlit
st.title("Dự đoán hình ảnh online")

uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Hiển thị hình ảnh đã tải lên
    st.image(uploaded_file, caption='Hình ảnh đã tải lên.', use_column_width=True)
    st.write("")
    st.write("Đang dự đoán...")

    # Thực hiện dự đoán
    result = predict_image(uploaded_file)
    st.write(f"Kết quả dự đoán: {result}")