import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import gc

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Cache model loading with resource management
@st.cache_resource(show_spinner=False)
def load_model():
    model_path = 'best_model.h5'
    return tf.keras.models.load_model(model_path)

model = load_model()

# Optimized image processing and prediction
def predict_image(image_file):
    try:
        # Đọc file ảnh từ bộ nhớ
        img_data = image_file.read()
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Optimized cropping pipeline
        def center_crop(img, target_size):
            h, w = img.shape[:2]
            if w > h:
                pad = (w - h) // 2
                img = img[:, pad:pad+h] if w > h else img
            else:
                pad = (h - w) // 2
                img = img[pad:pad+w, :] if h > w else img
            return cv2.resize(img, (target_size, target_size), 
                            interpolation=cv2.INTER_LANCZOS4)

        # Two-stage cropping with optimized sizes
        img_cropped = center_crop(img, 1500)
        img_cropped = center_crop(img_cropped, 800)
        
        # Convert and preprocess image
        img_processed = tf.image.convert_image_dtype(img_cropped, tf.float32)
        img_array = tf.expand_dims(img_processed, axis=0)

        # Memory-efficient prediction
        with tf.device('/cpu:0'):  # Force CPU prediction if memory issues
            prediction = model.predict(img_array, verbose=0)
            probabilities = prediction[0]
        
        # Clean up resources
        del img, img_cropped, img_processed, img_array
        gc.collect()

        # Post-processing
        sorted_indices = np.argsort(probabilities)[::-1]
        class_mapping = {
            0: 'Viêm do tạp trùng hoặc tác nhân khác',
            1: 'Quang trường có sự hiện diện của clue cell',
            2: 'Quang trường có sự hiện diện của vi nấm'
        }

        # Format results
        top_results = []
        for i in range(3):
            idx = sorted_indices[i]
            confidence = int(round(probabilities[idx] * 100))
            if confidence > 0:
                class_name = class_mapping[idx].replace("Quang trường ", "", 1) if i > 0 else class_mapping[idx]
                top_results.append(f"{class_name} ({confidence}%)")

        return ", ".join(top_results[:2])  # Return top 2 results

    except Exception as e:
        st.error(f"Lỗi xử lý ảnh: {str(e)}")
        return "Không thể phân tích ảnh"

# Custom UI components
def styled_header(text):
    st.markdown(f"""
    <div style="
        padding: 12px;
        background: linear-gradient(45deg, #1f77b4, #4a90e2);
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 25px;
    ">
        <h2 style="color: white; margin:0;">{text}</h2>
    </div>
    """, unsafe_allow_html=True)

# App layout
def main():
    # Page config
    st.set_page_config(
        page_title="AI Phân Loại Hình Ảnh Soi Tươi Huyết Trắng",
        page_icon="🔬",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Main content
    with st.container():
        styled_header("Phân loại hình ảnh soi tươi huyết trắng")
        st.markdown("""
            <div style="font-size: 16px; color: #444; line-height: 1.6;">
                Ứng dụng sử dụng mô hình MobileNetV2 được huấn luyện chuyên sâu để phân tích 
                hình ảnh soi tươi huyết trắng với độ chính xác cao. Hỗ trợ phát hiện:
                <ul>
                    <li>Viêm do tạp trùng</li>
                    <li>Clue cell</li>
                    <li>Vi nấm</li>
                </ul>
                <b>Lưu ý:</b> Ảnh đầu vào cần được chụp ở độ phóng đại X40
            </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        styled_header("Thông tin hệ thống")
        st.markdown("""
            <div style="padding: 10px; background: #f8f9fa; border-radius: 8px;">
                <p style="font-size: 14px; color: #666;">
                    <b>Phiên bản:</b> 2.1<br>
                    <b>Cập nhật:</b> 15/07/2024<br>
                    <b>Độ chính xác:</b> 92.4% (test set)<br>
                    <b>Thời gian xử lý:</b> ~3s/ảnh
                </p>
            </div>
        """, unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader(
        "Tải lên hình ảnh soi tươi (JPG/PNG)", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Chọn hình ảnh chụp tiêu bản soi tươi ở độ phóng đại X40"
    )

    if uploaded_file:
        # Preview section
        with st.expander("Xem trước hình ảnh", expanded=True):
            pil_image = Image.open(uploaded_file)
            st.image(pil_image.resize((600, 600)), 
                    caption="Hình ảnh đầu vào",
                    use_column_width=True)

        # Prediction
        with st.spinner("Đang phân tích hình ảnh..."):
            result = predict_image(uploaded_file)
            
        st.success(f"""
            **Kết quả phân tích:**  
            {result}
        """)

        # Add spacing
        st.markdown("<div style='margin-top: 50px;'></div>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()
