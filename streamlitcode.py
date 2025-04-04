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
        #img_resized = cv2.resize(img_cropped_800, (800, 800), interpolation=cv2.INTER_LANCZOS4)
        
        # Convert and preprocess image
        img_processed = cv2.cvtColor(img_cropped_800, cv2.COLOR_BGR2RGB)
        
        img_array = tf.expand_dims(img_processed, axis=0)
        
        #img_array = np.array(img_processed)
        #img_array = np.expand_dims(img_array, axis=0)

        # Memory-efficient prediction
        with tf.device('/cpu:0'):  # Force CPU prediction if memory issues
            prediction = model.predict(img_array, verbose=0)
            probabilities = prediction[0]

        # Tính toán khuyến nghị
        bacterial_prob = probabilities[0]
        clue_prob = probabilities[1]
        fungus_prob = probabilities[2]

        if fungus_prob >= 0.6:
            if bacterial_prob >= 0.2:
                recommendation = "Viêm do nhiễm khuẩn, vi nấm (+)"
            else:
                recommendation = "Vi nấm (+)"
        elif bacterial_prob >= 0.6:
            recommendation = "Viêm do nhiễm khuẩn"
        elif clue_prob >= 0.6:
            recommendation = "Clue cell (+)"
        else:
            recommendation = "Viêm do nhiễm khuẩn hoặc tác nhân khác"
        
        # Clean up resources
        del img, img_cropped_1500, img_cropped_800, img_processed, img_array, fungus_prob, bacterial_prob, clue_prob
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

        return recommendation, ", ".join(top_results[:2])  # Return top 2 results

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
        styled_header("Thông tin phần mềm")
        st.markdown("""
            <div style="padding: 10px; background: #f8f9fa; border-radius: 8px;">
                <p style="font-size: 14px; color: #666;">
                    <b>Phiên bản:</b> 1.1<br>
                    <b>Cập nhật:</b> 04/4/2025<br>
                    <b>Tác giả:</b> Nguyễn Trương Công Minh<br>
                    <b>Độ chính xác:</b> AUC-PR (avg) 87% (val + test set)<br>
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
            #pil_image = Image.open(uploaded_file)
            #st.image(pil_image.resize((600, 600)), 
            #        caption="Hình ảnh đầu vào",
            #        use_column_width=True)
            st.image(uploaded_file, caption="Hình ảnh đã tải lên", use_column_width=True)

        # Prediction
        with st.spinner("Đang phân tích hình ảnh..."):
            recommendation, result = predict_image(uploaded_file)
            
        st.success(f"""
            **Khuyến nghị:** {recommendation}
            \n**Chi tiết:** {result}
        """)

        # Add spacing
        st.markdown("<div style='margin-top: 50px;'></div>", 
                   unsafe_allow_html=True)

if __name__ == "__main__":
    main()
