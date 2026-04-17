import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.set_page_config(page_title="Auto-Inspect AI", page_icon="🚗")
st.title("🚗 Automotive Defect Detection")
st.markdown("Upload a photo of the casting part to check for manufacturing defects.")


@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('automotive_defect_detector.h5')

model = load_my_model()


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    
    st.write("🔍 Analyzing for defects...")
    
    
    img = image.resize((300, 300))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
   
    prediction = model.predict(img_array)
    
   
    if prediction[0] < 0.5:
        confidence = (1 - prediction[0][0]) * 100
        st.error(f"❌ **DEFECT DETECTED** (Confidence: {confidence:.2f}%)")
        st.write("This part should be rejected from the production line.")
    else:
        confidence = prediction[0][0] * 100
        st.success(f"✅ **QUALITY OK** (Confidence: {confidence:.2f}%)")
        st.write("This part meets the quality standards.")