import streamlit as st
from PIL import Image

def render_visualization(original_image: Image.Image, processed_image: Image.Image, num_detected: int):
    st.subheader("Segmentation Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_image, caption="Original Image", width="stretch")
        
    with col2:
        st.image(processed_image, caption=f"Segmented Image ({num_detected} coins detected)", width="stretch")