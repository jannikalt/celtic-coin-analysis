import streamlit as st

def render_sidebar():
    st.sidebar.header("Configuration")
    
    st.sidebar.subheader("Segmentation Parameters")
    threshold = st.sidebar.slider("Threshold", 0.0, 1.0, 0.5, 0.01, help="Confidence threshold for segmentation.")
    mask_threshold = st.sidebar.slider("Mask Threshold", 0.0, 1.0, 0.45, 0.01, help="Threshold for mask binarization.")
    
    st.sidebar.subheader("Image Pre-processing")
    border_width = st.sidebar.slider("Border Width", 0, 500, 200, 10, help="Width of the border to add around the image.")
    border_color = st.sidebar.color_picker("Border Fill Color", "#FFFFFF", help="Color of the added border.")
    
    st.sidebar.subheader("Cropping")
    crop_padding = st.sidebar.slider("Crop Padding (px)", 0, 100, 10, 1, help="Padding around the detected object when cropping.")
    
    st.sidebar.subheader("Matching")
    color_weight = st.sidebar.slider("Color Weight", 0.0, 1.0, 0.3, 0.05, help="Weight of color similarity in matching score (vs edge similarity).")

    return {
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "border_width": border_width,
        "border_color": border_color,
        "crop_padding": crop_padding,
        "color_weight": color_weight
    }
