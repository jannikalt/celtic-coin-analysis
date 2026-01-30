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

    st.sidebar.subheader("Retrieval")
    retrieval_enabled = st.sidebar.checkbox("Enable Similarity Retrieval", value=False)
    retrieval_top_k = st.sidebar.slider("Top-K Similar Coins", 1, 20, 5, 1)
    retrieval_dataset_path = st.sidebar.text_input(
        "Dataset TSV/CSV (optional)",
        value="",
        help="If set, build DINOv3 index on demand from this dataset",
    )
    retrieval_batch_size = st.sidebar.slider("Index Batch Size", 4, 128, 32, 4)
    rerank_enabled = st.sidebar.checkbox("Rerank with Dense Patches", value=False)
    rerank_top_k = st.sidebar.slider("Rerank Top-K", 1, 50, 20, 1)
    rerank_stride = st.sidebar.slider("Patch Stride", 1, 8, 2, 1, help="Higher stride = faster, less precise")
    rerank_weight = st.sidebar.slider("Dense Weight", 0.0, 1.0, 0.5, 0.05, help="Weight for dense rerank score")

    return {
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "border_width": border_width,
        "border_color": border_color,
        "crop_padding": crop_padding,
        "color_weight": color_weight,
        "retrieval": {
            "enabled": retrieval_enabled,
            "top_k": retrieval_top_k,
            "dataset_path": retrieval_dataset_path.strip(),
            "batch_size": retrieval_batch_size,
            "rerank": {
                "enabled": rerank_enabled,
                "top_k": rerank_top_k,
                "stride": rerank_stride,
                "weight": rerank_weight,
            },
        },
    }
