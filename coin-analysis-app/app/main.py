import streamlit as st
import requests
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Add the current directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add coin-classification to sys.path for retrieval model imports
root_dir = Path(__file__).resolve().parents[2]
coin_classification_path = root_dir / "coin-classification"
if coin_classification_path.exists():
    sys.path.append(str(coin_classification_path))

from app.core.segmentation import CoinSegmenter
from app.core.processing import add_border, overlay_masks, crop_coins
from app.core.matching import CoinMatcher
from app.core.retrieval import build_dinov3_retriever_from_dataset
from app.ui.sidebar import render_sidebar
from app.ui.visualization import render_visualization
from app.ui.gallery import render_gallery
from app.ui.matching_view import render_matches
from app.ui.retrieval_view import render_retrieval_results

# Page Configuration
st.set_page_config(
    page_title="Celtic Coin Segmenter",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        h1 {
            color: #2c3e50;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
        }
        .stDownloadButton>button {
            width: 100%;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")
        return None

def main():
    st.title("Celtic Coin Segmenter")
    st.markdown("Upload an image or provide a URL to automatically segment coins using SAM3.")

    # Initialize Segmenter
    segmenter = CoinSegmenter()
    matcher = CoinMatcher()

    # Sidebar Configuration
    config = render_sidebar()

    tab_segmenter, tab_retrieval = st.tabs(["Segmenter", "Retrieval"])

    with tab_segmenter:
        st.subheader("Segmentation")
        input_method = st.radio("Select Input Method", ["Upload Image", "Image URL"], horizontal=True)

        image = None
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            url = st.text_input("Enter Image URL")
            if url:
                image = load_image_from_url(url)

        if image:
            with st.spinner("Processing image..."):
                processed_image = add_border(image, config["border_width"], config["border_color"])

                results = segmenter.segment_image(
                    processed_image,
                    threshold=config["threshold"],
                    mask_threshold=config["mask_threshold"],
                )

                if results:
                    masks = results.get("masks")
                    boxes = results.get("boxes")

                    annotated_image = overlay_masks(processed_image.copy(), masks)
                    cropped_coins, cropped_masks = crop_coins(
                        processed_image, boxes, masks, padding=config["crop_padding"]
                    )

                    if len(cropped_coins) >= 3:
                        with st.spinner("Matching coins..."):
                            matches = matcher.match_coins(
                                masks,
                                original_image=processed_image,
                                color_weight=config.get("color_weight", 0.3),
                            )
                            render_matches(matches, cropped_coins)

                    render_visualization(processed_image, annotated_image, len(cropped_coins))
                    render_gallery(cropped_coins, cropped_masks)
                else:
                    st.warning("No objects detected or model failed to load.")

    with tab_retrieval:
        st.subheader("Instance Retrieval")
        st.markdown("Provide obverse and/or reverse images for retrieval.")

        col_left, col_right = st.columns(2)
        with col_left:
            obv_file = st.file_uploader("Obverse image", type=["jpg", "jpeg", "png", "webp"], key="obv_retrieval")
        with col_right:
            rev_file = st.file_uploader("Reverse image", type=["jpg", "jpeg", "png", "webp"], key="rev_retrieval")

        obv_image = Image.open(obv_file).convert("RGB") if obv_file is not None else None
        rev_image = Image.open(rev_file).convert("RGB") if rev_file is not None else None

        retrieval_cfg = config.get("retrieval", {})
        if st.button("Run Retrieval", type="primary"):
            if not retrieval_cfg.get("enabled"):
                st.warning("Enable retrieval in the sidebar first.")
            elif obv_image is None and rev_image is None:
                st.warning("Please upload at least one image.")
            else:
                config_path = retrieval_cfg.get("config_path")
                embeddings_path = retrieval_cfg.get("embeddings_path")
                metadata_path = retrieval_cfg.get("metadata_path")
                checkpoint_override = retrieval_cfg.get("checkpoint_override") or None
                dataset_path = retrieval_cfg.get("dataset_path")
                batch_size = retrieval_cfg.get("batch_size", 32)

                with st.spinner("Retrieving similar coins..."):
                    try:
                        if dataset_path:
                            retriever = build_dinov3_retriever_from_dataset(
                                data_path=dataset_path,
                                model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                                input_size=224,
                                batch_size=batch_size,
                            )

                        rerank_cfg = retrieval_cfg.get("rerank", {})
                        matches = retriever.search_images(
                            obv_image=obv_image,
                            rev_image=rev_image,
                            top_k=retrieval_cfg.get("top_k", 5),
                            rerank_top_k=rerank_cfg.get("top_k") if rerank_cfg.get("enabled") else None,
                            patch_stride=rerank_cfg.get("stride", 1),
                            dense_weight=rerank_cfg.get("weight", 0.5),
                        )
                        query_images = []
                        if obv_image is not None:
                            query_images.append(obv_image)
                        if rev_image is not None:
                            query_images.append(rev_image)
                        results_by_coin = [
                            {
                                "query_index": 0,
                                "query_image": obv_image or rev_image,
                                "query_images": query_images if len(query_images) > 1 else None,
                                "matches": matches,
                            }
                        ]
                        display_views = "both_concat" if (obv_image is not None and rev_image is not None) else (
                            "obv" if obv_image is not None else "rev"
                        )
                        render_retrieval_results(results_by_coin, retrieval_cfg.get("top_k", 5), display_views)
                    except Exception as e:
                        st.error(f"Retrieval failed: {e}")

if __name__ == "__main__":
    main()
