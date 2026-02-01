import streamlit as st
import requests
from PIL import Image
import io
import sys
import os
from pathlib import Path
import pandas as pd

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
from app.core.classification import build_classifier_from_checkpoint
from app.core.side_classification import build_side_classifier_from_checkpoint
from app.ui.sidebar import render_sidebar
from app.ui.visualization import render_visualization
from app.ui.gallery import render_gallery
from app.ui.matching_view import render_matches
from app.ui.retrieval_view import render_retrieval_results
from app.ui.dataset_view import render_dataset_viewer

# Page Configuration
st.set_page_config(
    page_title="Celtic Coin Analysis",
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
    st.title("Celtic Coin Analysis")
    st.markdown("Upload an image or provide a URL to automatically segment coins using SAM3.")

    # Initialize Segmenter
    segmenter = CoinSegmenter()
    matcher = CoinMatcher()

    # Sidebar Configuration
    config = render_sidebar()
    
    # Load side classifier if enabled
    side_classifier = None
    if config["side_classification"]["enabled"] and config["side_classification"]["model_dir"]:
        try:
            with st.spinner("Loading side classifier..."):
                side_classifier = build_side_classifier_from_checkpoint(
                    config["side_classification"]["model_dir"],
                    device="cuda"
                )
            st.sidebar.success("✓ Side classifier loaded")
        except Exception as e:
            st.sidebar.error(f"Failed to load side classifier: {e}")

    tab_segmenter, tab_retrieval, tab_dataset = st.tabs(["Segmenter", "Retrieval", "Dataset Viewer"])

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
                            render_matches(matches, cropped_coins, cropped_masks, side_classifier)
                    elif len(cropped_coins) == 2:
                        # For exactly 2 coins, show them as a pair with side classification
                        st.info("Found exactly 2 coins - displaying as a pair")
                        
                        # Create a dummy match for the pair
                        matches = [{
                            'indices': (0, 1),
                            'score': 0.0,
                            'edge_score': 0.0,
                            'color_score': 0.0,
                            'angle': 0,
                            'flipped': False
                        }]
                        render_matches(matches, cropped_coins, cropped_masks, side_classifier)

                    render_visualization(processed_image, annotated_image, len(cropped_coins))
                    render_gallery(cropped_coins, cropped_masks)
                else:
                    st.warning("No objects detected or model failed to load.")

    with tab_retrieval:
        st.subheader("Instance Retrieval & Classification")
        st.markdown("Provide obverse and/or reverse images for retrieval or classification.")

        col_left, col_right = st.columns(2)
        with col_left:
            obv_file = st.file_uploader("Obverse image", type=["jpg", "jpeg", "png", "webp"], key="obv_retrieval")
        with col_right:
            rev_file = st.file_uploader("Reverse image", type=["jpg", "jpeg", "png", "webp"], key="rev_retrieval")

        obv_image = Image.open(obv_file).convert("RGB") if obv_file is not None else None
        rev_image = Image.open(rev_file).convert("RGB") if rev_file is not None else None

        # Display uploaded images
        if obv_image or rev_image:
            cols = st.columns(2)
            if obv_image:
                with cols[0]:
                    st.image(obv_image, caption="Obverse", width='stretch')
            if rev_image:
                with cols[1]:
                    st.image(rev_image, caption="Reverse", width='stretch')

        retrieval_cfg = config.get("retrieval", {})
        classification_cfg = config.get("classification", {})
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            run_retrieval = st.button("Run Retrieval", type="primary" if retrieval_cfg.get("enabled") else "secondary")
        with col2:
            run_classification = st.button("Run Classification", type="primary" if classification_cfg.get("enabled") else "secondary")
        
        # Classification
        if run_classification:
            if not classification_cfg.get("enabled"):
                st.warning("Enable classification in the sidebar first.")
            elif not classification_cfg.get("model_dir"):
                st.warning("Specify a model directory in the sidebar.")
            elif obv_image is None and rev_image is None:
                st.warning("Please upload at least one image.")
            else:
                with st.spinner("Loading classifier..."):
                    try:
                        classifier = build_classifier_from_checkpoint(
                            model_dir=classification_cfg["model_dir"],
                            device="cuda" if st.session_state.get("use_gpu", True) else "cpu",
                        )
                        
                        with st.spinner("Classifying..."):
                            result = classifier.predict(obverse=obv_image, reverse=rev_image)
                            
                            st.success(f"**Predicted Class:** {result['predicted_class']}")
                            st.metric("Confidence", f"{result['confidence']:.2%}")
                            
                            st.subheader("Top Predictions")
                            top_probs = list(result["probabilities"].items())
                            
                            for class_name, prob in top_probs:
                                st.progress(prob, text=f"{class_name}: {prob:.2%}")
                            
                            with st.expander("View All Class Probabilities"):
                                import pandas as pd
                                all_probs = sorted(
                                    result["all_probabilities"].items(),
                                    key=lambda x: x[1],
                                    reverse=True
                                )
                                df = pd.DataFrame(all_probs, columns=["Class", "Probability"])
                                df["Probability"] = df["Probability"].apply(lambda x: f"{x:.4f}")
                                st.dataframe(df, width='stretch')
                                
                    except Exception as e:
                        st.error(f"Classification failed: {e}")
                        import traceback
                        with st.expander("Error details"):
                            st.code(traceback.format_exc())
        
        # Retrieval
        if run_retrieval:
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

    with tab_dataset:
        dataset_path = retrieval_cfg.get("dataset_path", "")
        render_dataset_viewer(dataset_path)

if __name__ == "__main__":
    main()
