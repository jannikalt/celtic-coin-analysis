import streamlit as st
import requests
from PIL import Image
import io
import sys
import os

# Add the current directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.segmentation import CoinSegmenter
from app.core.processing import add_border, overlay_masks, crop_coins
from app.core.matching import CoinMatcher
from app.ui.sidebar import render_sidebar
from app.ui.visualization import render_visualization
from app.ui.gallery import render_gallery
from app.ui.matching_view import render_matches

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

    # Image Input
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
        # Process Image
        with st.spinner("Processing image..."):
            # 1. Add Border
            processed_image = add_border(image, config["border_width"], config["border_color"])
            
            # 2. Segment
            results = segmenter.segment_image(
                processed_image, 
                threshold=config["threshold"], 
                mask_threshold=config["mask_threshold"]
            )
            
            if results:
                masks = results.get("masks")
                boxes = results.get("boxes")
                
                # 3. Overlay Masks
                annotated_image = overlay_masks(processed_image.copy(), masks)
                
                # 4. Crop Coins and Masks
                cropped_coins, cropped_masks = crop_coins(processed_image, boxes, masks, padding=config["crop_padding"])
                
                # 5. Match Coins (if >= 3 detected)
                if len(cropped_coins) >= 3:
                    with st.spinner("Matching coins..."):
                        matches = matcher.match_coins(
                            masks, 
                            original_image=processed_image, 
                            color_weight=config.get("color_weight", 0.3)
                        )
                        render_matches(matches, cropped_coins)

                # 6. Render Results
                render_visualization(processed_image, annotated_image, len(cropped_coins))
                render_gallery(cropped_coins, cropped_masks)
            else:
                st.warning("No objects detected or model failed to load.")

if __name__ == "__main__":
    main()
