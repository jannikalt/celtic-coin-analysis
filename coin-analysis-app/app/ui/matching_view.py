import streamlit as st
from PIL import Image
import numpy as np


def render_matches(matches: list, cropped_images: list[Image.Image], cropped_masks: list = None, side_classifier=None):
    """
    Render matched coin pairs with optional side classification.
    
    Args:
        matches: List of match dictionaries
        cropped_images: List of cropped coin images
        cropped_masks: Optional list of cropped coin masks for side classification
        side_classifier: Optional SideClassifier instance
    """
    st.subheader("Matched Coin Pairs")
    
    if not matches:
        st.info("No matches found or not enough coins to match.")
        return

    for idx, match in enumerate(matches):
        i, j = match['indices']
        score = match['score']
        score_edge = match['edge_score']
        score_color = match['color_score']
        angle = match['angle']
        flipped = match.get('flipped', False)
        
        flip_text = ", Flipped" if flipped else ""
        st.markdown(f"**Match {idx + 1}** (Score: {score:.2f}, Edge Score: {score_edge:.2f}, Color Score: {score_color:.2f}, Rotation: {angle}°{flip_text})")
        
        # Perform side classification if available
        side_result = None
        if side_classifier is not None and cropped_masks is not None:
            try:
                # Convert PIL images to numpy arrays (BGR for OpenCV)
                img_a = np.array(cropped_images[i].convert('RGB'))[:, :, ::-1]
                img_b = np.array(cropped_images[j].convert('RGB'))[:, :, ::-1]
                
                # Get masks (convert to grayscale if needed)
                mask_a = cropped_masks[i]
                mask_b = cropped_masks[j]
                
                # Ensure masks are grayscale numpy arrays
                if isinstance(mask_a, Image.Image):
                    mask_a = np.array(mask_a.convert('L'))
                if isinstance(mask_b, Image.Image):
                    mask_b = np.array(mask_b.convert('L'))
                
                side_result = side_classifier.predict_side_order(img_a, img_b, mask_a, mask_b)
                
            except Exception as e:
                st.warning(f"Side classification failed: {e}")
        
        # Display images with side classification info
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(cropped_images[i], caption=f"Coin {i + 1}", width="stretch")
            if side_result:
                # Determine which side this coin is based on predicted order
                if side_result['predicted_order'] == 'obv-rev':
                    st.info(f"Obverse (conf: {side_result['confidence']:.1%})")
                else:
                    st.info(f"Reverse (conf: {side_result['confidence']:.1%})")
        
        with col2:
            st.image(cropped_images[j], caption=f"Coin {j + 1}", width="stretch")
            if side_result:
                # Determine which side this coin is based on predicted order
                if side_result['predicted_order'] == 'obv-rev':
                    st.info(f"Reverse (conf: {side_result['confidence']:.1%})")
                else:
                    st.info(f"Obverse (conf: {side_result['confidence']:.1%})")
        
        # Show detailed side classification info
        if side_result:
            with st.expander("Side Classification Details"):
                st.write(f"**Predicted Order:** {side_result['predicted_order']}")
                st.write(f"**Confidence:** {side_result['confidence']:.2%}")
                st.write("**Probabilities:**")
                for order, prob in side_result['probabilities'].items():
                    st.write(f"  - {order}: {prob:.2%}")
        
        st.divider()
