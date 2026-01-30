import streamlit as st
from PIL import Image

def render_matches(matches: list, cropped_images: list[Image.Image]):
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(cropped_images[i], caption=f"Coin {i + 1}", width="stretch")
        with col2:
            st.image(cropped_images[j], caption=f"Coin {j + 1}", width="stretch")
        
        st.divider()
