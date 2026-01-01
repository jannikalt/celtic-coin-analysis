import streamlit as st
from PIL import Image
import io

def render_gallery(cropped_images: list[Image.Image], cropped_masks: list[Image.Image]):
    st.subheader("Detected Coins")
    
    if not cropped_images:
        st.info("No coins detected yet.")
        return

    # Display in a grid
    cols = st.columns(4)
    
    for idx, (img, mask) in enumerate(zip(cropped_images, cropped_masks)):
        col_img = cols[idx % 4]
        with col_img:
            st.image(img, caption=f"Coin {idx + 1}", width="stretch")
            
            # Create a download button for each image
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label=f"Download #{idx + 1}",
                data=byte_im,
                file_name=f"coin_{idx + 1}.png",
                mime="image/png",
                key=f"download_{idx}"
            )

            st.image(mask, caption=f"Mask {idx + 1}", width="stretch")
            # Create a download button for each mask
            buf_mask = io.BytesIO()
            mask.save(buf_mask, format="PNG")
            byte_mask_im = buf_mask.getvalue()

            st.download_button(
                label=f"Download Mask #{idx + 1}",
                data=byte_mask_im,
                file_name=f"coin_mask_{idx + 1}.png",
                mime="image/png",
                key=f"download_mask_{idx}"
            )
