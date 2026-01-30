import torch
from transformers import Sam3Processor, Sam3Model
import streamlit as st
from PIL import Image
import gc

class CoinSegmenter:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    @st.cache_resource
    def load_model(_self):
        """
        Loads the SAM3 model and processor. 
        Cached by Streamlit to avoid reloading on every run.
        """
        try:
            model = Sam3Model.from_pretrained("facebook/sam3").to(_self.device)
            processor = Sam3Processor.from_pretrained("facebook/sam3")
            return model, processor
        except Exception as e:
            st.error(f"Error loading SAM3 model: {e}")
            return None, None

    def initialize(self):
        if self.model is None or self.processor is None:
            self.model, self.processor = self.load_model()

    def segment_image(self, image: Image.Image, threshold: float = 0.5, mask_threshold: float = 0.45):
        """
        Segments the image using the loaded SAM3 model.
        """
        self.initialize()
        if not self.model or not self.processor:
            return None

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Segment using text prompt "Coin"
        inputs = self.processor(images=image, text="Coin", return_tensors="pt").to(self.device)

        with torch.inference_mode(), torch.amp.autocast('cuda' if self.device == "cuda" else 'cpu'):
            outputs = self.model(**inputs)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        return results
