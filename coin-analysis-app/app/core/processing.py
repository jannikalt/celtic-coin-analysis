from PIL import Image, ImageOps, ImageDraw
import numpy as np
import matplotlib.cm
import torch

def add_border(image: Image.Image, border_width: int, fill_color: str = 'white') -> Image.Image:
    """
    Adds a border to the image.
    """
    return ImageOps.expand(image, border=border_width, fill=fill_color)

def overlay_masks(image: Image.Image, masks: torch.Tensor, alpha_val: float = 0.5) -> Image.Image:
    """
    Overlays masks on the image.
    Adapted from user provided code.
    """
    image = image.convert("RGBA")
    if masks is None or len(masks) == 0:
        return image
        
    masks_np = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks_np.shape[0]
    # Use a colormap that gives distinct colors
    cmap = matplotlib.cm.get_cmap("rainbow").resampled(n_masks)
    colors = [
        tuple(int(c * 255) for c in cmap(i)[:3])
        for i in range(n_masks)
    ]

    # Create a composite overlay layer
    full_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))

    for mask_layer, color in zip(masks_np, colors):
        # mask_layer is 2D array of 0 or 255
        mask_img = Image.fromarray(mask_layer)
        
        # Create a solid color image for this mask
        color_layer = Image.new("RGBA", image.size, color + (0,))
        
        # Create alpha channel: where mask is 255, alpha is int(255 * alpha_val)
        # mask_img.point(lambda v: int(v * alpha_val) if v > 0 else 0)
        # Optimization: Just use the mask as alpha, scaled
        mask_alpha = mask_img.point(lambda v: int(v * alpha_val) if v > 128 else 0)
        
        color_layer.putalpha(mask_alpha)
        full_overlay = Image.alpha_composite(full_overlay, color_layer)

    # Composite the full overlay onto the original image
    return Image.alpha_composite(image, full_overlay)

def crop_coins(image: Image.Image, boxes: torch.Tensor, masks: torch.Tensor, padding: int = 10) -> tuple[list[Image.Image], list[Image.Image]]:
    """
    Crops coins and masks from the image based on bounding boxes.
    boxes: Tensor of shape (N, 4) in xyxy format.
    """
    cropped_images = []
    cropped_masks = []
    width, height = image.size
    
    if boxes is None:
        return [], []

    for box, mask in zip(boxes, masks):
        # box is [x1, y1, x2, y2]
        x1, y1, x2, y2 = box.tolist()
        
        # Apply padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        
        crop = image.crop((x1, y1, x2, y2))
        cropped_mask = mask[y1:y2, x1:x2]
        cropped_images.append(crop)
        
        mask_cpu = (cropped_mask.squeeze().cpu().numpy().astype('uint8') * 255)
        mask_img = Image.fromarray(mask_cpu)
        cropped_masks.append(mask_img)

    return cropped_images, cropped_masks
