import pandas as pd
import re
import os
import sys
import torch
from transformers import Sam3Processor, Sam3Model
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
import numpy as np
import glob
import cv2

def extract_number_from_text(text, pattern):
    """
    Extracts a float number from text using the provided regex pattern to find the substring,
    and then parsing the number within that substring.
    """
    if pd.isna(text):
        return None
    # Ensure text is string
    text = str(text)
    
    match = re.search(pattern, text)
    if match:
        found_str = match.group(0)
        # Extract the number part: digits, optional comma/dot, optional digits
        # look for the first number-like sequence in the match
        
        # Matches: 12  12,34  12.34
        number_match = re.search(r'[0-9]+([.,][0-9]+)?', found_str)
        if number_match:
            num_str = number_match.group(0)
            num_str = num_str.replace(',', '.')
            try:
                return float(num_str)
            except ValueError:
                return None
    return None

def crop_coins(image: Image.Image, boxes: torch.Tensor, masks: torch.Tensor) -> tuple[list[Image.Image], list[Image.Image]]:
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
        padding = int(0.015 * max((x2 - x1), (y2 - y1)))
        
        # Apply padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(width, int(x2) + padding)
        y2 = min(height, int(y2) + padding)
        
        crop = image.crop((x1, y1, x2, y2))
        cropped_mask = mask[y1:y2, x1:x2]
        cropped_images.append(crop)
        cropped_masks.append(cropped_mask)
        
    return cropped_images, cropped_masks

def clean_dataset(input_file, images_dir, output_file):
    print(f"Reading {input_file}...")
    # Try reading with default comma separator first
    try:
        df = pd.read_csv(input_file, dtype=str)
        # if only 1 column, it might be semicolon separated
        if df.shape[1] <= 1:
             df = pd.read_csv(input_file, sep=';', dtype=str)
    except Exception:
        # Fallback to semicolon
        df = pd.read_csv(input_file, sep=';', dtype=str)

    print(f"Original columns: {list(df.columns)}")

    # ------------------------------------------------------------------
    # 1. Rename columns
    # ------------------------------------------------------------------
    rename_map = {
        "LotID": "id",
        "Typ nach OCC  2025-10-27": "label",
        "Titel": "title",
        "Beschreibung": "description",
        "Auktionsdatum": "auction_date",
        "Detail-Link": "url",
        "Bild-Link": "img_url"
    }
    
    # Only rename columns that exist
    df = df.rename(columns=rename_map)
    
    # ------------------------------------------------------------------
    # 2. Remove entries with invalid pairs
    # ------------------------------------------------------------------
    ids_to_remove = ["2192365", "2192366", "2444062", "2444063", "2543763", "2543764", "2280339"]
    # Ensure 'id' column exists before filtering
    if 'id' in df.columns:
        initial_count = len(df)
        df = df[~df['id'].isin(ids_to_remove)]
        print(f"Removed {initial_count - len(df)} rows with invalid IDs.")
    
    # ------------------------------------------------------------------
    # 3. Extract weight and diameter
    # ------------------------------------------------------------------
    # Regexes
    weight_regex = r'(-|\s|\()+[0-9]+(,|\.)[0-9]+\s*g(\s|\.|\)|,)'
    diameter_regex = r'(=|-|\s|\()+[0-9]+(,|\.)?[0-9]?\s*mm(\s|\.|\)|,|;|m)'
    
    # Apply to description column if it exists
    if 'description' in df.columns:
        print("Extracting weight and diameter from description...")
        df['weight'] = df['description'].apply(lambda x: extract_number_from_text(x, weight_regex))
        df['max_diameter'] = df['description'].apply(lambda x: extract_number_from_text(x, diameter_regex))
    else:
        print("Warning: 'description' column not found. Skipping extraction.")
        df['weight'] = None
        df['max_diameter'] = None

    # ------------------------------------------------------------------
    # 4. Reorder columns
    # ------------------------------------------------------------------
    desired_columns = ["id", "label", "title", "description", "auction_date", "url", "img_url", "weight", "max_diameter"]
    
    # Filter to only columns that exist in df
    final_columns = [c for c in desired_columns if c in df.columns]
    
    df = df.reindex(columns=final_columns)
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)

    print(f"Splitting coin images into obverse and reverse images...")
    # ------------------------------------------------------------------
    # 5. Split coin images into obverse and reverse images
    # ------------------------------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load SAM3 model and processor
    try:
        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Error loading SAM3 model: {e}")
        return
    
    for idx, row in tqdm(df.iterrows(), "Processing images"):
        coin_id = row.get("id", None)
        img_file = glob.glob(os.path.join(images_dir, f'{coin_id}*.*'))
        if pd.isna(coin_id) or not img_file or len(img_file) != 1:
            print(f"Skipping row {idx} due to missing ID or image file or multiple image files.")
            continue
        
        
        # ------------------------------------------------------------------
        # 5a. Load image and add border
        # ------------------------------------------------------------------
        # Load image
        image = Image.open(img_file[0]).convert("RGB")
        # Add border
        border_width = int(0.25 * max(image.size))
        fill_color = 'white'
        processed_image = ImageOps.expand(image, border=border_width, fill=fill_color)

        # ------------------------------------------------------------------
        # 5b. Segment logos and replace with white
        # ------------------------------------------------------------------
        # Segment using text prompt "Logo"
        inputs = processor(images=processed_image, text="Logo", return_tensors="pt").to(device)

        with torch.inference_mode(), torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            outputs = model(**inputs)

        threshold = 0.5
        mask_threshold = 0.45
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        logo_masks = results.get("masks")
        logo_boxes = results.get("boxes")
        if not ((logo_masks is None or len(logo_masks) < 1) or (logo_boxes is None or len(logo_boxes) < 1)):
            img_np = np.array(processed_image)
            for mask in logo_masks:
                # Torch mask → NumPy binary mask
                mask = mask.squeeze().cpu().numpy() > 0   # boolean (H, W)

                # Replace masked pixels with white
                img_np[mask] = [255, 255, 255]

            # NumPy → PIL
            processed_image = Image.fromarray(img_np)
        
        # ------------------------------------------------------------------
        # 5c. Segment coins
        # ------------------------------------------------------------------
        # Segment using text prompt "Coin"
        inputs = processor(images=processed_image, text="Coin", return_tensors="pt").to(device)

        with torch.inference_mode(), torch.amp.autocast('cuda' if device == "cuda" else 'cpu'):
            outputs = model(**inputs)

        threshold = 0.5
        mask_threshold = 0.35
        # Post-process results
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        if results is None:
            print(f"No segmentation results for image ID {coin_id}.")
            continue

        masks = results.get("masks")
        boxes = results.get("boxes")

        if (masks is None or len(masks) < 2) or (boxes is None or len(boxes) < 2):
            print(f"Not enough masks/boxes for image ID {coin_id}.")
            continue
            
        if (len(masks) != len(boxes)):
            print(f"Mismatch in number of masks and boxes for image ID {coin_id}.")
            continue
        
        # ------------------------------------------------------------------
        # 5d. Check masks for quality and overlap with logo
        # ------------------------------------------------------------------
        # Check for overlap with logo mask
        skip = False
        for mask in masks:
            mask_cpu = mask.squeeze().cpu().numpy().astype('uint8') * 255
            if logo_masks is not None and len(logo_masks) > 0:
                for logo_mask in logo_masks:
                    logo_mask = logo_mask.squeeze().cpu().numpy().astype('uint8') * 255
                    if np.any((logo_mask > 0) & (mask_cpu > 0)):
                        print(f"Mask overlaps with logo for image ID {coin_id}.")
                        break

            # Ensure binary uint8
            mask_cpu = (mask_cpu > 0).astype(np.uint8) * 255
            # Ensure contiguous memory (important!)
            mask_cpu = np.ascontiguousarray(mask_cpu)

             # Check mask quality

            # Check for disconnected regions
            # mask: uint8, values 0 or 255
            num_labels, labels = cv2.connectedComponents(mask_cpu)
            # label 0 = background
            num_regions = num_labels - 1
            has_disconnected_regions = num_regions > 1

            # Check for holes
            h, w = mask_cpu.shape

            # Copy mask and flood fill from corner
            floodfill = mask_cpu.copy()
            cv2.floodFill(floodfill, None, (0, 0), 255)

            # Invert floodfilled image
            floodfill_inv = cv2.bitwise_not(floodfill)

            # Holes are where original mask is background but floodfill finds enclosed areas
            holes = floodfill_inv & (~mask_cpu)

            has_holes = np.any(holes)

            if has_disconnected_regions or has_holes:
                print(f"Mask has disconnected regions or holes for image ID {coin_id}.")
        if skip:
            continue

        # ------------------------------------------------------------------
        # 5e. Save cropped coins and masks
        # ------------------------------------------------------------------
        cropped_coins, cropped_masks = crop_coins(processed_image, boxes, masks)

        # Store masks and cropped images in individual directory
        coin_dir = os.path.join(images_dir, str(coin_id))
        os.makedirs(coin_dir, exist_ok=True)
        for i, crop in enumerate(cropped_coins):
            crop_file = os.path.join(coin_dir, f"{coin_id}_coin_{i+1}.png")
            crop.save(crop_file)
        for i, mask in enumerate(cropped_masks):
            mask_cpu = (mask.squeeze().cpu().numpy().astype('uint8') * 255)
            mask_img = Image.fromarray(mask_cpu)
            mask_file = os.path.join(coin_dir, f"{coin_id}_mask_{i+1}.png")
            mask_img.save(mask_file)

        if (len(masks) > 2):
            print(f"More than 2 coins detected for image ID {coin_id}, skipping obverse/reverse split.")

    print("Done.")

if __name__ == "__main__":
    # python data/cleanup/coinarchives/clean_coinarchives_csv.py ./data/cleanup/coinarchives/coinarchives_dataset_mit_OCC_Typen.csv ./data/cleanup/coinarchives/coinarchives ./data/cleanup/coinarchives/coinarchives_dataset_clean.csv

    if len(sys.argv) != 4:
        print("Usage: python clean_occ.py <input.csv> <images_dir> <output.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    images_dir = sys.argv[2]
    output_csv = sys.argv[3]

    if os.path.exists(input_csv) and os.path.isdir(images_dir):
        clean_dataset(input_csv, images_dir, output_csv)
    else:
        print(f"Error: Input file {input_csv} not found.")
