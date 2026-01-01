import os
import pandas as pd
import sys
import itertools
from dataclasses import dataclass

import torch
from tqdm import tqdm
from PIL import Image, ImageOps
from transformers import Sam3Processor, Sam3Model
import numpy as np
import cv2


@dataclass(frozen=True)
class ImageRecord:
    side: str
    coin_id: str
    src_path: str
    dst_path: str
    dhash: str
    width: int
    height: int


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute a simple 64-bit dHash and return it as 16-hex chars (for hash_size=8)."""
    if hash_size <= 0:
        raise ValueError("hash_size must be > 0")

    gray = image.convert("L")
    # dHash uses (hash_size + 1) x hash_size
    resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.uint8)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Pack into int
    bit_string = diff.flatten()
    value = 0
    for bit in bit_string:
        value = (value << 1) | int(bool(bit))
    # Width in hex digits
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def _hamming_distance_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("hashes must have same length")
    return int(bin(int(a, 16) ^ int(b, 16)).count("1"))


def preprocess_images_and_find_duplicates(
    base_dir: str,
    obv_img_dir: str,
    rev_img_dir: str,
    corrected_obv_dir: str,
    corrected_rev_dir: str,
    duplicates_csv_path: str,
    fingerprints_csv_path: str | None = None,
    near_duplicate_hamming_threshold: int = 0,
) -> tuple[str, str]:
    """EXIF-transpose images, save corrected copies, compute dHash fingerprints, and write duplicates CSV.

    Returns (corrected_obv_dir, corrected_rev_dir) to be used for downstream segmentation.
    """
    os.makedirs(corrected_obv_dir, exist_ok=True)
    os.makedirs(corrected_rev_dir, exist_ok=True)

    all_records: list[ImageRecord] = []

    def process_dir(side: str, src_dir: str, dst_dir: str):
        if not os.path.isdir(src_dir):
            print(f"Image directory not found: {src_dir}")
            return

        for name in tqdm(sorted(os.listdir(src_dir)), desc=f"Preprocessing {side} images"):
            src_path = os.path.join(src_dir, name)
            if not os.path.isfile(src_path) or not _is_image_file(src_path):
                continue

            coin_id = os.path.splitext(name)[0]
            dst_path = os.path.join(dst_dir, f"{coin_id}.jpg")

            try:
                with Image.open(src_path) as img:
                    img = ImageOps.exif_transpose(img)
                    img = img.convert("RGB")
                    dhash = _compute_dhash(img)
                    width, height = img.size

                    # Persist corrected image as JPEG for consistent downstream loading
                    img.save(dst_path, format="JPEG", quality=100, optimize=True)

                all_records.append(
                    ImageRecord(
                        side=side,
                        coin_id=str(coin_id),
                        src_path=src_path,
                        dst_path=dst_path,
                        dhash=dhash,
                        width=width,
                        height=height,
                    )
                )
            except Exception as e:
                print(f"Failed to preprocess image {src_path}: {e}")

    process_dir("obv", obv_img_dir, corrected_obv_dir)
    process_dir("rev", rev_img_dir, corrected_rev_dir)

    if len(all_records) == 0:
        print("No images found for preprocessing; skipping duplicate detection.")
        return corrected_obv_dir, corrected_rev_dir

    # Write fingerprints CSV if requested
    fp_df = pd.DataFrame([
        {
            "side": r.side,
            "coin_id": r.coin_id,
            "src_path": r.src_path,
            "corrected_path": r.dst_path,
            "dhash": r.dhash,
            "width": r.width,
            "height": r.height,
        }
        for r in all_records
    ])
    if fingerprints_csv_path:
        fp_df.to_csv(fingerprints_csv_path, index=False)

    # Duplicate detection
    duplicates_rows: list[dict] = []

    if near_duplicate_hamming_threshold <= 0:
        # Exact duplicates by hash
        groups: dict[str, list[ImageRecord]] = {}
        for r in all_records:
            groups.setdefault(r.dhash, []).append(r)
        for dhash, group in groups.items():
            if len(group) <= 1:
                continue
            for a, b in itertools.combinations(group, 2):
                duplicates_rows.append(
                    {
                        "dhash": dhash,
                        "hamming": 0,
                        "side_a": a.side,
                        "coin_id_a": a.coin_id,
                        "path_a": a.dst_path,
                        "side_b": b.side,
                        "coin_id_b": b.coin_id,
                        "path_b": b.dst_path,
                    }
                )
    else:
        # Near duplicates: compare all pairs (O(n^2)). Use carefully.
        for a, b in tqdm(itertools.combinations(all_records, 2), desc="Comparing fingerprints"):
            dist = _hamming_distance_hex(a.dhash, b.dhash)
            if dist <= near_duplicate_hamming_threshold:
                duplicates_rows.append(
                    {
                        "dhash": a.dhash,
                        "hamming": dist,
                        "side_a": a.side,
                        "coin_id_a": a.coin_id,
                        "path_a": a.dst_path,
                        "side_b": b.side,
                        "coin_id_b": b.coin_id,
                        "path_b": b.dst_path,
                    }
                )

    dup_df = pd.DataFrame(duplicates_rows)
    dup_df.to_csv(duplicates_csv_path, index=False)
    print(f"Wrote duplicates report: {duplicates_csv_path} ({len(dup_df)} pairs)")

    return corrected_obv_dir, corrected_rev_dir

def clean_occ_dataset(input_csv, output_csv):
    # Read CSV (semicolon separated)
    df = pd.read_csv(input_csv, sep=";", dtype=str)

    print(f"Initial dataset contains {len(df)} rows.")
    print("Cleaning dataset...")
    # ------------------------------------------------------------------
    # 1. Remove rows with blank type
    # ------------------------------------------------------------------
    df = df[df["Type | Code | Code"].notna()]
    df = df[df["Type | Code | Code"].str.strip() != ""]

    # ------------------------------------------------------------------
    # 2. Normalize specific Type | Code | Code values
    # ------------------------------------------------------------------
    label_fixes = {
        "KS 1/8/1 Horse with backward facing head": "Horse with backward facing head"
    }
    df["Type | Code | Code"] = df["Type | Code | Code"].replace(label_fixes)

    # ------------------------------------------------------------------
    # 3. Remove empty columns
    # ------------------------------------------------------------------
    df = df.drop(columns=[
        "Collection | Name",
        "Collection | Surname"
    ], errors="ignore")

    # ------------------------------------------------------------------
    # 4. Rename columns
    # ------------------------------------------------------------------
    df = df.rename(columns={
        "Id": "id",
        "Type | Code | Code": "label",
        "Type | Obverse design | Obverse design | Description": "obverse_description",
        "Type | Reverse design | Reverse design | Description": "reverse_description",
        "Collection | Name.1": "collection",
        "Inventory number": "inv_num",
        "Weight": "weight",
        "Diameter max.": "max_diameter",
        "Findspot | Name": "findspot"
    })

    # ------------------------------------------------------------------
    # 5. Converge rows with duplicate id
    # ------------------------------------------------------------------
    cleaned_rows = []

    for id_value, group in df.groupby("id", sort=False):
        group = group.reset_index(drop=True)

        # Initialize label_alt column
        group["label_alt"] = ""

        if len(group) == 1:
            cleaned_rows.append(group.iloc[0])

        else:
            first = group.iloc[0].copy()
            second = group.iloc[1].copy()

            # Special exception for "Manching 2"
            if first["label"] == "Manching 2":
                second["label_alt"] = "Manching 2"
                cleaned_rows.append(second)
            else:
                first["label_alt"] = second["label"]
                cleaned_rows.append(first)

    cleaned_df = pd.DataFrame(cleaned_rows)

    # ------------------------------------------------------------------
    # 6. Write cleaned dataset
    # ------------------------------------------------------------------
    cleaned_df.to_csv(output_csv, index=False)

    # ------------------------------------------------------------------
    # 7. Preprocess images (EXIF transpose + persist) and detect duplicates
    # ------------------------------------------------------------------
    base_dir = os.path.dirname(input_csv)
    obv_img_dir = os.path.join(base_dir, "occ_obv")
    rev_img_dir = os.path.join(base_dir, "occ_rev")
    corrected_obv_dir = os.path.join(base_dir, "occ_obv_exif")
    corrected_rev_dir = os.path.join(base_dir, "occ_rev_exif")

    duplicates_csv = os.path.join(os.path.dirname(output_csv), "occ_image_duplicates.csv")
    fingerprints_csv = os.path.join(os.path.dirname(output_csv), "occ_image_fingerprints.csv")

    corrected_obv_dir, corrected_rev_dir = preprocess_images_and_find_duplicates(
        base_dir=base_dir,
        obv_img_dir=obv_img_dir,
        rev_img_dir=rev_img_dir,
        corrected_obv_dir=corrected_obv_dir,
        corrected_rev_dir=corrected_rev_dir,
        duplicates_csv_path=duplicates_csv,
        fingerprints_csv_path=fingerprints_csv,
        near_duplicate_hamming_threshold=0,
    )

    print(f"Generating masks for {2*len(cleaned_df)} images...")
    # ------------------------------------------------------------------
    # 8. Generate segmentation masks for images
    # ------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load SAM3 model and processor
    try:
        model = Sam3Model.from_pretrained("facebook/sam3").to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        print(f"Error loading SAM3 model: {e}")
        return
    
    coin_qual_df = pd.DataFrame(columns=["id", "side", "mask_quality", "has_artifacts"])
    coin_qual_df['mask_quality'] = coin_qual_df['mask_quality'].astype(float)

    # Use corrected (EXIF-transposed) images for segmentation
    obv_img_dir = corrected_obv_dir
    rev_img_dir = corrected_rev_dir

    for idx, row in tqdm(cleaned_df.iterrows(), total=len(cleaned_df), desc="Processing images"):
        coin_id = row.get("id", None)
        obv_img_file = os.path.join(obv_img_dir, f'{coin_id}.jpg')
        rev_img_file = os.path.join(rev_img_dir, f'{coin_id}.jpg')
        if pd.isna(coin_id) or not os.path.isfile(obv_img_file) or not os.path.isfile(rev_img_file):
            print(f"Skipping row {idx} due to missing ID or image file.")
            continue

        for side, img_file in [("obv", obv_img_file), ("rev", rev_img_file)]:
            # ------------------------------------------------------------------
            # 7a. Load image and add border
            # ------------------------------------------------------------------
            # Load image
            image = Image.open(img_file).convert("RGB")
            # Add border
            border_width = int(0.35 * max(image.size))
            fill_color = 'white'
            processed_image = ImageOps.expand(image, border=border_width, fill=fill_color)

            # ------------------------------------------------------------------
            # 7b. Segment coin
            # ------------------------------------------------------------------
            success = False
            masks = None
            boxes = None
            for coin_threshold in [0.5, 0.35, 0.1, 0.05]:
                inputs = processor(images=processed_image, text="Coin", return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=coin_threshold,
                    mask_threshold=0.5,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

                if results is None:
                    print(f"No segmentation results for image ID {coin_id}#{side}.")
                    continue

                masks = results.get("masks")
                boxes = results.get("boxes")

                if (masks is None or len(masks) != 1) or (boxes is None or len(boxes) != 1):
                    continue

                if (len(masks) != len(boxes)):
                    print(f"Mismatch in number of masks and boxes for image ID {coin_id}#{side}.")
                    continue

                success = True
                break
            if not success:
                coin_qual_df.loc[len(coin_qual_df)] = {
                    "id": coin_id,
                    "side": side,
                    "mask_quality": -1.0
                }
                print(f"Failed to segment coin for image ID {coin_id}#{side}.")
                continue

            coin_qual_df.loc[len(coin_qual_df)] = {
                "id": coin_id,
                "side": side,
                "mask_quality": coin_threshold
            }

            # ------------------------------------------------------------------
            # 7c. Check mask for quality and overlap with logo
            # ------------------------------------------------------------------
            mask = masks[0]
            box = boxes[0]

            mask_cpu = mask.squeeze().cpu().numpy().astype('uint8') * 255
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
            # Copy mask and flood fill from corner
            floodfill = mask_cpu.copy()
            cv2.floodFill(floodfill, None, (0, 0), 255)

            # Invert floodfilled image
            floodfill_inv = cv2.bitwise_not(floodfill)

            # Holes are where original mask is background but floodfill finds enclosed areas
            holes = floodfill_inv & (~mask_cpu)

            has_holes = np.any(holes)

            if has_disconnected_regions or has_holes:
                coin_qual_df.at[len(coin_qual_df)-1, 'has_artifacts'] = "1"
                print(f"Mask has disconnected regions or holes for image ID {coin_id}#{side}.")

            # Crop border area from mask
            h, w = mask_cpu.shape

            bw = border_width

            mask_cpu_no_border = mask_cpu[
                bw : h - bw,
                bw : w - bw
            ]

            # Save mask
            mask_image = Image.fromarray(mask_cpu_no_border)
            mask_filename = os.path.join(os.path.dirname(input_csv), f"occ_{side}", f"{coin_id}_mask.jpg")
            os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
            mask_image.save(mask_filename)

    qual_output_csv = os.path.join(os.path.dirname(output_csv), "occ_coin_mask_quality.csv")
    coin_qual_df.to_csv(qual_output_csv, index=False)


if __name__ == "__main__":
    # python ./data/cleanup/clean_occ_csv.py ../keltisches_kleinsilber/keltisches_kleinsilber/occ_dataset.csv ./occ_dataset_clean.csv

    if len(sys.argv) != 3:
        print("Usage: python clean_occ.py <input.csv> <output.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    clean_occ_dataset(input_csv, output_csv)
