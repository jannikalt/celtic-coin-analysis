"""
Build a dataframe for the merged dataset that includes image paths for obverse and reverse images based on IDs.
Saves the resulting dataframe as a TSV file.
"""

import os
import pandas as pd

# --- CONFIG ---
tsv_path = "merged_dataset_rated.csv"        # path to your TSV file
dir_obv = "obv"           # directory containing obverse images
dir_rev = "rev"           # directory containing reverse images
output_path = "merged_dataset_pairs.tsv"    # where to save the result
# ----------------

# Read CSV
df = pd.read_csv(tsv_path, dtype=str)

# Remove rows with empty id
df = df[df["id"].notna() & (df["id"].str.strip() != "")]

# Function to find image path for a given id inside a directory
def find_image_path(directory, image_id):
    # Check all typical image extensions
    extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"]
    for ext in extensions:
        path = os.path.abspath(os.path.join(directory, image_id + ext))
        if os.path.isfile(path):
            return path
    return None  # if nothing found

# Create new columns with image paths
df["obverse_path"] = df["id"].apply(lambda x: find_image_path(dir_obv, x))
df["reverse_path"] = df["id"].apply(lambda x: find_image_path(dir_rev, x))

# Save output
df.to_csv(output_path, sep="\t", index=False)

print("Done. Saved with image paths:", output_path)
