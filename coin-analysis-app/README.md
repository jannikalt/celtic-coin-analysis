# Celtic Coin Analysis

A modern web application for automatic coin segmentation using the SAM3 model.

## Features

- **Automatic Segmentation**: Uses Facebook's SAM3 model to detect and segment coins.
- **Configurable Parameters**: Adjust segmentation thresholds, mask thresholds, and border settings.
- **Interactive UI**: Upload images or provide URLs, view results instantly.
- **Gallery & Download**: View individual cropped coins and download them.
- **Instance Retrieval**: Find similar coins in your dataset using DINOv3-based similarity search with optional dense patch reranking.
- **Dataset Viewer**: Browse and explore your coin dataset with interactive label and coin browsing.

## Installation

1.  **Clone the repository** (if applicable) or navigate to the project folder.

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install PyTorch with CUDA support separately if you have a compatible GPU. Visit [pytorch.org](https://pytorch.org/) for instructions.*

## Usage

Run the application using Streamlit:

```bash
streamlit run app/main.py
```

The application will open in your default web browser.

### Segmenter Tab

1. **Upload or provide URL**: Choose an image containing coins.
2. **Adjust parameters**: Use the sidebar to fine-tune segmentation and matching settings.
3. **View results**: See segmented coins, matching pairs (if 3+ coins detected), and download individual crops.

### Retrieval Tab

The retrieval tab allows you to find similar coins in your dataset based on visual similarity.

#### Setup

1. Enable retrieval in the sidebar by checking **"Enable Similarity Retrieval"**.
2. Provide the path to your dataset file (TSV or CSV format) in the **"Dataset TSV/CSV"** field.
3. Configure the retrieval parameters:
   - **Top-K Similar Coins**: Number of similar coins to retrieve (default: 5)
   - **Index Batch Size**: Batch size for building the embedding index (default: 32)
   - **Rerank with Dense Patches**: Enable fine-grained reranking using patch-level features
   - **Rerank Top-K**: Number of top results to rerank (default: 20)
   - **Patch Stride**: Stride for patch extraction during reranking (higher = faster but less precise)
   - **Dense Weight**: Weight for combining coarse and dense similarity scores (default: 0.5)

#### Usage

1. Upload an **obverse image**, **reverse image**, or both.
2. Click **"Run Retrieval"** to search for similar coins.
3. View the top-K most similar coins from your dataset with their similarity scores.

#### Dataset Format

Your dataset file (TSV or CSV) must contain the following columns:

- **`id`**: Unique identifier for each coin
- **`label`**: Class or category label for the coin
- **`obverse_path`**: Absolute or relative path to the obverse image
- **`reverse_path`**: Absolute or relative path to the reverse image

Example:
```tsv
id	label	obverse_path	reverse_path
coin_001	Type_A	data/coins/001_obv.jpg	data/coins/001_rev.jpg
coin_002	Type_B	data/coins/002_obv.jpg	data/coins/002_rev.jpg
```

**Note**: The retrieval system builds embeddings on-demand using DINOv3. The first run may take some time depending on your dataset size.

### Dataset Viewer Tab

The dataset viewer provides an interactive interface to browse and explore your coin dataset.

#### Setup

1. Specify the dataset path in the sidebar (same path used for retrieval).
2. Navigate to the **"Dataset Viewer"** tab.

#### Features

**Label Overview:**
- View all labels in the dataset with the number of coins per label
- Click on any label to view all coins with that label

**Coin Browser:**
- Browse all coins for a selected label
- View obverse and reverse images side-by-side
- Click on any coin to view detailed metadata

**Coin Details:**
- View high-resolution obverse and reverse images
- See all metadata fields from the dataset (ID, label, paths, and any additional columns)
- Navigate back to the coin list or label overview

This tab is particularly useful for:
- Exploring the dataset composition and distribution
- Verifying image paths and data quality
- Understanding label categories and their contents
- Inspecting individual coin metadata

## Project Structure

```
coin-segmenter/
├── app/
│   ├── core/            # Core logic (Segmentation, Processing)
│   ├── ui/              # UI Components
│   └── main.py          # Application Entry Point
├── assets/              # Static assets
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```

## Coin matching

1. Preprocesses masks (center & resize).
    - Masks are cropped to their bounding box, resized to 128x128, and centered to ensure fair comparison.
2. Extracts the edges.
3. Compares edges by rotating (0-360°) and flipping.
    - The `get_best_rotation_match` method rotates one mask in 10-degree steps (0-360) and calculates the Intersection over Union (IoU) with the other mask.
4. Selects the best matches based on the edge overlap (IoU).
    - It calculates pairwise scores for all detected coins. It then greedily selects the best matches (highest IoU) until coins are exhausted.
