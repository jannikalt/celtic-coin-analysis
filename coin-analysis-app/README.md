# Celtic Coin Analysis

A modern web application for automatic coin segmentation using the SAM3 model.

## Features

- **Automatic Segmentation**: Uses Meta's SAM3 model to detect and segment coins.
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
3. **View results**: See segmented coins, matching pairs (if 3+ coins detected), their predicted sides, and download individual crops.

#### Segmentation Parameters

Fine-tune the SAM3 segmentation and processing behavior using these sidebar controls:

**Segmentation:**
- **Threshold**: Confidence threshold for segmentation (default: 0.5, range: 0.0-1.0)
  - Higher values = more confident detections only
- **Mask Threshold**: Binarization threshold for masks (default: 0.45, range: 0.0-1.0)
  - Adjust if masks appear too large or too small

**Image Pre-processing:**
- **Border Width**: Width of border to add around the image (default: 200px, range: 0-500)
  - Helps detect coins near image edges
- **Border Fill Color**: Color of the added border (default: white)
  - Use color picker to adjust

**Cropping:**
- **Crop Padding (px)**: Padding around detected objects when cropping (default: 10px, range: 0-100)
  - Adds margin around each cropped coin

#### Coin matching

To allow processing images which contain the obverse and reverse sides of more than one coin in an image a coin matching algorithm is employed:

1. Preprocesses masks (center & resize).
    - Masks are cropped to their bounding box, resized to 128x128, and centered to ensure fair comparison.
2. Extracts the edges.
3. Compares edges by rotating (0-360°) and flipping.
    - The `get_best_rotation_match` method rotates one mask in 10-degree steps (0-360) and calculates the Intersection over Union (IoU) with the other mask.
4. Selects the best matches based on the edge overlap (IoU) and the mean color of the coin.
    - It calculates pairwise scores for all detected coins. It then greedily selects the best matches (highest score) until coins are exhausted.

**Matching Parameters:**
- **Color Weight**: Weight of color similarity in matching score vs edge similarity (default: 0.3, range: 0.0-1.0)
  - Higher values = more emphasis on color matching
  - Lower values = more emphasis on edge/shape matching

#### Side classification

The side classification feature predicts which detected coin side is obverse vs. reverse and is performed on the matched pairs.

**Training a Side Classifier:**

First, train a side classifier using the coin-classification CLI:

```bash
python -m coin_classifier.cli \
    --mode side-classifier \
    --data "path/to/your/dataset.tsv" \
    --out-dir "classification/runs/side_classifier" \
    --epochs 10 \
    --wandb
```

The side classifier uses DINOv3 with token-mask encoding to learn which side is obverse vs. reverse. Each coin pair generates two training samples (original and flipped order).

**Using the Side Classifier in the App:**

1. Enable side classification in the sidebar by checking **"Enable Side Classification"**.
2. Specify the **Side Classifier Model Directory** path (e.g., `classification/runs/side_classifier`).
   - This directory must contain: `best_model.pt`, `label_mapping.json`, and `run_config.json`
3. Upload an image and run segmentation.
4. When 2+ coins are detected, the app will:
   - Match coins into pairs
   - Predict the side order for each pair ("obv-rev" or "rev-obv")
   - Display predictions with confidence scores

**Example Side Classifier Directory Structure:**
```
classification/runs/side_classifier/
├── best_model.pt              # Trained model weights
├── label_mapping.json         # Side order mapping
├── run_config.json            # Training configuration
└── confusion_matrix.png       # Training results
```

### Retrieval Tab

The retrieval tab allows you to find similar coins in your dataset based on visual similarity and classify coins using trained models.

#### Similarity Retrieval Setup

1. Enable retrieval in the sidebar by checking **"Enable Similarity Retrieval"**.
2. Provide the path to your dataset file (TSV or CSV format) in the **"Dataset TSV/CSV"** field.
3. Configure the retrieval parameters:
   - **Top-K Similar Coins**: Number of similar coins to retrieve (default: 5)
   - **Index Batch Size**: Batch size for building the embedding index (default: 32)
   - **Rerank with Dense Patches**: Enable fine-grained reranking using patch-level features
   - **Rerank Top-K**: Number of top results to rerank (default: 20)
   - **Patch Stride**: Stride for patch extraction during reranking (higher = faster but less precise)
   - **Dense Weight**: Weight for combining coarse and dense similarity scores (default: 0.5)

#### Similarity Retrieval Usage

1. Upload an **obverse image**, **reverse image**, or both.
2. Click **"Run Retrieval"** to search for similar coins.
3. View the top-K most similar coins from your dataset with their similarity scores.

#### Classification

The classification feature allows you to predict coin types using a trained classifier model.

**Training a Classifier:**

First, train a classifier using the coin-classification CLI:

```bash
python -m coin_classifier.cli \
    --data "path/to/your/dataset.tsv" \
    --mode classifier \
    --views both_concat \
    --head-type linear \
    --epochs 30 \
    --batch-size 16 \
    --lr 0.001 \
    --out-dir "classification/runs/my_classifier"
```

**Using the Classifier in the App:**

1. Enable classification in the sidebar by checking **"Enable Classification"**.
2. Specify the **Model Directory** path (e.g., `classification/runs/my_classifier`).
   - This directory must contain: `best_model.pt`, `label_encoder.json`, and `run_config.json`
3. Upload **obverse** and/or **reverse** images (based on what the model was trained with).
4. Click **"Run Classification"** to predict the coin type.
5. View:
   - Predicted class with confidence
   - Top-10 most probable classes with probabilities
   - Full probability distribution (expandable)

**Example Model Directory Structure:**
```
classification/runs/my_classifier/
├── best_model.pt              # Trained model weights
├── label_encoder.json         # Class label mapping
├── run_config.json            # Training configuration
└── confusion_matrix.png       # Training results
```

**Tips:**
- The model must match the input views: use `both_concat` for both obverse+reverse, `rev` for reverse only, etc.
- Higher confidence scores indicate more certain predictions
- Use `--omit-classes` during training to exclude problematic classes

#### Usage

Your dataset file (TSV or CSV) must contain the following columns:

- **`id`**: Unique identifier for each coin
- **`label`**: Class or category label for the coin
- **`obverse_path`**: Absolute or relative path to the obverse image
- **`reverse_path`**: Absolute or relative path to the reverse image

Example:
```tsv
id	label	obverse_path	reverse_path
001	Type_A	data/obv/001.jpg	data/rev/001.jpg
002	Type_B	data/obv/002.jpg	data/rev/002.jpg
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
coin-analysis-app/
├── app/
│   ├── core/            # Core logic (Segmentation, Processing)
│   ├── ui/              # UI Components
│   └── main.py          # Application Entry Point
├── assets/              # Static assets
├── requirements.txt     # Python dependencies
└── README.md            # Documentation
```
