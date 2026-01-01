# Celtic Coin Segmenter

A modern web application for automatic coin segmentation using the SAM3 model.

## Features

- **Automatic Segmentation**: Uses Facebook's SAM3 model to detect and segment coins.
- **Configurable Parameters**: Adjust segmentation thresholds, mask thresholds, and border settings.
- **Interactive UI**: Upload images or provide URLs, view results instantly.
- **Gallery & Download**: View individual cropped coins and download them.

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
