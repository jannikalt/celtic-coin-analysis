# Celtic Coin Analysis Project

A comprehensive system for analyzing Celtic coins using deep learning, combining automatic segmentation, classification, similarity retrieval, and side order prediction.

## Overview

This project provides end-to-end tools for working with Celtic coin images:

- **Automatic Segmentation**: Detect and segment individual coins from images using SAM3
- **Coin Matching**: Match obverse and reverse sides of the same coin using edge and color similarity
- **Type Classification**: Classify coins into different types using DINOv3-based models
- **Side Classification**: Predict whether coin pairs are in obverse-reverse or reverse-obverse order
- **Similarity Retrieval**: Find similar coins in a dataset using deep learning features
- **Dataset Management**: Tools for merging, cleaning, and rating coin datasets

## Project Structure

```
celtic-coin-analysis-now/
├── coin-analysis-app/       # Streamlit web application
├── coin-classification/     # Training pipeline for classifiers
├── data/                    # Dataset management and merging
└── README.md                # This file
```

## Components

### 1. Coin Analysis App

**Location**: `coin-analysis-app/`

A modern Streamlit web application providing an interactive interface for:
- Automatic coin segmentation with SAM3
- Coin matching and side classification
- Instance retrieval with DINOv3 embeddings
- Type classification inference
- Dataset browsing and exploration

📖 [Full Documentation](coin-analysis-app/README.md)

### 2. Coin Classification

**Location**: `coin-classification/`

A unified training pipeline for DINOv3-based coin classification with three modes:

1. **Classifier Mode**: Linear/MLP head for coin type classification
2. **Metric Learning Mode**: ArcFace + triplet loss for embedding space learning
3. **Side Classifier Mode**: Binary classification for obverse-reverse order prediction

📖 [Full Documentation](coin-classification/README.md)

### 3. Dataset Management

**Location**: `data/`

Scripts and documentation for:
- Merging OCC and Coinarchives datasets
- Cleaning and standardizing metadata
- Generating coin masks with SAM3
- Manual condition rating (0-2 scale per side)
- De-duplication and quality control

**Output**: Merged dataset with ~1000+ Celtic coin records.

📖 [Full Documentation](data/README.md)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers (Hugging Face)
- Streamlit
- OpenCV
- NumPy, Pandas, PIL

See individual component `requirements.txt` files for complete dependencies.

## Technical Details

- **Backbone**: DINOv3-ViT (facebook/dinov3-vits16-pretrain-lvd1689m)
- **Segmentation**: SAM3 (Meta AI)
- **Classification**: Linear/MLP heads on frozen DINOv3 features
- **Side Classification**: Token-mask encoder with mask-weighted pooling
- **Retrieval**: Cosine similarity on DINOv3 embeddings
- **Matching**: IoU-based edge similarity with rotation search

## Authors

This project was developed as part of a Computer Science module by:

- [Malaz Al-Mahdi](https://github.com/Malaz-al-Mahdi)
- [Jannik Alt](https://github.com/jannikalt)
- Mu Li

## License

[MIT License](LICENSE)

## Acknowledgments

- DINOv3: Meta AI
- SAM3: Meta AI
