# DINOv3 Coin Classification

Unified, modular pipeline for classifying coin images with DINOv3 features. Supports three modes:
1) linear classifier on DINOv3 features,
2) metric learning with ArcFace + triplet loss, and
3) side classification (obverse-reverse order prediction).

## Features
- Dataset loading (TSV/CSV with id,label,obverse_path,reverse_path).
- DINOv3 backbone with configurable views (obv/rev/both), frozen during training.
- Three training modes:
  - `classifier`: linear head on top of DINOv3 features.
  - `metric`: ArcFace + triplet loss metric learning for embedding space separation.
  - `side-classifier`: binary classification for predicting coin side order (obv-rev vs rev-obv).
- Evaluation: per-class metrics, confusion matrix, optional kNN or HDBSCAN eval on learned embeddings.

## Dataset format
TSV or CSV with these required columns:

| column | description |
|---|---|
| `id` | unique sample id |
| `label` | class label (string) |
| `obverse_path` | path to obverse image |
| `reverse_path` | path to reverse image |
| `obv_condition` | optional condition rating 0–2 for obverse |
| `rev_condition` | optional condition rating 0–2 for reverse |

Example (TSV):

```
id	label	obverse_path	reverse_path
1	TypeA	D:/data/obv/1.jpg	D:/data/rev/1.jpg
2	TypeB	D:/data/obv/2.jpg	D:/data/rev/2.jpg
```

## Project layout

```
classification/
  coin_classifier/
    cli.py
    data/
    models/
    losses/
    training/
    eval/
    retrieval/
    utils/
  requirements.txt
  README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run

Classifier mode:

```bash
python -m coin_classifier.cli --data path/to/data.tsv --out-dir runs/exp1 --mode classifier
```

Metric learning mode:

```bash
python -m coin_classifier.cli --data path/to/data.tsv --out-dir runs/exp_metric --mode metric --eval-method knn
```

Side classification mode:

```bash
python -m coin_classifier.cli --mode side-classifier --data path/to/data.tsv --out-dir runs/side_classifier --epochs 10 --wandb
```

The side classifier predicts whether a coin pair is in "obv-rev" or "rev-obv" order using token-mask encoding with DINOv3. Each coin pair generates two training samples (original and flipped order). The same dataset format is used as regular classification.

Single-image prediction (classifier head):
 (for classifier/metric modes)
- `--mode`: `classifier`, `metric`, or `side-classifier`
- `--embedding-dim`: output dimension for metric embeddings
- `--eval-method`: `none`, `knn`, `hdbscan` (for metric mode)
- `--min-condition`: only include samples with average condition above this value
- `--head-type`: `linear` or `mlp` (classifier head)
- `--proj-type`: `linear` or `mlp` (metric head)
- `--disable-mask-pooling`: disable mask-weighted pooling (side-classifier mode
## Key CLI options
- `--views`: `rev`, `obv`, `both_avg`, `both_concat`
- `--mode`: `classifier` or `metric`
- `--embedding-dim`: output dimension for metric embeddings
- `--eval-method`: `none`, `knn`, `hdbscan`
- `--min-condition`: only include samples with average condition above this value
- `--head-type`: `linear` or `mlp` (classifier head)
- `--proj-type`: `linear` or `mlp` (metric head)

## Outputs
- `run_config.json`: full CLI config (classifier mode)
- `label_mapping.json`: side order mapping (side-classifier mode)
- `best_model.pt`: best model checkpoint
- `confusion_matrix.png`: confusion matrix (classifier/side-classifier modes)
- `metrics_epoch_*.json`: per-epoch metrics (classifier/side-classifier modes
- `metrics_epoch_*.json`: per-epoch metrics (classifier mode)
- `embeddings_val.npy`, `labels_val.npy`: embeddings and labels (metric mode)
- `knn_metrics.json`: kNN metrics (metric mode)
- `hdbscan_metrics.json`: HDBSCAN metrics (metric mode)
- `hdbscan_labels.npy`: HDBSCAN cluster labels (metric mode)
