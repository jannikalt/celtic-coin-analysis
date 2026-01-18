# DINOv3 Coin Classification

Unified, modular pipeline for classifying coin images with DINOv3 features. Supports two approaches in one codebase:
1) linear classifier on DINOv3 features, and
2) metric learning with ArcFace + triplet loss.

## Features
- Dataset loading (TSV/CSV with id,label,obverse_path,reverse_path).
- DINOv3 backbone with configurable views (obv/rev/both), frozen during training.
- Two training modes:
  - `classifier`: linear head on top of DINOv3 features.
  - `metric`: ArcFace + triplet loss metric learning for embedding space separation.
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

Single-image prediction (classifier head):

```bash
python -m coin_classifier.cli --model-name facebook/dinov3-base --views rev \
  --predict-rev path/to/rev.jpg --checkpoint runs/exp1/best_model.pt \
  --label-encoder runs/exp1/label_encoder.json
```

## Key CLI options
- `--views`: `rev`, `obv`, `both_avg`, `both_concat`
- `--mode`: `classifier` or `metric`
- `--embedding-dim`: output dimension for metric embeddings
- `--eval-method`: `none`, `knn`, `hdbscan`
- `--min-condition`: only include samples with average condition above this value
- `--head-type`: `linear` or `mlp` (classifier head)
- `--proj-type`: `linear` or `mlp` (metric head)

## Outputs
- `run_config.json`: full CLI config
- `label_encoder.json`: class mapping
- `best_model.pt`: best classifier model (classifier mode)
- `confusion_matrix.png`: confusion matrix (classifier mode)
- `metrics_epoch_*.json`: per-epoch metrics (classifier mode)
- `embeddings_val.npy`, `labels_val.npy`: embeddings and labels (metric mode)
- `knn_metrics.json`: kNN metrics (metric mode)
- `hdbscan_metrics.json`: HDBSCAN metrics (metric mode)
- `hdbscan_labels.npy`: HDBSCAN cluster labels (metric mode)

## Optional dependencies
HDBSCAN evaluation requires installing:

```bash
pip install hdbscan
```
