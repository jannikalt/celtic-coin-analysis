"""CLI entrypoint for the unified DINOv3 coin classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Optional
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from transformers import AutoImageProcessor

from .data.dataset import load_dataframe, CoinImagesDataset, DinoV3Collator
from .data.splits import stratified_split
from .models.classifier import DinoV3Classifier
from .models.metric import MetricEmbeddingModel
from .training.trainer import train_epoch_classifier, eval_classifier, resolve_device
from .training.metric_trainer import MetricConfig, train_metric_epoch, eval_metric_embeddings
from .losses.arcface import ArcFace
from .eval.metrics import classification_metrics
from .eval.plots import save_confusion_matrix
from .eval.knn_eval import knn_evaluate
from .eval.hdbscan_eval import hdbscan_evaluate
from .utils.seed import set_seed
from .utils.io import ensure_dir, save_json

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Unified DINOv3 coin classification pipeline")

    parser.add_argument("--data", required=True, help="Path to TSV/CSV with id,label,obverse_path,reverse_path")
    parser.add_argument("--out-dir", default="runs/exp", help="Output directory")

    parser.add_argument("--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m", help="DINOv3 model name")
    parser.add_argument("--views", default="both_concat", choices=["rev", "obv", "both_avg", "both_concat"])
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-condition", type=float, default=None, help="Minimum average condition to include")

    parser.add_argument("--mode", default="classifier", choices=["classifier", "metric"])

    parser.add_argument("--head-type", default="linear", choices=["linear", "mlp"], help="Classifier head type")
    parser.add_argument("--head-hidden", type=int, default=512)
    parser.add_argument("--head-dropout", type=float, default=0.1)

    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--proj-type", default="linear", choices=["linear", "mlp"], help="Metric projection head type")
    parser.add_argument("--proj-hidden", type=int, default=512)
    parser.add_argument("--proj-dropout", type=float, default=0.1)
    parser.add_argument("--arcface-s", type=float, default=30.0)
    parser.add_argument("--arcface-m", type=float, default=0.5)
    parser.add_argument("--triplet-margin", type=float, default=0.3)
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--arcface-weight", type=float, default=1.0)

    parser.add_argument("--eval-method", default="none", choices=["none", "knn", "hdbscan"], help="Embedding eval method")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5)
    parser.add_argument("--hdbscan-min-samples", type=int, default=5)

    parser.add_argument("--predict-obv", type=str, default=None, help="Obverse image path for prediction")
    parser.add_argument("--predict-rev", type=str, default=None, help="Reverse image path for prediction")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained model checkpoint")
    parser.add_argument("--label-encoder", type=str, default=None, help="Path to label_encoder.json")
    parser.add_argument("--predict-output", type=str, default=None, help="Optional path to save JSON prediction")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="coin-classification", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="W&B run name")

    return parser


def _filter_by_condition(df, min_condition: Optional[float]):
    """Filter samples by average obverse/reverse condition threshold."""
    if min_condition is None:
        return df
    if "obv_condition" not in df.columns or "rev_condition" not in df.columns:
        raise ValueError("Missing obv_condition or rev_condition columns for condition filtering")

    obv = df["obv_condition"].astype(float)
    rev = df["rev_condition"].astype(float)
    avg = (obv + rev) / 2.0
    return df.loc[avg >= float(min_condition)].reset_index(drop=True)


def _load_label_encoder(path: str) -> LabelEncoder:
    """Load label encoder classes from JSON and return a fitted encoder."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    classes = data.get("classes", [])
    if not classes:
        raise ValueError("label_encoder.json missing 'classes' list")
    le = LabelEncoder()
    le.fit(classes)
    return le


def _predict_single(args, processor, device: str):
    """Run single-sample prediction with a trained classifier head."""
    if not args.checkpoint or not args.label_encoder:
        raise ValueError("--checkpoint and --label-encoder are required for prediction")

    le = _load_label_encoder(args.label_encoder)
    model = DinoV3Classifier(
        args.model_name,
        num_classes=len(le.classes_),
        views=args.views,
        head_type=args.head_type,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    sample = {"label": 0, "id": "predict"}
    if args.views in {"obv", "both_avg", "both_concat"}:
        if not args.predict_obv:
            raise ValueError("--predict-obv is required for views including obverse")
        img_obv = Image.open(args.predict_obv).convert("RGB")
        img_obv = img_obv.resize((args.input_size, args.input_size), resample=Image.BILINEAR)
        sample["obv_image"] = img_obv
    if args.views in {"rev", "both_avg", "both_concat"}:
        if not args.predict_rev:
            raise ValueError("--predict-rev is required for views including reverse")
        img_rev = Image.open(args.predict_rev).convert("RGB")
        img_rev = img_rev.resize((args.input_size, args.input_size), resample=Image.BILINEAR)
        sample["rev_image"] = img_rev

    collator = DinoV3Collator(processor=processor, views=args.views)
    batch = collator([sample])
    pixel_values_rev = batch.get("pixel_values_rev")
    pixel_values_obv = batch.get("pixel_values_obv")

    if pixel_values_rev is not None:
        pixel_values_rev = pixel_values_rev.to(device)
    if pixel_values_obv is not None:
        pixel_values_obv = pixel_values_obv.to(device)

    with torch.no_grad():
        logits = model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = str(le.classes_[pred_idx])

    result = {
        "pred_label": pred_label,
        "pred_index": pred_idx,
        "probabilities": {str(le.classes_[i]): float(probs[i]) for i in range(len(probs))},
    }

    print(json.dumps(result, indent=2))
    if args.predict_output:
        Path(args.predict_output).write_text(json.dumps(result, indent=2), encoding="utf-8")


def _init_wandb(args, config: dict):
    """Initialize W&B if enabled and available."""
    if not args.wandb:
        return None
    if not WANDB_AVAILABLE:
        raise RuntimeError("W&B is not available. Install with: pip install wandb")
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=config,
    )


def main():
    """Run training and evaluation for the selected mode."""
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = ensure_dir(Path(args.out_dir))

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    try:
        processor.size = {"height": args.input_size, "width": args.input_size}
    except Exception:
        pass

    if args.predict_obv or args.predict_rev or args.checkpoint:
        _predict_single(args, processor, device)
        return

    df = load_dataframe(args.data)
    df = _filter_by_condition(df, args.min_condition)
    if args.mode == "metric":
        # Remove labels with fewer than 2 samples for metric learning / knn eval
        counts = df["label"].value_counts()
        keep_labels = counts[counts >= 2].index
        df = df[df["label"].isin(keep_labels)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No samples remaining after condition filtering")

    df_train, df_val = stratified_split(df, val_ratio=args.val_ratio, seed=args.seed)

    label_encoder = LabelEncoder()
    label_encoder.fit(df["label"].values)

    train_ds = CoinImagesDataset(df_train, label_encoder=label_encoder, input_size=args.input_size, views=args.views)
    val_ds = CoinImagesDataset(df_val, label_encoder=label_encoder, input_size=args.input_size, views=args.views)

    collator = DinoV3Collator(processor=processor, views=args.views)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    run = _init_wandb(args, vars(args))

    if args.mode == "classifier":
        model = DinoV3Classifier(
            args.model_name,
            num_classes=len(label_encoder.classes_),
            views=args.views,
            head_type=args.head_type,
            head_hidden=args.head_hidden,
            head_dropout=args.head_dropout,
        )
        model.to(device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1) # label smoothing to avoid overfitting when using small unbalanced datasets

        best_val_acc = -1.0
        for epoch in range(1, args.epochs + 1):
            train_out = train_epoch_classifier(model, train_loader, optimizer, criterion, device)
            val_out = eval_classifier(model, val_loader, criterion, device)

            metrics = classification_metrics(val_out["y_true"], val_out["y_pred"], [str(x) for x in label_encoder.classes_])
            metrics["train_loss"] = train_out["loss"]
            metrics["val_loss"] = val_out["loss"]
            metrics["epoch"] = epoch

            save_json(metrics, out_dir / f"metrics_epoch_{epoch}.json")

            if run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": metrics["train_loss"],
                        "val_loss": metrics["val_loss"],
                        "val_accuracy": metrics["accuracy"],
                        "val_f1_weighted": metrics["f1_weighted"],
                    }
                )

            if metrics["accuracy"] > best_val_acc:
                best_val_acc = metrics["accuracy"]
                torch.save(model.state_dict(), out_dir / "best_model.pt")

        save_json({"classes": label_encoder.classes_.tolist()}, out_dir / "label_encoder.json")
        save_confusion_matrix(np.array(metrics["confusion_matrix"]), [str(x) for x in label_encoder.classes_], out_dir / "confusion_matrix.png")

    else:
        model = MetricEmbeddingModel(
            model_name=args.model_name,
            embedding_dim=args.embedding_dim,
            views=args.views,
            proj_type=args.proj_type,
            proj_hidden=args.proj_hidden,
            proj_dropout=args.proj_dropout,
        )
        model.to(device)

        arcface = ArcFace(args.embedding_dim, len(label_encoder.classes_), s=args.arcface_s, m=args.arcface_m).to(device)

        optimizer = torch.optim.AdamW(
            list(filter(lambda p: p.requires_grad, model.parameters())) + list(arcface.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        cfg = MetricConfig(
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            arcface_s=args.arcface_s,
            arcface_m=args.arcface_m,
            triplet_margin=args.triplet_margin,
            triplet_weight=args.triplet_weight,
            arcface_weight=args.arcface_weight,
        )

        for epoch in range(1, args.epochs + 1):
            train_out = train_metric_epoch(model, train_loader, optimizer, arcface=arcface, cfg=cfg)
            if run is not None:
                wandb.log({"epoch": epoch, "train_loss": train_out["loss"]})

        eval_out = eval_metric_embeddings(model, val_loader, device)
        np.save(out_dir / "embeddings_val.npy", eval_out["embeddings"])
        np.save(out_dir / "labels_val.npy", eval_out["labels"])

        if args.eval_method == "knn":
            knn_metrics = knn_evaluate(eval_out["embeddings"], eval_out["labels"], n_neighbors=args.knn_k)
            save_json(knn_metrics, out_dir / "knn_metrics.json")
            if run is not None:
                wandb.log({"knn_accuracy": knn_metrics["accuracy"], "knn_f1_weighted": knn_metrics["f1_weighted"]})
        elif args.eval_method == "hdbscan":
            hdbscan_out = hdbscan_evaluate(
                eval_out["embeddings"],
                min_cluster_size=args.hdbscan_min_cluster_size,
                min_samples=args.hdbscan_min_samples,
            )
            save_json(hdbscan_out["metrics"], out_dir / "hdbscan_metrics.json")
            np.save(out_dir / "hdbscan_labels.npy", hdbscan_out["labels"])
            if run is not None:
                wandb.log(hdbscan_out["metrics"])

        save_json({"classes": label_encoder.classes_.tolist()}, out_dir / "label_encoder.json")

    config = vars(args)
    save_json(config, out_dir / "run_config.json")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
