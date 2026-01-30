from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import streamlit as st

from coin_classifier.models.backbone import DinoV3Backbone
from coin_classifier.data.dataset import load_dataframe


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return embeddings / norms


class CoinRetriever:
    def __init__(
        self,
        model: DinoV3Backbone,
        patch_model: AutoModel | None,
        processor: AutoImageProcessor,
        views: str,
        device: str,
        embeddings: np.ndarray | dict[str, np.ndarray],
        metadata: list[dict[str, Any]],
        embedding_source: str,
    ):
        self.model = model
        self.patch_model = patch_model
        self.processor = processor
        self.views = views
        self.device = device
        self.embeddings = embeddings
        self.metadata = metadata
        self.embedding_source = embedding_source

    def _extract_patch_features(self, image: Image.Image) -> torch.Tensor:
        if self.patch_model is None:
            raise RuntimeError("Patch model not initialized")
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.patch_model(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state[:, 1:, :]
        tokens = tokens.squeeze(0)
        tokens = torch.nn.functional.normalize(tokens, dim=1)
        return tokens

    def _patch_match_score(self, query: Image.Image, candidate: Image.Image, stride: int = 1) -> float:
        q_tokens = self._extract_patch_features(query)
        c_tokens = self._extract_patch_features(candidate)
        if stride > 1:
            q_tokens = q_tokens[::stride]
        sim = q_tokens @ c_tokens.T
        max_sim = sim.max(dim=1).values
        return float(max_sim.mean().item())

    def _resolve_candidate_images(self, item: dict[str, Any]) -> list[Image.Image]:
        paths: list[str] = []
        if self.views == "rev":
            if item.get("reverse_path"):
                paths.append(str(item["reverse_path"]))
        elif self.views == "obv":
            if item.get("obverse_path"):
                paths.append(str(item["obverse_path"]))
        else:
            if item.get("obverse_path"):
                paths.append(str(item["obverse_path"]))
            if item.get("reverse_path"):
                paths.append(str(item["reverse_path"]))

        images = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception:
                continue
        return images

    def _prepare_inputs(self, image: Image.Image) -> dict[str, torch.Tensor]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs: dict[str, torch.Tensor] = {}
        if self.views in {"rev", "both_concat", "both_avg"}:
            rev_inputs = self.processor(images=[image], return_tensors="pt")
            inputs["pixel_values_rev"] = rev_inputs["pixel_values"]
        if self.views in {"obv", "both_concat", "both_avg"}:
            obv_inputs = self.processor(images=[image], return_tensors="pt")
            inputs["pixel_values_obv"] = obv_inputs["pixel_values"]
        return inputs

    def _encode_single_dinov3(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        self.model.eval()
        with torch.no_grad():
            emb = self.model(pixel_values)
        emb = emb.detach().cpu().numpy()[0]
        emb = emb / max(np.linalg.norm(emb), 1e-12)
        return emb

    def _encode_pair_dinov3(self, obv_image: Image.Image, rev_image: Image.Image) -> np.ndarray:
        emb_obv = self._encode_single_dinov3(obv_image)
        emb_rev = self._encode_single_dinov3(rev_image)
        emb = np.concatenate([emb_obv, emb_rev], axis=0)
        emb = emb / max(np.linalg.norm(emb), 1e-12)
        return emb

    def encode_image(self, image: Image.Image) -> np.ndarray:
        inputs = self._prepare_inputs(image)
        if "pixel_values_rev" in inputs:
            inputs["pixel_values_rev"] = inputs["pixel_values_rev"].to(self.device)
        if "pixel_values_obv" in inputs:
            inputs["pixel_values_obv"] = inputs["pixel_values_obv"].to(self.device)

        self.model.eval()
        with torch.no_grad():
            if self.embedding_source == "metric":
                emb = self.model(
                    pixel_values_rev=inputs.get("pixel_values_rev"),
                    pixel_values_obv=inputs.get("pixel_values_obv"),
                )
            else:
                if self.views == "rev":
                    emb = self.model(inputs.get("pixel_values_rev"))
                elif self.views == "obv":
                    emb = self.model(inputs.get("pixel_values_obv"))
                elif self.views == "both_avg":
                    emb_rev = self.model(inputs.get("pixel_values_rev"))
                    emb_obv = self.model(inputs.get("pixel_values_obv"))
                    emb = 0.5 * (emb_rev + emb_obv)
                elif self.views == "both_concat":
                    emb_rev = self.model(inputs.get("pixel_values_rev"))
                    emb_obv = self.model(inputs.get("pixel_values_obv"))
                    emb = torch.cat([emb_obv, emb_rev], dim=1)
                else:
                    raise ValueError(f"Unknown views mode: {self.views}")
        emb = emb.detach().cpu().numpy()[0]
        emb = emb / max(np.linalg.norm(emb), 1e-12)
        return emb

    def search_images(
        self,
        obv_image: Image.Image | None,
        rev_image: Image.Image | None,
        top_k: int = 5,
        rerank_top_k: int | None = None,
        patch_stride: int = 1,
        dense_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        if self.embedding_source != "dinov3":
            raise ValueError("Pair retrieval is only supported for DINOv3 embeddings")

        if isinstance(self.embeddings, np.ndarray):
            raise ValueError("Expected multi-embedding index for DINOv3 retrieval")

        if obv_image is not None and rev_image is not None:
            query = self._encode_pair_dinov3(obv_image, rev_image)
            emb_matrix = self.embeddings.get("both_concat")
            if emb_matrix is None:
                raise ValueError("Missing both_concat embeddings in index")
        elif obv_image is not None:
            query = self._encode_single_dinov3(obv_image)
            emb_matrix = self.embeddings.get("obv")
            if emb_matrix is None:
                raise ValueError("Missing obv embeddings in index")
        elif rev_image is not None:
            query = self._encode_single_dinov3(rev_image)
            emb_matrix = self.embeddings.get("rev")
            if emb_matrix is None:
                raise ValueError("Missing rev embeddings in index")
        else:
            raise ValueError("Provide at least one image for retrieval")

        scores = emb_matrix @ query
        top_k = min(int(top_k), len(scores))
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            item = dict(self.metadata[int(i)])
            item["score"] = float(scores[int(i)])
            results.append(item)

        if rerank_top_k and rerank_top_k > 0 and self.patch_model is not None:
            rerank_k = min(int(rerank_top_k), len(results))
            rerank_candidates = results[:rerank_k]
            query_for_dense = obv_image or rev_image
            if query_for_dense is not None:
                for item in rerank_candidates:
                    candidate_images = self._resolve_candidate_images(item)
                    if not candidate_images:
                        item["score_dense"] = None
                        continue
                    scores_dense = [
                        self._patch_match_score(query_for_dense, c_img, stride=patch_stride)
                        for c_img in candidate_images
                    ]
                    item["score_dense"] = float(max(scores_dense))
                for item in rerank_candidates:
                    if item.get("score_dense") is not None:
                        item["score"] = (1.0 - dense_weight) * item["score"] + dense_weight * item["score_dense"]
                results[:rerank_k] = sorted(rerank_candidates, key=lambda x: x["score"], reverse=True)
        return results

    def search_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        rerank_top_k: int | None = None,
        patch_stride: int = 1,
        dense_weight: float = 0.5,
    ) -> list[dict[str, Any]]:
        query = self.encode_image(image)
        scores = self.embeddings @ query
        top_k = min(int(top_k), len(scores))
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            item = dict(self.metadata[int(i)])
            item["score"] = float(scores[int(i)])
            results.append(item)

        if rerank_top_k and rerank_top_k > 0 and self.patch_model is not None:
            rerank_k = min(int(rerank_top_k), len(results))
            rerank_candidates = results[:rerank_k]
            for item in rerank_candidates:
                candidate_images = self._resolve_candidate_images(item)
                if not candidate_images:
                    item["score_dense"] = None
                    continue
                scores_dense = [self._patch_match_score(image, c_img, stride=patch_stride) for c_img in candidate_images]
                item["score_dense"] = float(max(scores_dense))
            for item in rerank_candidates:
                if item.get("score_dense") is not None:
                    item["score"] = (1.0 - dense_weight) * item["score"] + dense_weight * item["score_dense"]
            results[:rerank_k] = sorted(rerank_candidates, key=lambda x: x["score"], reverse=True)
        return results


@st.cache_resource(show_spinner=False)
def build_dinov3_retriever_from_dataset(
    data_path: str,
    model_name: str,
    input_size: int = 224,
    batch_size: int = 32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        processor.size = {"height": input_size, "width": input_size}
    except Exception:
        pass

    df = load_dataframe(data_path).reset_index(drop=True)

    model = DinoV3Backbone(model_name).to(device)
    model.eval()
    patch_model = AutoModel.from_pretrained(model_name).to(device)
    patch_model.eval()

    emb_map: dict[str, list[np.ndarray]] = {"obv": [], "rev": [], "both_concat": []}
    metadata: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            chunk = df.iloc[start : start + batch_size]
            obv_images = [Image.open(p).convert("RGB") for p in chunk["obverse_path"]]
            rev_images = [Image.open(p).convert("RGB") for p in chunk["reverse_path"]]

            obv_inputs = processor(images=obv_images, return_tensors="pt")
            rev_inputs = processor(images=rev_images, return_tensors="pt")
            pixel_values_obv = obv_inputs["pixel_values"].to(device)
            pixel_values_rev = rev_inputs["pixel_values"].to(device)

            emb_obv = model(pixel_values_obv).detach().cpu().numpy()
            emb_rev = model(pixel_values_rev).detach().cpu().numpy()
            emb_map["obv"].append(emb_obv)
            emb_map["rev"].append(emb_rev)
            emb_concat = np.concatenate([emb_obv, emb_rev], axis=1)
            emb_map["both_concat"].append(emb_concat)

    for key, chunks in emb_map.items():
        if chunks:
            emb_map[key] = _normalize_embeddings(np.concatenate(chunks, axis=0))
        else:
            emb_map[key] = np.empty((0, 0), dtype=np.float32)

    for _, row in df.iterrows():
        metadata.append(
            {
                "id": row["id"],
                "label": row["label"],
                "obverse_path": row["obverse_path"],
                "reverse_path": row["reverse_path"],
            }
        )

    return CoinRetriever(model, patch_model, processor, "both_concat", device, emb_map, metadata, "dinov3")