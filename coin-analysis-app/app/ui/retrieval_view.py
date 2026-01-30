from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image


def _resolve_label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("class") or item.get("category") or "unknown")


def _resolve_image_paths(item: dict[str, Any], views: str) -> list[tuple[str, str]]:
    paths = []
    if views == "rev":
        if item.get("reverse_path"):
            paths.append(("rev", str(item["reverse_path"])))
    elif views == "obv":
        if item.get("obverse_path"):
            paths.append(("obv", str(item["obverse_path"])))
    else:
        if item.get("obverse_path"):
            paths.append(("obv", str(item["obverse_path"])))
        if item.get("reverse_path"):
            paths.append(("rev", str(item["reverse_path"])))

    if not paths:
        for key in ("image_path", "path"):
            value = item.get(key)
            if value:
                paths.append(("img", str(value)))
                break
    return paths


def _try_load_image(path: str) -> Image.Image | None:
    try:
        if not Path(path).exists():
            return None
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def render_retrieval_results(results_by_coin: list[dict[str, Any]], top_k: int, views: str):
    st.subheader("Similar Coins")

    if not results_by_coin:
        st.info("No retrieval results available.")
        return

    for entry in results_by_coin:
        idx = entry["query_index"]
        query_image = entry.get("query_image")
        query_images = entry.get("query_images")
        matches = entry["matches"]

        st.markdown(f"**Coin {idx + 1}** — Top {top_k} similar")
        if query_images:
            cols = st.columns(min(len(query_images), 2))
            for i, img in enumerate(query_images[:2]):
                with cols[i]:
                    st.image(img, caption=f"Query {i + 1}", width="stretch")
        elif query_image is not None:
            st.image(query_image, caption=f"Query Coin {idx + 1}", width="stretch")

        if not matches:
            st.warning("No neighbors found.")
            st.divider()
            continue

        cols = st.columns(min(top_k, 4))
        for i, item in enumerate(matches):
            col = cols[i % len(cols)]
            label = _resolve_label(item)
            score = item.get("score", 0.0)
            score_dense = item.get("score_dense")
            img_paths = _resolve_image_paths(item, views)
            with col:
                if img_paths:
                    if len(img_paths) == 1:
                        _, img_path = img_paths[0]
                        img = _try_load_image(img_path)
                        if img is not None:
                            caption = f"{label} ({score:.3f})"
                            if score_dense is not None:
                                caption += f" | dense {score_dense:.3f}"
                            st.image(img, caption=caption, width="stretch")
                        else:
                            st.write(f"{label} ({score:.3f})")
                            st.caption("Image not found")
                    else:
                        text = f"{label} ({score:.3f})"
                        if score_dense is not None:
                            text += f" | dense {score_dense:.3f}"
                        st.write(text)
                        subcols = st.columns(2)
                        for sub_idx, (tag, img_path) in enumerate(img_paths[:2]):
                            img = _try_load_image(img_path)
                            with subcols[sub_idx]:
                                if img is not None:
                                    st.image(img, caption=f"{tag}", width="stretch")
                                else:
                                    st.caption(f"{tag} image not found")
                else:
                    st.write(f"{label} ({score:.3f})")

        st.divider()
