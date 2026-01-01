"""merge_datasets.py

Merge OCC and CoinArchives datasets (CSV) and also merge their images/masks.

Requirements implemented
- OCC schema is expected to contain the columns:
    id,label,obverse_description,reverse_description,collection,inv_num,weight,max_diameter,findspot,label_alt
- Only columns with the exact same name are shared across datasets; the others are left empty
    for rows from the other dataset.
- Add a column "dataset" with values "occ" or "coinarchives".
- Copy images and masks for each dataset into output_path/obv and output_path/rev based on id.
- Convert image/mask file format to --img_ext_out.

Expected image layout (input)
The script expects each dataset's images to already be separated into obv/ and rev/ folders.
Supported folder naming patterns (first match wins):
    - <root>/obv and <root>/rev
    - <root>/<dataset>_obv and <root>/<dataset>_rev
    - <root>/occ_obv and <root>/occ_rev (for OCC)
    - <root>/coinarchives_obv and <root>/coinarchives_rev (for CoinArchives)

Within each side folder, files are expected to be named:
    - image: <id>.<ext>
    - mask:  <id>_mask.<ext>
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("Missing dependency 'Pillow'. Install it via: pip install Pillow") from e

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Combine OCC and CoinArchives datasets and images."
        )
    )
    p.add_argument(
        "--coinarchives_input_csv",
        type=str,
        required=True,
        help='coinarchives_dataset.csv (columns: id,label,title,description,auction_date,url,img_url,weight,max_diameter)',
    )
    p.add_argument(
        "--occ_input_csv",
        type=str,
        required=True,
        help='occ_dataset.csv',
    )
    p.add_argument("--output_path", type=str, required=True, help="Output directory; creates obv/ and rev/")
    p.add_argument(
        "--coinarchives_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing CoinArchives side folders. "
            "Expected: <root>/obv and <root>/rev (or coinarchives_obv/coinarchives_rev). "
            "Default: <coinarchives_csv_dir>"
        ),
    )
    p.add_argument(
        "--occ_dir",
        type=str,
        default=None,
        help=(
            "Root directory containing OCC side folders. "
            "Expected: <root>/obv and <root>/rev (or occ_obv/occ_rev). "
            "Default: <occ_csv_dir>"
        ),
    )
    p.add_argument(
        "--img_ext_out",
        type=str,
        default=".png",
        help="Output image extension (default: .png). Use lossless formats for no quality loss.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images/masks if present",
    )
    p.add_argument(
        "--on_missing",
        type=str,
        default="error",
        choices=["error", "skip"],
        help="What to do if an id is missing an image or mask (default: error)",
    )
    return p.parse_args()

def main():
    args = parse_args()

    merge_occ_and_coinarchives_datasets(
        occ_input_csv=Path(args.occ_input_csv),
        coinarchives_input_csv=Path(args.coinarchives_input_csv),
        occ_dir_name=(Path(args.occ_dir) if args.occ_dir else None),
        coinarchives_dir_name=(Path(args.coinarchives_dir) if args.coinarchives_dir else None),
        img_ext_out=args.img_ext_out,
        output_path=Path(args.output_path),
        overwrite=bool(args.overwrite),
        on_missing=str(args.on_missing),
    )


def merge_occ_and_coinarchives_datasets(
    occ_input_csv: Path,
    coinarchives_input_csv: Path,
    occ_dir_name: Path = None,
    coinarchives_dir_name: Path = None,
    img_ext_out: str = ".jpg",
    output_path: Path = None,
    overwrite: bool = False,
    on_missing: str = "error",
) -> None:

    if output_path is None:
        raise ValueError("output_path is required")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    out_obv = output_path / "obv"
    out_rev = output_path / "rev"
    out_obv.mkdir(parents=True, exist_ok=True)
    out_rev.mkdir(parents=True, exist_ok=True)

    img_ext_out = _normalize_ext(img_ext_out)
    if img_ext_out in {".jpg", ".jpeg"}:
        print(
            "Warning: --img_ext_out is JPEG (lossy). "
            "If you need truly lossless conversion, use --img_ext_out .png or .tiff"
        )

    occ_input_csv = Path(occ_input_csv)
    coinarchives_input_csv = Path(coinarchives_input_csv)
    if not occ_input_csv.exists():
        raise FileNotFoundError(f"OCC CSV not found: {occ_input_csv}")
    if not coinarchives_input_csv.exists():
        raise FileNotFoundError(f"CoinArchives CSV not found: {coinarchives_input_csv}")

    occ_root = Path(occ_dir_name) if occ_dir_name else occ_input_csv.parent
    coin_root = Path(coinarchives_dir_name) if coinarchives_dir_name else coinarchives_input_csv.parent

    occ_obv_dir, occ_rev_dir = _resolve_side_dirs(occ_root, dataset="occ")
    coin_obv_dir, coin_rev_dir = _resolve_side_dirs(coin_root, dataset="coinarchives")

    occ_df = pd.read_csv(occ_input_csv, dtype=str, keep_default_na=False)
    coin_df = pd.read_csv(coinarchives_input_csv, dtype=str, keep_default_na=False)

    if "id" not in occ_df.columns:
        raise ValueError(f"OCC CSV must contain column 'id'. Found: {list(occ_df.columns)}")
    if "id" not in coin_df.columns:
        raise ValueError(f"CoinArchives CSV must contain column 'id'. Found: {list(coin_df.columns)}")

    occ_df["id"] = occ_df["id"].astype(str)
    coin_df["id"] = coin_df["id"].astype(str)

    # Add dataset column
    occ_df["dataset"] = "occ"
    coin_df["dataset"] = "coinarchives"

    # Handle potential id collisions by prefixing dataset
    occ_ids = set(occ_df["id"].tolist())
    coin_ids = set(coin_df["id"].tolist())
    colliding_ids = sorted(occ_ids.intersection(coin_ids))
    if colliding_ids:
        print(f"Warning: {len(colliding_ids)} colliding ids found across datasets; prefixing with dataset name")
        occ_df["id"] = [f"occ_{x}" if x in colliding_ids else x for x in occ_df["id"].tolist()]
        coin_df["id"] = [f"coinarchives_{x}" if x in colliding_ids else x for x in coin_df["id"].tolist()]

    # Column ordering: start with OCC schema, then append additional columns from either dataset
    occ_schema = [
        "id",
        "label",
        "obverse_description",
        "reverse_description",
        "collection",
        "inv_num",
        "weight",
        "max_diameter",
        "findspot",
        "label_alt",
    ]

    ordered_cols: List[str] = []
    for c in ["id", "dataset"]:
        if c not in ordered_cols:
            ordered_cols.append(c)
    for c in occ_schema:
        if c not in ordered_cols:
            ordered_cols.append(c)

    for c in list(occ_df.columns):
        if c not in ordered_cols:
            ordered_cols.append(c)
    for c in list(coin_df.columns):
        if c not in ordered_cols:
            ordered_cols.append(c)

    occ_df = _ensure_columns(occ_df, ordered_cols)
    coin_df = _ensure_columns(coin_df, ordered_cols)

    # Copy images/masks for both datasets; drop rows if requested.
    occ_df = _copy_assets_for_df(
        df=occ_df,
        src_obv_dir=occ_obv_dir,
        src_rev_dir=occ_rev_dir,
        dst_obv_dir=out_obv,
        dst_rev_dir=out_rev,
        img_ext_out=img_ext_out,
        overwrite=overwrite,
        on_missing=on_missing,
        dataset="occ",
        colliding_ids=set(colliding_ids),
    )
    coin_df = _copy_assets_for_df(
        df=coin_df,
        src_obv_dir=coin_obv_dir,
        src_rev_dir=coin_rev_dir,
        dst_obv_dir=out_obv,
        dst_rev_dir=out_rev,
        img_ext_out=img_ext_out,
        overwrite=overwrite,
        on_missing=on_missing,
        dataset="coinarchives",
        colliding_ids=set(colliding_ids),
    )

    merged_df = pd.concat([occ_df, coin_df], axis=0, ignore_index=True)
    merged_df = merged_df[ordered_cols]

    out_csv = output_path / "merged_dataset.csv"
    merged_df.to_csv(out_csv, index=False)

    print(f"Merged rows: {len(merged_df)} (occ={len(occ_df)}, coinarchives={len(coin_df)})")
    print(f"Wrote merged CSV: {out_csv}")
    print(f"Wrote images/masks to: {out_obv} and {out_rev}")


# ----------------------------- Helpers -----------------------------


_SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _normalize_ext(ext: str) -> str:
    ext = (ext or "").strip().lower()
    if not ext:
        raise ValueError("img_ext_out must be non-empty")
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _resolve_side_dirs(root: Path, dataset: str) -> Tuple[Path, Path]:
    root = Path(root)
    candidates: Sequence[Tuple[str, str]] = (
        ("obv", "rev"),
        (f"{dataset}_obv", f"{dataset}_rev"),
        ("occ_obv", "occ_rev"),
        ("coinarchives_obv", "coinarchives_rev"),
    )
    for obv_name, rev_name in candidates:
        obv_dir = root / obv_name
        rev_dir = root / rev_name
        if obv_dir.exists() and rev_dir.exists() and obv_dir.is_dir() and rev_dir.is_dir():
            return obv_dir, rev_dir
    raise FileNotFoundError(
        f"Could not find obv/rev side folders under: {root}. "
        "Expected either obv/rev, <dataset>_obv/<dataset>_rev, or occ_obv/occ_rev, coinarchives_obv/coinarchives_rev."
    )


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df


def _build_stem_index(folder: Path) -> Dict[str, List[Path]]:
    folder = Path(folder)
    index: Dict[str, List[Path]] = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in _SUPPORTED_EXTS:
            continue
        index.setdefault(p.stem, []).append(p)
    return index


def _pick_best(paths: List[Path]) -> Path:
    if len(paths) == 1:
        return paths[0]
    # Prefer PNG, then TIFF, then JPG.
    pref = {".png": 0, ".tif": 1, ".tiff": 1, ".jpg": 2, ".jpeg": 2, ".bmp": 3, ".webp": 4}
    return sorted(paths, key=lambda p: (pref.get(p.suffix.lower(), 99), str(p)))[0]


def _find_asset(stem_index: Dict[str, List[Path]], stem: str) -> Optional[Path]:
    paths = stem_index.get(stem)
    if not paths:
        return None
    return _pick_best(paths)


def _safe_copy_or_convert(
    *,
    src: Path,
    dst: Path,
    overwrite: bool,
    is_mask: bool,
    img_ext_out: str,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {dst} (use --overwrite)")

    if src.suffix.lower() == img_ext_out.lower():
        shutil.copy2(src, dst)
        return

    with Image.open(src) as im:
        im.load()

        if is_mask:
            # Preserve mask as single channel if possible.
            if im.mode not in ("L", "1"):
                im = im.convert("L")

        if img_ext_out in {".jpg", ".jpeg"}:
            if im.mode in ("RGBA", "LA", "P"):
                im = im.convert("RGB")
            im.save(dst, quality=100, subsampling=0, optimize=True)
            return

        if img_ext_out == ".png":
            im.save(dst, compress_level=0)
            return

        if img_ext_out in {".tif", ".tiff"}:
            # Uncompressed TIFF is lossless.
            im.save(dst, compression="raw")
            return

        # Fallback
        im.save(dst)


def _copy_assets_for_df(
    *,
    df: pd.DataFrame,
    src_obv_dir: Path,
    src_rev_dir: Path,
    dst_obv_dir: Path,
    dst_rev_dir: Path,
    img_ext_out: str,
    overwrite: bool,
    on_missing: str,
    dataset: str,
    colliding_ids: set,
) -> pd.DataFrame:
    """Copy obv/rev images+masks for each id in df.

    Note: if id collisions were handled by prefixing, source files still use original ids.
    This function reconstructs the source id by stripping the prefix.
    """

    src_obv_idx = _build_stem_index(src_obv_dir)
    src_rev_idx = _build_stem_index(src_rev_dir)

    kept_rows = []
    missing_count = 0

    for _, row in df.iterrows():
        out_id = str(row["id"])
        src_id = out_id
        prefix = f"{dataset}_"
        if out_id.startswith(prefix):
            src_id = out_id[len(prefix) :]

        obv_img = _find_asset(src_obv_idx, src_id)
        obv_mask = _find_asset(src_obv_idx, f"{src_id}_mask")
        rev_img = _find_asset(src_rev_idx, src_id)
        rev_mask = _find_asset(src_rev_idx, f"{src_id}_mask")

        if not (obv_img and obv_mask and rev_img and rev_mask):
            missing_count += 1
            if on_missing == "skip":
                continue
            raise FileNotFoundError(
                f"Missing assets for dataset={dataset}, id={src_id}. "
                f"Found obv_img={obv_img}, obv_mask={obv_mask}, rev_img={rev_img}, rev_mask={rev_mask}. "
                f"Searched in: {src_obv_dir} and {src_rev_dir}"
            )

        dst_obv_img = dst_obv_dir / f"{out_id}{img_ext_out}"
        dst_obv_mask = dst_obv_dir / f"{out_id}_mask{img_ext_out}"
        dst_rev_img = dst_rev_dir / f"{out_id}{img_ext_out}"
        dst_rev_mask = dst_rev_dir / f"{out_id}_mask{img_ext_out}"

        _safe_copy_or_convert(src=obv_img, dst=dst_obv_img, overwrite=overwrite, is_mask=False, img_ext_out=img_ext_out)
        _safe_copy_or_convert(src=obv_mask, dst=dst_obv_mask, overwrite=overwrite, is_mask=True, img_ext_out=img_ext_out)
        _safe_copy_or_convert(src=rev_img, dst=dst_rev_img, overwrite=overwrite, is_mask=False, img_ext_out=img_ext_out)
        _safe_copy_or_convert(src=rev_mask, dst=dst_rev_mask, overwrite=overwrite, is_mask=True, img_ext_out=img_ext_out)

        kept_rows.append(row)

    if missing_count:
        print(f"Warning: dataset={dataset} skipped {missing_count} rows due to missing assets")
    return pd.DataFrame(kept_rows).reset_index(drop=True)


if __name__ == "__main__":
    main()