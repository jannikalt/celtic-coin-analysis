import csv
import os
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageTk
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Pillow is required for this GUI. Install with: pip install pillow\n"
        f"Original error: {exc}"
    )


REQUIRED_COLUMNS = [
    "id",
    "dataset",
    "label",
    "obverse_description",
    "reverse_description",
    "collection",
    "inv_num",
    "weight",
    "max_diameter",
    "findspot",
    "label_alt",
    "title",
    "description",
    "auction_date",
    "url",
    "img_url",
]

SIDE_CONDITION_COLUMNS = {
    "obv": "obv_condition",
    "rev": "rev_condition",
}

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


@dataclass
class ViewState:
    coin_index: int = 0
    side: str = "obv"  # "obv" or "rev"


class ConditionRaterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Coin Condition Rater")

        self.csv_path: Optional[str] = None
        self.image_root: Optional[str] = None

        self.rows: List[Dict[str, str]] = []
        self.fieldnames: List[str] = []
        self.dirty: bool = False

        self.state = ViewState(coin_index=0, side="obv")

        self._coin_photo: Optional[ImageTk.PhotoImage] = None
        self._mask_photo: Optional[ImageTk.PhotoImage] = None

        self._suspend_rating_callback: bool = False

        self.rating_var = tk.IntVar(value=-1)

        self._build_ui()
        self._bind_keys()

        self._set_enabled(False)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)

        toolbar = ttk.Frame(outer)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(5, weight=1)

        self.load_btn = ttk.Button(toolbar, text="Load", command=self.load_csv)
        self.store_btn = ttk.Button(toolbar, text="Store", command=self.store_csv)
        self.back_btn = ttk.Button(toolbar, text="Back", command=self.go_back)
        self.next_btn = ttk.Button(toolbar, text="Next", command=self.go_next)

        self.load_btn.grid(row=0, column=0, padx=(0, 6))
        self.store_btn.grid(row=0, column=1, padx=(0, 18))
        self.back_btn.grid(row=0, column=2, padx=(0, 6))
        self.next_btn.grid(row=0, column=3, padx=(0, 6))

        self.status_var = tk.StringVar(value="Load a CSV to begin.")
        self.status_lbl = ttk.Label(toolbar, textvariable=self.status_var)
        self.status_lbl.grid(row=0, column=5, sticky="e")

        info = ttk.Frame(outer)
        info.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        info.columnconfigure(1, weight=1)

        ttk.Label(info, text="ID:").grid(row=0, column=0, sticky="w")
        self.id_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.id_var).grid(row=0, column=1, sticky="w")

        ttk.Label(info, text="Side:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        self.side_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.side_var).grid(row=0, column=3, sticky="w")

        ttk.Label(info, text="Label:").grid(row=1, column=0, sticky="w")
        self.label_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.label_var).grid(row=1, column=1, sticky="w")

        ttk.Label(info, text="Dataset:").grid(row=1, column=2, sticky="w", padx=(18, 0))
        self.dataset_var = tk.StringVar(value="")
        ttk.Label(info, textvariable=self.dataset_var).grid(row=1, column=3, sticky="w")

        main = ttk.Frame(outer)
        main.grid(row=2, column=0, sticky="nsew")
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.image_lbl = ttk.Label(main, text="", anchor="center")
        self.image_lbl.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.mask_lbl = ttk.Label(main, text="", anchor="center")
        self.mask_lbl.grid(row=0, column=1, sticky="nsew")

        rating = ttk.Frame(outer)
        rating.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        ttk.Label(rating, text="Condition:").grid(row=0, column=0, sticky="w")
        for i in (0, 1, 2):
            rb = ttk.Radiobutton(
                rating,
                text=str(i),
                value=i,
                variable=self.rating_var,
                command=self.on_rating_selected,
            )
            rb.grid(row=0, column=i + 1, padx=(8, 0), sticky="w")

        self.unsaved_var = tk.StringVar(value="")
        self.unsaved_lbl = ttk.Label(rating, textvariable=self.unsaved_var)
        self.unsaved_lbl.grid(row=0, column=10, sticky="e")
        rating.columnconfigure(10, weight=1)

    def _bind_keys(self) -> None:
        self.root.bind("<Left>", lambda _e: self.go_back())
        self.root.bind("<Right>", lambda _e: self.go_next())
        self.root.bind("0", lambda _e: self._set_rating_and_advance(0))
        self.root.bind("1", lambda _e: self._set_rating_and_advance(1))
        self.root.bind("2", lambda _e: self._set_rating_and_advance(2))

    def _set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.store_btn.configure(state=state)
        self.back_btn.configure(state=state)
        self.next_btn.configure(state=state)

    def _ensure_can_discard(self) -> bool:
        if not self.dirty:
            return True
        resp = messagebox.askyesno(
            "Unsaved changes",
            "You have unsaved condition ratings. Continue without storing?",
        )
        return bool(resp)

    def load_csv(self) -> None:
        if not self._ensure_can_discard():
            return

        csv_path = filedialog.askopenfilename(
            title="Open CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not csv_path:
            return

        try:
            rows, fieldnames = self._read_csv(csv_path)
        except Exception as exc:
            messagebox.showerror("Failed to read CSV", str(exc))
            return

        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            messagebox.showerror(
                "Missing columns",
                "The CSV is missing required columns:\n\n" + "\n".join(missing),
            )
            return

        # Ensure per-side condition columns exist.
        for col in ("obv_condition", "rev_condition"):
            if col not in fieldnames:
                fieldnames = list(fieldnames) + [col]
                for row in rows:
                    row[col] = ""

        img_root_default = os.path.dirname(os.path.abspath(csv_path))
        image_root = filedialog.askdirectory(
            title="Select image root folder (contains obv/ and rev/)",
            initialdir=img_root_default,
        )
        if not image_root:
            image_root = img_root_default

        self.csv_path = csv_path
        self.image_root = image_root
        self.rows = rows
        self.fieldnames = fieldnames
        self.dirty = False

        # Jump to first unrated side (obv first, then rev).
        start_index = 0
        start_side = "obv"
        found = False
        for idx, row in enumerate(self.rows):
            obv = (row.get("obv_condition") or "").strip()
            rev = (row.get("rev_condition") or "").strip()
            if obv not in ("0", "1", "2"):
                start_index, start_side, found = idx, "obv", True
                break
            if rev not in ("0", "1", "2"):
                start_index, start_side, found = idx, "rev", True
                break
        if not found:
            start_index, start_side = 0, "obv"
        self.state = ViewState(coin_index=start_index, side=start_side)

        self._set_enabled(True)
        self._refresh_view()

    def store_csv(self) -> None:
        if not self.rows or not self.fieldnames:
            return
        if self.csv_path is None:
            return

        default_dir = os.path.dirname(os.path.abspath(self.csv_path))
        base, ext = os.path.splitext(os.path.basename(self.csv_path))
        default_name = f"{base}_rated{ext or '.csv'}"

        target = filedialog.asksaveasfilename(
            title="Store CSV",
            defaultextension=".csv",
            initialdir=default_dir,
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*")],
        )
        if not target:
            return

        try:
            # If overwriting an existing file, create a simple backup.
            if os.path.exists(target):
                backup = target + ".bak"
                shutil.copy2(target, backup)

            self._write_csv(target, self.rows, self.fieldnames)
        except Exception as exc:
            messagebox.showerror("Failed to store CSV", str(exc))
            return

        self.dirty = False
        self._update_unsaved()
        messagebox.showinfo("Stored", f"Saved ratings to:\n{target}")

    def go_next(self) -> None:
        if not self.rows:
            return
        coin_index, side = self.state.coin_index, self.state.side

        if side == "obv":
            self.state.side = "rev"
        else:
            if coin_index < len(self.rows) - 1:
                self.state.coin_index += 1
                self.state.side = "obv"

        self._refresh_view()

    def go_back(self) -> None:
        if not self.rows:
            return
        coin_index, side = self.state.coin_index, self.state.side

        if side == "rev":
            self.state.side = "obv"
        else:
            if coin_index > 0:
                self.state.coin_index -= 1
                self.state.side = "rev"

        self._refresh_view()

    def _set_rating_and_advance(self, rating: int) -> None:
        if not self.rows:
            return
        # Programmatic rating (keyboard). Set variable then handle as if clicked.
        self.rating_var.set(rating)
        self.on_rating_selected()

    def on_rating_selected(self) -> None:
        if not self.rows:
            return

        # Guard against callbacks fired by programmatic variable updates.
        if self._suspend_rating_callback:
            return

        rating = self.rating_var.get()
        if rating not in (0, 1, 2):
            return

        row = self.rows[self.state.coin_index]
        col = SIDE_CONDITION_COLUMNS.get(self.state.side)
        if col is None:
            return
        row[col] = str(rating)
        self.dirty = True
        self._update_unsaved()

        # Auto-advance: obv -> rev, rev -> next coin obv
        if self.state.side == "obv":
            self.state.side = "rev"
        else:
            if self.state.coin_index < len(self.rows) - 1:
                self.state.coin_index += 1
                self.state.side = "obv"

        # Defer refresh until Tk finishes processing the click event;
        # otherwise the radiobutton can re-apply its selection after we update.
        self.root.after_idle(self._refresh_view)

    def _refresh_view(self) -> None:
        if not self.rows:
            self.status_var.set("Load a CSV to begin.")
            self.image_lbl.configure(text="", image="")
            self.mask_lbl.configure(text="", image="")
            return

        row = self.rows[self.state.coin_index]
        coin_id = (row.get("id") or "").strip()
        dataset = (row.get("dataset") or "").strip()
        label = (row.get("label") or "").strip()

        self.id_var.set(coin_id)
        self.dataset_var.set(dataset)
        self.label_var.set(label)
        self.side_var.set(self.state.side)

        col = SIDE_CONDITION_COLUMNS.get(self.state.side)
        condition = (row.get(col, "") if col else "")
        condition = (condition or "").strip()
        self._suspend_rating_callback = True
        try:
            if condition in ("0", "1", "2"):
                self.rating_var.set(int(condition))
            else:
                # Clear selection when unrated.
                self.rating_var.set(-1)
        finally:
            self._suspend_rating_callback = False

        self._update_unsaved()

        n = len(self.rows)
        self.status_var.set(
            f"{self.state.coin_index + 1}/{n}  |  {coin_id}  |  {self.state.side}  |  {col or 'condition'}={condition or '∅'}"
        )

        self._show_image_for(row, self.state.side)

    def _update_unsaved(self) -> None:
        self.unsaved_var.set("Unsaved changes" if self.dirty else "")

    def _show_image_for(self, row: Dict[str, str], side: str) -> None:
        if self.image_root is None:
            self._coin_photo = None
            self._mask_photo = None
            self.image_lbl.configure(text="No image root selected.", image="")
            self.mask_lbl.configure(text="", image="")
            return

        coin_id = (row.get("id") or "").strip()
        dataset = (row.get("dataset") or "").strip()

        image_path = self._find_image_path(self.image_root, dataset, side, coin_id)
        if image_path is None:
            self._coin_photo = None
            self.image_lbl.configure(
                text=f"Image not found for {coin_id} ({side}).\n\nExpected under: {os.path.join(self.image_root, side)}",
                image="",
                anchor="center",
                justify="center",
            )
            self._mask_photo = None
            self.mask_lbl.configure(text="", image="")
            return

        try:
            img = Image.open(image_path)
            img = img.convert("RGB")
        except Exception as exc:
            self._coin_photo = None
            self.image_lbl.configure(
                text=f"Failed to open image:\n{image_path}\n\n{exc}",
                image="",
                anchor="center",
                justify="center",
            )
            self._mask_photo = None
            self.mask_lbl.configure(text="", image="")
            return

        # Fit to available window size (roughly). Recompute on each refresh.
        window_w = max(800, self.root.winfo_width())
        window_h = max(600, self.root.winfo_height())
        max_h = max(300, window_h - 220)
        coin_max_w = max(400, int((window_w - 60) * 0.72))
        mask_max_w = max(200, int((window_w - 60) * 0.22))

        img = self._resize_to_fit(img, coin_max_w, max_h)
        self._coin_photo = ImageTk.PhotoImage(img)
        self.image_lbl.configure(text="", image=self._coin_photo, anchor="center")

        mask_path = self._find_mask_path(image_path)
        if mask_path is None:
            self._mask_photo = None
            self.mask_lbl.configure(text="Mask not found", image="", anchor="center")
            return

        try:
            mask_img = Image.open(mask_path)
            mask_img = mask_img.convert("RGB")
        except Exception as exc:
            self._mask_photo = None
            self.mask_lbl.configure(text=f"Failed to open mask\n{exc}", image="", anchor="center")
            return

        mask_img = self._resize_to_fit(mask_img, mask_max_w, max_h)
        self._mask_photo = ImageTk.PhotoImage(mask_img)
        self.mask_lbl.configure(text="", image=self._mask_photo, anchor="center")

    @staticmethod
    def _resize_to_fit(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
        w, h = img.size
        if w <= 0 or h <= 0:
            return img
        scale = min(max_w / w, max_h / h)
        if scale >= 1.0:
            return img
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        return img.resize(new_size, Image.Resampling.LANCZOS)

    @staticmethod
    def _read_csv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header row.")
            fieldnames = [name.strip() for name in reader.fieldnames]

            rows: List[Dict[str, str]] = []
            for row in reader:
                normalized: Dict[str, str] = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    normalized[key.strip()] = "" if value is None else str(value)
                rows.append(normalized)

        return rows, fieldnames

    @staticmethod
    def _write_csv(path: str, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = {k: row.get(k, "") for k in fieldnames}
                writer.writerow(out)

    @staticmethod
    def _find_image_path(image_root: str, dataset: str, side: str, coin_id: str) -> Optional[str]:
        if not coin_id:
            return None

        # Try a few common layouts.
        # 1) <root>/<side>/<id>.<ext>
        # 2) <root>/<dataset>/<side>/<id>.<ext>
        # 3) If id already has an extension, try exact.
        candidates: List[str] = []

        def add_candidates(base_dir: str) -> None:
            # exact file
            candidates.append(os.path.join(base_dir, coin_id))
            root_no_ext, ext = os.path.splitext(coin_id)
            if ext:
                return
            for e in IMAGE_EXTS:
                candidates.append(os.path.join(base_dir, coin_id + e))

        add_candidates(os.path.join(image_root, side))
        if dataset:
            add_candidates(os.path.join(image_root, dataset, side))

        for p in candidates:
            if os.path.isfile(p):
                return p

        # Final fallback: scan the folder for files starting with id.
        for base_dir in [os.path.join(image_root, side), os.path.join(image_root, dataset, side) if dataset else None]:
            if not base_dir or not os.path.isdir(base_dir):
                continue
            try:
                for name in os.listdir(base_dir):
                    if name.startswith(coin_id + ".") or name == coin_id:
                        full = os.path.join(base_dir, name)
                        if os.path.isfile(full):
                            return full
            except OSError:
                continue

        return None

    @staticmethod
    def _find_mask_path(image_path: str) -> Optional[str]:
        base, ext = os.path.splitext(image_path)
        direct = base + "_mask" + ext
        if os.path.isfile(direct):
            return direct

        # Try any extension if the mask extension differs.
        base_dir = os.path.dirname(image_path)
        name_root = os.path.basename(base) + "_mask"
        for e in IMAGE_EXTS:
            p = os.path.join(base_dir, name_root + e)
            if os.path.isfile(p):
                return p

        # Final fallback: scan folder for '<name>_mask.*'
        try:
            for name in os.listdir(base_dir):
                if name.startswith(name_root + "."):
                    full = os.path.join(base_dir, name)
                    if os.path.isfile(full):
                        return full
        except OSError:
            return None

        return None


def main() -> int:
    root = tk.Tk()
    # Reasonable default size; image will scale down if needed.
    root.geometry("1100x850")
    app = ConditionRaterApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (root.destroy() if app._ensure_can_discard() else None))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
