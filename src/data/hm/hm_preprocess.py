# File src/data/hm/hm_preprocess.py
"""
v5 — No filter + memory-safe split.
split_data() dùng chunk để tránh OOM khi load 31M rows.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

_VAL_START  = pd.Timestamp("2020-08-25")
_TEST_START = pd.Timestamp("2020-09-08")
_TEST_END   = pd.Timestamp("2020-09-22")
_CHUNK      = 500_000


class HMPreprocess:
    def __init__(self, raw_path, output_path):
        self.raw_path    = Path(raw_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.all_ids     = None

    # =========================
    # 1. LOAD ARTICLES
    # =========================
    def load_articles(self):
        print("[HM] Loading articles (no filter)...")
        articles = pd.read_csv(self.raw_path / "articles.csv")
        articles["article_id"] = articles["article_id"].astype(str).str.zfill(10)
        self.valid_articles_df = articles
        self.all_ids = set(articles["article_id"].unique())
        print(f"[HM] Total articles: {len(self.all_ids):,}")

    # =========================
    # 2. SAVE ARTICLES
    # =========================
    def save_articles(self):
        self.valid_articles_df.to_csv(
            self.output_path / "articles.csv", index=False
        )
        print("[HM] Saved articles")

    # =========================
    # 3. LOAD TRANSACTIONS
    # =========================
    def load_transactions(self):
        print("[HM] Loading transactions...")
        out_path = self.output_path / "transactions.csv"
        first = True
        for chunk in pd.read_csv(
            self.raw_path / "transactions_train.csv",
            usecols=["t_dat", "customer_id", "article_id"],
            dtype={"customer_id": "string", "article_id": "string"},
            chunksize=_CHUNK,
        ):
            chunk["article_id"] = chunk["article_id"].str.zfill(10)
            chunk = chunk[chunk["article_id"].isin(self.all_ids)]
            chunk.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
            first = False
        print("[HM] DONE transactions")

    # =========================
    # 4. SPLIT — CHUNK-BASED (no OOM)
    # =========================
    def split_data(self):
        """
        Đọc transactions.csv theo chunk, ghi thẳng vào 3 file riêng.
        Không bao giờ load toàn bộ file vào RAM cùng lúc.
        """
        print("[HM] Splitting by date (chunk-based)...")

        tx_path = self.output_path / "transactions.csv"
        paths   = {
            "train": self.output_path / "train.csv",
            "val":   self.output_path / "val.csv",
            "test":  self.output_path / "test.csv",
        }
        handles = {k: open(v, "w") for k, v in paths.items()}
        counts  = {"train": 0, "val": 0, "test": 0}
        header_written = {"train": False, "val": False, "test": False}

        for chunk in pd.read_csv(tx_path, chunksize=_CHUNK,
                                 dtype={"customer_id": "string", "article_id": "string"}):
            chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])

            masks = {
                "train": chunk["t_dat"] < _VAL_START,
                "val":   (chunk["t_dat"] >= _VAL_START) & (chunk["t_dat"] < _TEST_START),
                "test":  (chunk["t_dat"] >= _TEST_START) & (chunk["t_dat"] <= _TEST_END),
            }

            for split, mask in masks.items():
                part = chunk[mask]
                if len(part) == 0:
                    continue
                part.to_csv(
                    handles[split],
                    header=not header_written[split],
                    index=False,
                )
                header_written[split] = True
                counts[split] += len(part)

        for h in handles.values():
            h.close()

        print(f"[HM] train: {counts['train']:,}  |  val: {counts['val']:,}  |  test: {counts['test']:,}")
        print("[HM] DONE split")

    # =========================
    # 5. RESIZE IMAGES
    # =========================
    def resize_images(self, size=(224, 224)):
        print("[HM] Resizing images...")
        input_dir  = self.raw_path / "images"
        output_dir = self.output_path / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        valid_images = [
            (aid, input_dir / aid[:3] / f"{aid}.jpg")
            for aid in self.all_ids
            if (input_dir / aid[:3] / f"{aid}.jpg").exists()
        ]
        print(f"[HM] Found images: {len(valid_images):,}")

        kept = 0
        for article_id, img_path in tqdm(valid_images, desc="Resizing"):
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size, Image.BILINEAR)
                sub_dir = output_dir / article_id[:3]
                sub_dir.mkdir(parents=True, exist_ok=True)
                img.save(sub_dir / f"{article_id}.jpg", quality=90)
                img.close()
                kept += 1
            except Exception as e:
                print(f"[ERROR] {img_path}: {e}")

        print(f"[HM] DONE images: {kept:,}")

    # =========================
    # 6. RUN
    # =========================
    def run(self):
        self.load_articles()
        self.save_articles()
        self.load_transactions()
        self.split_data()
        self.resize_images()


if __name__ == "__main__":
    pre = HMPreprocess(
        raw_path="data/raw/hm",
        output_path="data/processed/hm"
    )
    pre.run()