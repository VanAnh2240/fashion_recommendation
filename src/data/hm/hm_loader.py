# File src/data/hm/hm_loader.py

from pathlib import Path
import pandas as pd
import torch


class HMLoader:
    def __init__(self, processed_path):
        self.path = Path(processed_path)

    def load_split(self, split="train"):
        return pd.read_csv(self.path / f"{split}.csv")

    def build_interactions(self, df):
        users = df["customer_id"].astype("category").cat.codes
        items = df["article_id"].astype("category").cat.codes

        edge_index = torch.tensor(
            [users.values, items.values],
            dtype=torch.long
        )

        return edge_index

    def get_all(self):
        return (
            self.load_split("train"),
            self.load_split("val"),
            self.load_split("test")
        )