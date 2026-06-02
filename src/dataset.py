import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

LABEL_COLUMNS = [
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512


# ------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------

class IELTSDataset(Dataset):
    """
    PyTorch Dataset for IELTS essay scoring — v4.

    Each sample returns TWO tokenizations:
        - question only  →  q_input_ids, q_attention_mask
        - essay only     →  e_input_ids, e_attention_mask

    This matches the v4 dual-encoder architecture in model.py:
        - question encoding feeds cross-attention with essay for TR
        - essay encoding feeds CC, LR, RA heads directly

    The old concatenated "Question: {q}\\n\\nEssay: {e}" format is gone.
    """

    def __init__(self, csv_path: str, tokenizer: DebertaV2Tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer  = tokenizer
        self.max_length = max_length

        df = pd.read_csv(csv_path)

        required = ["Question", "Essay"] + LABEL_COLUMNS
        missing  = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        before = len(df)
        df     = df.dropna(subset=required)
        after  = len(df)
        if before != after:
            print(f"[dataset] Dropped {before - after} rows with missing values. {after} remaining.")

        df = df[df["Essay"].str.strip().str.len() > 0]
        df = df[df["Question"].str.strip().str.len() > 0]
        df = df.reset_index(drop=True)

        self.questions = df["Question"].tolist()
        self.essays    = df["Essay"].tolist()
        self.labels    = df[LABEL_COLUMNS].values.astype("float32")  # (N, 4)

        print(f"[dataset] Loaded {len(self.essays)} essays.")
        print(f"[dataset] Score ranges:")
        for i, col in enumerate(LABEL_COLUMNS):
            col_vals = self.labels[:, i]
            print(f"  {col}: min={col_vals.min():.1f}, max={col_vals.max():.1f}, mean={col_vals.mean():.2f}")

    def __len__(self) -> int:
        return len(self.essays)

    def _tokenize(self, text: str) -> dict:
        """
        Tokenizes a single string to (MAX_LENGTH,) tensors.
        Padding and truncation settings must be identical to inference.py.
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoded["input_ids"].squeeze(0),       # (512,)
            "attention_mask": encoded["attention_mask"].squeeze(0),  # (512,)
        }

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one sample with two tokenizations.

        Keys:
            q_input_ids:      (512,) — question only
            q_attention_mask: (512,)
            e_input_ids:      (512,) — essay only
            e_attention_mask: (512,)
            labels:           (4,)   — [TR, CC, LR, RA]
        """
        q_enc = self._tokenize(self.questions[idx])
        e_enc = self._tokenize(self.essays[idx])

        return {
            "q_input_ids":      q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "e_input_ids":      e_enc["input_ids"],
            "e_attention_mask": e_enc["attention_mask"],
            "labels":           torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ------------------------------------------------------------------
# Helper: build train/val dataloaders
# ------------------------------------------------------------------

def get_dataloaders(
    csv_path:    str,
    batch_size:  int   = 8,
    val_split:   float = 0.2,
    max_length:  int   = MAX_LENGTH,
    num_workers: int   = 0,
    seed:        int   = 42,
):
    print(f"[dataset] Loading tokenizer: {MODEL_NAME}")
    tokenizer    = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    full_dataset = IELTSDataset(csv_path, tokenizer, max_length)

    total         = len(full_dataset)
    overall_scores = full_dataset.labels[:, 0]
    bins           = np.array([score_to_stratum(s) for s in overall_scores])
    indices        = np.arange(total)

    train_indices, val_indices = train_test_split(
        indices,
        test_size    = val_split,
        random_state = seed,
        stratify     = bins,
    )
    print(f"[dataset] Stratified split: {len(train_indices)} train / {len(val_indices)} val")

    train_bins = bins[train_indices]
    val_bins   = bins[val_indices]
    print("[dataset] Band distribution check:")
    for b in sorted(np.unique(bins)):
        print(f"  Band {b}: {(train_bins==b).sum()} train / {(val_bins==b).sum()} val  "
              f"(total {(bins==b).sum()})")

    from torch.utils.data import Subset
    train_loader = DataLoader(
        Subset(full_dataset, train_indices),
        batch_size=batch_size, shuffle=True,  num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_indices),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


def score_to_stratum(score):
    if score < 4.5:   return 0
    elif score < 6.5: return 1
    elif score < 8.0: return 2
    else:             return 3


# ------------------------------------------------------------------
# Smoke test
# python src/dataset.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "data/ielts_relabeled.csv"

    print("=" * 50)
    print("dataset.py smoke test — v4")
    print("=" * 50)

    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    dataset   = IELTSDataset(csv_path, tokenizer)

    print(f"\nTotal samples: {len(dataset)}")

    sample = dataset[0]
    print("\nSample[0] keys and shapes:")
    for key, val in sample.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # Verify shapes
    assert sample["q_input_ids"].shape      == (MAX_LENGTH,), "q_input_ids shape wrong"
    assert sample["q_attention_mask"].shape == (MAX_LENGTH,), "q_attention_mask shape wrong"
    assert sample["e_input_ids"].shape      == (MAX_LENGTH,), "e_input_ids shape wrong"
    assert sample["e_attention_mask"].shape == (MAX_LENGTH,), "e_attention_mask shape wrong"
    assert sample["labels"].shape           == (4,),          "labels shape wrong"
    print("\nAll shapes correct ✓")

    # Question and essay should tokenize differently
    assert not torch.equal(sample["q_input_ids"], sample["e_input_ids"]), \
        "q and e input_ids are identical — something is wrong"
    print("q_input_ids != e_input_ids ✓  (question and essay tokenize differently)")

    print("\nBuilding dataloaders (batch_size=4)...")
    train_loader, val_loader = get_dataloaders(csv_path, batch_size=4)
    batch = next(iter(train_loader))
    print("\nFirst training batch:")
    for key, val in batch.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    print("\n" + "=" * 50)
    print("All checks passed. dataset.py v4 ready.")
    print("=" * 50)