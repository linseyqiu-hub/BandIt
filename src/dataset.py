import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import DebertaV2Tokenizer


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# The 5 scoring criteria — ORDER MATTERS.
# model.py and train.py must use this same list.
# Position 0 = Overall, 1 = Task_Response, ..., 4 = Range_Accuracy.
LABEL_COLUMNS = [
    "Overall",
    "Task_Response",
    "Coherence_Cohesion",
    "Lexical_Resource",
    "Range_Accuracy",
]

# DeBERTa-v3-base tokenizer identifier on HuggingFace Hub.
MODEL_NAME = "microsoft/deberta-v3-base"

# Max tokens per essay.
# DeBERTa-v3-base hard limit is 512.
# Essays longer than this are truncated; shorter ones are padded.
MAX_LENGTH = 512


# ------------------------------------------------------------------
# Dataset class
# ------------------------------------------------------------------

class IELTSDataset(Dataset):
    """
    PyTorch Dataset for IELTS essay scoring.

    Reads ielts_labeled.csv, tokenizes essays with the DeBERTa tokenizer,
    and returns (input_ids, attention_mask, labels) tensors on demand.

    Each call to __getitem__(idx) returns:
        {
            "input_ids":      torch.long tensor, shape (MAX_LENGTH,)
            "attention_mask": torch.long tensor, shape (MAX_LENGTH,)
            "labels":         torch.float tensor, shape (5,)
        }

    Tokenization is lazy — it happens in __getitem__, not __init__.
    This keeps construction fast and allows DataLoader worker parallelism.
    """

    def __init__(self, csv_path: str, tokenizer: DebertaV2Tokenizer, max_length: int = MAX_LENGTH):
        """
        Args:
            csv_path:   Path to ielts_labeled.csv.
            tokenizer:  Already-loaded DebertaV2Tokenizer instance.
            max_length: Token sequence length. Pad/truncate to this. Default 512.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # --- Load CSV ---
        df = pd.read_csv(csv_path)

        # --- Validate required columns exist ---
        required = ["Essay"] + LABEL_COLUMNS
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        # --- Drop rows where essay text or any score is missing ---
        before = len(df)
        df = df.dropna(subset=required)
        after = len(df)
        if before != after:
            print(f"[dataset] Dropped {before - after} rows with missing values. {after} rows remaining.")

        # --- Drop rows where essay is empty string ---
        df = df[df["Essay"].str.strip().str.len() > 0]
        if len(df) != after:
            print(f"[dataset] Dropped {after - len(df)} rows with empty essays. {len(df)} rows remaining.")

        # --- Reset index so __getitem__ can use iloc cleanly ---
        df = df.reset_index(drop=True)

        # Store only the columns we need — saves memory
        self.essays = df["Essay"].tolist()                        # list of strings
        self.labels = df[LABEL_COLUMNS].values.astype("float32")  # numpy array (N, 5)

        print(f"[dataset] Loaded {len(self.essays)} essays.")
        print(f"[dataset] Score ranges:")
        for i, col in enumerate(LABEL_COLUMNS):
            col_vals = self.labels[:, i]
            print(f"  {col}: min={col_vals.min():.1f}, max={col_vals.max():.1f}, mean={col_vals.mean():.2f}")

    def __len__(self) -> int:
        """Returns total number of essays in this dataset."""
        return len(self.essays)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns one sample as a dict of tensors.

        Tokenization happens here — lazily, on demand.

        Args:
            idx: Integer index, 0 to len(dataset)-1.

        Returns:
            {
                "input_ids":      LongTensor (MAX_LENGTH,)
                "attention_mask": LongTensor (MAX_LENGTH,)
                "labels":         FloatTensor (5,)
            }
        """
        essay = self.essays[idx]

        # Tokenize the essay.
        # padding="max_length" pads short essays to MAX_LENGTH with 0s.
        # truncation=True cuts essays longer than MAX_LENGTH.
        # return_tensors="pt" gives us PyTorch tensors directly.
        encoded = self.tokenizer(
            essay,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # tokenizer returns shape (1, MAX_LENGTH) because it expects batches.
        # We squeeze out the batch dim to get (MAX_LENGTH,).
        input_ids      = encoded["input_ids"].squeeze(0)       # (512,)
        attention_mask = encoded["attention_mask"].squeeze(0)  # (512,)

        # Labels: numpy row → FloatTensor (5,)
        labels = torch.tensor(self.labels[idx], dtype=torch.float)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ------------------------------------------------------------------
# Helper: build train/val datasets + dataloaders in one call
# ------------------------------------------------------------------

def get_dataloaders(
    csv_path: str,
    batch_size: int = 8,
    val_split: float = 0.2,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Convenience function used by train.py.

    Loads the full dataset, splits into train/val, wraps in DataLoaders.

    Args:
        csv_path:    Path to ielts_labeled.csv.
        batch_size:  Samples per batch. Default 8 (safe for CPU).
        val_split:   Fraction of data for validation. Default 0.2.
        max_length:  Token sequence length. Default 512.
        num_workers: DataLoader worker processes. Default 0 (main process only).
                     On Windows, keep this 0 to avoid multiprocessing issues.
        seed:        Random seed for reproducible split.

    Returns:
        train_loader: DataLoader for training set
        val_loader:   DataLoader for validation set
    """
    # Load tokenizer
    print(f"[dataset] Loading tokenizer: {MODEL_NAME}")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

    # Build the full dataset
    full_dataset = IELTSDataset(csv_path, tokenizer, max_length)

    # Compute split sizes
    total      = len(full_dataset)
    val_size   = int(total * val_split)
    train_size = total - val_size
    print(f"[dataset] Split: {train_size} train / {val_size} val")

    # Random split — uses a Generator for reproducibility
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # DataLoaders
    # shuffle=True for train (expose different orderings each epoch)
    # shuffle=False for val (order doesn't matter for evaluation)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader


# ------------------------------------------------------------------
# Quick smoke test — run this file directly to verify everything works
# python src/dataset.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    csv_path = "data/ielts_labeled.csv"

    print("=" * 50)
    print("dataset.py smoke test")
    print("=" * 50)

    # 1. Load tokenizer
    print(f"\nLoading tokenizer from {MODEL_NAME} ...")
    tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.")

    # 2. Build dataset
    print(f"\nBuilding dataset from {csv_path} ...")
    dataset = IELTSDataset(csv_path, tokenizer)

    # 3. Check length
    print(f"\nTotal samples: {len(dataset)}")

    # 4. Fetch one sample and inspect shapes + dtypes
    print("\nFetching sample[0] ...")
    sample = dataset[0]
    for key, val in sample.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    # 5. Verify input_ids values are in valid range
    vocab_size = tokenizer.vocab_size
    assert sample["input_ids"].max().item() < vocab_size, "input_ids out of vocab range!"
    print(f"\n  input_ids max={sample['input_ids'].max().item()} < vocab_size={vocab_size} ✓")

    # 6. Verify attention_mask is only 0s and 1s
    unique_mask_vals = sample["attention_mask"].unique().tolist()
    assert all(v in [0, 1] for v in unique_mask_vals), "attention_mask has values other than 0 and 1!"
    print(f"  attention_mask unique values: {unique_mask_vals} ✓")

    # 7. Verify labels are in IELTS band range [0, 9]
    labels = sample["labels"]
    assert labels.min().item() >= 0 and labels.max().item() <= 9, "Labels out of IELTS band range!"
    print(f"  labels: {labels.tolist()} ✓")

    # 8. Build dataloaders and fetch one batch
    print("\nBuilding dataloaders (batch_size=4) ...")
    train_loader, val_loader = get_dataloaders(csv_path, batch_size=4)

    batch = next(iter(train_loader))
    print("\nFirst training batch:")
    for key, val in batch.items():
        print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

    print("\n" + "=" * 50)
    print("All checks passed. dataset.py is ready.")
    print("=" * 50)