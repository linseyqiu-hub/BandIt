import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

from dataset import MODEL_NAME, LABEL_COLUMNS


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# Size of DeBERTa-v3-base's hidden dimension.
# 12 attention heads × 64 dims per head = 768.
# This is the size of the CLS vector we read from the backbone.
HIDDEN_SIZE = 768

# Number of scoring criteria = number of output neurons in the head.
NUM_LABELS = len(LABEL_COLUMNS)  # 4


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class BandItScorer(nn.Module):
    """
    IELTS essay scoring model.

    Architecture:
        DeBERTa-v3-base (pretrained backbone, 86M params)
            ↓
        [CLS] token vector (768,)   ← essay summary
            ↓
        Linear(768 → 5)             ← scoring head
            ↓
        [Overall, Task_Response, Coherence_Cohesion, Lexical_Resource, Range_Accuracy]

    The backbone extracts a rich semantic representation of the essay.
    The scoring head maps that representation to 5 band scores.

    Both are updated during fine-tuning, but at different learning rates:
        - backbone: small lr (e.g. 2e-5) — nudge pretrained weights
        - head:     larger lr (e.g. 1e-3) — train from scratch
    train.py handles the learning rate split via parameter groups.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        """
        Args:
            pretrained: If True, load pretrained DeBERTa weights from HuggingFace.
                        If False, initialise DeBERTa with random weights (for testing only).
            dropout:    Dropout rate applied to CLS vector before the scoring head.
                        Regularisation — reduces overfitting on small dataset (1435 essays).
        """
        super().__init__()

        # --- Backbone ---
        # DebertaV2Model is the bare transformer — no task-specific head on top.
        # It outputs hidden states for every token position.
        # We do NOT use DebertaV2ForSequenceClassification because that adds
        # HuggingFace's own classification head. We want to attach our own.
        if pretrained:
            self.deberta = DebertaV2Model.from_pretrained(MODEL_NAME)
        else:
            # random weights — only used in smoke test to skip the download
            config = DebertaV2Config.from_pretrained(MODEL_NAME)
            self.deberta = DebertaV2Model(config)

        # --- Dropout ---
        # Applied to the CLS vector before the scoring head.
        # During training: randomly zeros out some of the 768 features.
        # During inference: disabled automatically when model.eval() is called.
        self.dropout = nn.Dropout(dropout)

        # --- Scoring head ---
        # Linear(768 → 5): one weight matrix (5, 768) + bias (5,).
        # Randomly initialised — learns from scratch during fine-tuning.
        # No activation function — raw regression output, not classification.
        self.scoring_head = nn.Linear(HIDDEN_SIZE, NUM_LABELS)

        # Initialise scoring head weights with small values.
        # Default PyTorch init is fine, but explicit init makes behaviour clear.
        nn.init.normal_(self.scoring_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.scoring_head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — maps tokenized essays to band score predictions.

        Args:
            input_ids:      LongTensor, shape (batch, 512)
                            Token IDs from the DeBERTa tokenizer.
            attention_mask: LongTensor, shape (batch, 512)
                            1 for real tokens, 0 for padding.

        Returns:
            scores: FloatTensor, shape (batch, 5)
                    Predicted band scores in order:
                    [Overall, Task_Response, Coherence_Cohesion, Lexical_Resource, Range_Accuracy]
        """

        # --- Backbone forward pass ---
        # DeBERTa processes the full token sequence through 12 transformer blocks.
        # last_hidden_state contains output vectors for every token position.
        # shape: (batch, 512, 768)
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # --- Extract CLS vector ---
        # Position 0 is always [CLS] — the essay summary vector.
        # After 12 layers of bidirectional attention, it has attended to
        # every token in the essay and aggregates global meaning.
        # shape: (batch, 512, 768) → (batch, 768)
        cls_vector = outputs.last_hidden_state[:, 0, :]

        # --- Dropout ---
        # Randomly zeros features during training for regularisation.
        # shape unchanged: (batch, 768)
        cls_vector = self.dropout(cls_vector)

        # --- Scoring head ---
        # Linear projection from essay representation to 5 band scores.
        # shape: (batch, 768) → (batch, 5)
        scores = self.scoring_head(cls_vector.float())

        return scores

    def get_parameter_groups(self, backbone_lr: float = 2e-5, head_lr: float = 1e-3) -> list:
        """
        Returns parameter groups with different learning rates for train.py.

        Usage in train.py:
            optimizer = AdamW(model.get_parameter_groups(), weight_decay=0.01)

        Args:
            backbone_lr: Learning rate for DeBERTa parameters. Default 2e-5.
            head_lr:     Learning rate for scoring head parameters. Default 1e-3.

        Returns:
            List of dicts, one per parameter group.
        """
        return [
            {"params": self.deberta.parameters(),       "lr": backbone_lr},
            {"params": self.dropout.parameters(),       "lr": head_lr},
            {"params": self.scoring_head.parameters(),  "lr": head_lr},
        ]

    def freeze_backbone(self):
        """
        Freeze all DeBERTa parameters — only the scoring head will be trained.

        Useful for a warm-up phase: train the head for a few epochs first,
        then unfreeze the backbone for full fine-tuning.

        Call model.unfreeze_backbone() to reverse.
        """
        for param in self.deberta.parameters():
            param.requires_grad = False
        print("[model] Backbone frozen. Only scoring head will be trained.")

    def unfreeze_backbone(self):
        """
        Unfreeze all DeBERTa parameters for full fine-tuning.
        """
        for param in self.deberta.parameters():
            param.requires_grad = True
        print("[model] Backbone unfrozen. Full fine-tuning enabled.")

    def count_parameters(self) -> dict:
        """
        Returns trainable and total parameter counts.
        Useful for verifying freeze/unfreeze state.
        """
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {
            "total":     total,
            "trainable": trainable,
            "frozen":    frozen,
        }


# ------------------------------------------------------------------
# Smoke test — run directly to verify architecture
# python src/model.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("model.py smoke test")
    print("=" * 50)

    # Use pretrained=False to skip the 900MB download during testing.
    # The architecture is identical — only weights differ.
    print("\nBuilding model (random weights, no download)...")
    model = BandItScorer(pretrained=False)
    print("Model built.")

    # 1. Parameter count
    counts = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  total:     {counts['total']:,}")
    print(f"  trainable: {counts['trainable']:,}")
    print(f"  frozen:    {counts['frozen']:,}")

    # 2. Forward pass with dummy input
    # Simulates one batch of 4 essays, 512 tokens each
    print("\nRunning forward pass (batch=4, seq_len=512)...")
    batch_size = 4
    seq_len    = 512

    dummy_input_ids      = torch.randint(0, 1000, (batch_size, seq_len))  # (4, 512)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)  # (4, 512)

    with torch.no_grad():
        scores = model(dummy_input_ids, dummy_attention_mask)

    print(f"  input_ids shape:      {dummy_input_ids.shape}")
    print(f"  attention_mask shape: {dummy_attention_mask.shape}")
    print(f"  output scores shape:  {scores.shape}")
    assert scores.shape == (batch_size, NUM_LABELS), \
        f"Expected ({batch_size}, {NUM_LABELS}), got {scores.shape}"
    print(f"  output shape correct: ({batch_size}, {NUM_LABELS}) ✓")

    # 3. Verify output is raw scores (no sigmoid/softmax clamping)
    print(f"\n  sample output (random weights, not meaningful):")
    print(f"  {scores[0].tolist()}")

    # 4. Test freeze / unfreeze
    print("\nTesting freeze_backbone()...")
    model.freeze_backbone()
    counts_frozen = model.count_parameters()
    print(f"  trainable after freeze: {counts_frozen['trainable']:,}  (should be ~3,845)")

    print("\nTesting unfreeze_backbone()...")
    model.unfreeze_backbone()
    counts_unfrozen = model.count_parameters()
    print(f"  trainable after unfreeze: {counts_unfrozen['trainable']:,}  (should match total)")

    # 5. Test parameter groups
    print("\nTesting get_parameter_groups()...")
    groups = model.get_parameter_groups()
    print(f"  number of groups: {len(groups)}")
    for i, g in enumerate(groups):
        n_params = sum(p.numel() for p in g["params"])
        print(f"  group {i}: lr={g['lr']}, params={n_params:,}")

    print("\n" + "=" * 50)
    print("All checks passed. model.py is ready.")
    print("=" * 50)