import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

from core.config import MODEL_NAME, LABEL_COLUMNS


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

HIDDEN_SIZE = 768   # DeBERTa-v3-base hidden dim (12 heads × 64)
NUM_LABELS  = len(LABEL_COLUMNS)  # 4


# ------------------------------------------------------------------
# Cross-attention module
# ------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Single-head cross-attention for combining question and essay vectors.

    Used exclusively by the TR head:
        query  = q_vec  (what the question is asking)
        key    = e_vec  (what the essay contains)
        value  = e_vec

    Output: a 768-dim vector representing "essay meaning as seen through
    the lens of the question" — exactly what TR needs.

    This is a lightweight module (~1.8M params total across Q/K/V projections).
    With frozen backbone it is the most expressive part of the trainable model.
    """

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.hidden_size = hidden_size

        # Linear projections for Q, K, V — standard attention setup
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        # Output projection — maps attended vector back to hidden_size
        self.W_o = nn.Linear(hidden_size, hidden_size)

        # Scale factor — prevents dot products from growing too large
        # standard: 1 / sqrt(d_k)
        self.scale = hidden_size ** -0.5

        # Small init to avoid large attention weights at start of training
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, q_vec: torch.Tensor, e_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q_vec: (batch, 768) — CLS from question-only encoding
            e_vec: (batch, 768) — CLS from essay-only encoding

        Returns:
            (batch, 768) — attended vector for TR head input
        """
        # Project to Q, K, V spaces
        Q = self.W_q(q_vec)   # (batch, 768)
        K = self.W_k(e_vec)   # (batch, 768)
        V = self.W_v(e_vec)   # (batch, 768)

        # Scaled dot-product attention
        # Q · K^T gives a scalar per sample — how much the question
        # aligns with the essay at the CLS summary level
        # unsqueeze/squeeze to allow bmm: (batch, 1, 768) × (batch, 768, 1)
        attn_score = torch.bmm(
            Q.unsqueeze(1),          # (batch, 1, 768)
            K.unsqueeze(2),          # (batch, 768, 1)
        ).squeeze(-1) * self.scale   # (batch, 1)

        attn_weight = torch.sigmoid(attn_score)   # (batch, 1) — soft gate

        # Weighted value: how much of the essay to "let through"
        # based on question-essay alignment
        attended = attn_weight * V   # (batch, 768) — broadcast

        # Output projection
        output = self.W_o(attended)  # (batch, 768)

        return output


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

class BandItScorer(nn.Module):
    """
    IELTS essay scoring model — v4 architecture.

    Architecture:
        Pass 1: question only  →  DeBERTa  →  q_vec (768)  ┐
                                                             ├→ CrossAttention → tr_vec → TR head  → TR score
        Pass 2: essay only     →  DeBERTa  →  e_vec (768)  ┘
                                                             ├→ CC head → CC score
                                                             ├→ LR head → LR score
                                                             └→ RA head → RA score

    Key design decisions:
        - Backbone permanently frozen — only heads + cross-attention train.
          With 1434 essays, frozen backbone prevents overfitting and makes
          training ~50x faster (no backprop through 86M params).
        - Two forward passes, shared backbone — same DeBERTa weights called
          twice with different inputs. Not separate models — one model, two calls.
        - TR gets cross-attention(q_vec, e_vec) — question-aware representation.
          Task Response is literally "did the essay answer the question."
        - CC, LR, RA get e_vec directly — genuinely question-free.
          Coherence, lexical resource, grammatical range don't depend on the question.
        - Four independent Linear(768→1) heads — each criterion learns its own
          projection instead of sharing one weight matrix.
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()

        # --- Shared backbone ---
        # One DeBERTa instance, called twice per forward pass.
        # Permanently frozen — weights never update.
        if pretrained:
            self.deberta = DebertaV2Model.from_pretrained(MODEL_NAME)
        else:
            config = DebertaV2Config.from_pretrained(MODEL_NAME)
            self.deberta = DebertaV2Model(config)

        # Freeze immediately — permanent, not a warm-up phase
        self._freeze_backbone()

        # --- Dropout ---
        # Applied to both q_vec and e_vec before heads
        self.dropout = nn.Dropout(dropout)

        # --- Cross-attention for TR ---
        # Combines q_vec (question) and e_vec (essay) into a TR-specific vector
        self.cross_attention = CrossAttention(HIDDEN_SIZE)

        # --- Four independent scoring heads ---
        # TR: takes cross-attended vector (768) → scalar
        # CC, LR, RA: take e_vec (768) → scalar each
        self.tr_head = nn.Linear(HIDDEN_SIZE, 1)
        self.cc_head = nn.Linear(HIDDEN_SIZE, 1)
        self.lr_head = nn.Linear(HIDDEN_SIZE, 1)
        self.ra_head = nn.Linear(HIDDEN_SIZE, 1)

        # Small init for all heads
        for head in [self.tr_head, self.cc_head, self.lr_head, self.ra_head]:
            nn.init.normal_(head.weight, mean=0.0, std=0.02)
            nn.init.zeros_(head.bias)

    def _freeze_backbone(self):
        """Permanently freezes all DeBERTa parameters."""
        for param in self.deberta.parameters():
            param.requires_grad = False

    def forward(
        self,
        q_input_ids:      torch.Tensor,   # (batch, 512) — question only
        q_attention_mask: torch.Tensor,   # (batch, 512)
        e_input_ids:      torch.Tensor,   # (batch, 512) — essay only
        e_attention_mask: torch.Tensor,   # (batch, 512)
    ) -> torch.Tensor:
        """
        Dual-pass forward: question encoding + essay encoding → 4 scores.

        Args:
            q_input_ids:      token IDs for question-only sequences
            q_attention_mask: attention mask for question sequences
            e_input_ids:      token IDs for essay-only sequences
            e_attention_mask: attention mask for essay sequences

        Returns:
            scores: FloatTensor (batch, 4)
                    order: [Task_Response, Coherence_Cohesion, Lexical_Resource, Range_Accuracy]
        """

        # --- Pass 1: question only ---
        # No gradient needed — backbone is frozen
        with torch.no_grad():
            q_outputs = self.deberta(
                input_ids=q_input_ids,
                attention_mask=q_attention_mask,
            )
        q_vec = q_outputs.last_hidden_state[:, 0, :]   # CLS (batch, 768)
        q_vec = self.dropout(q_vec.float())

        # --- Pass 2: essay only ---
        with torch.no_grad():
            e_outputs = self.deberta(
                input_ids=e_input_ids,
                attention_mask=e_attention_mask,
            )
        e_vec = e_outputs.last_hidden_state[:, 0, :]   # CLS (batch, 768)
        e_vec = self.dropout(e_vec.float())

        # --- TR: cross-attention(q_vec, e_vec) → head ---
        tr_vec = self.cross_attention(q_vec, e_vec)    # (batch, 768)
        tr_score = self.tr_head(tr_vec)                # (batch, 1)

        # --- CC, LR, RA: e_vec → heads directly ---
        cc_score = self.cc_head(e_vec)                 # (batch, 1)
        lr_score = self.lr_head(e_vec)                 # (batch, 1)
        ra_score = self.ra_head(e_vec)                 # (batch, 1)

        # Concatenate into (batch, 4)
        # Order matches LABEL_COLUMNS: [TR, CC, LR, RA]
        scores = torch.cat([tr_score, cc_score, lr_score, ra_score], dim=1)

        return scores

    def get_parameter_groups(self, head_lr: float = 3e-4) -> list:
        """
        Returns trainable parameter groups for the optimizer.

        Backbone is frozen — only cross_attention and four heads are included.
        Single lr since all trainable params are heads (no backbone to nudge gently).
        """
        trainable = (
            list(self.cross_attention.parameters()) +
            list(self.tr_head.parameters()) +
            list(self.cc_head.parameters()) +
            list(self.lr_head.parameters()) +
            list(self.ra_head.parameters())
        )
        return [{"params": trainable, "lr": head_lr}]

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen    = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}


# ------------------------------------------------------------------
# Smoke test
# python src/model.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("model.py smoke test — v4 architecture")
    print("=" * 50)

    print("\nBuilding model (random weights, no download)...")
    model = BandItScorer(pretrained=False, dropout=0.4)
    print("Model built.")

    counts = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  total:     {counts['total']:,}")
    print(f"  trainable: {counts['trainable']:,}  ← cross_attention + 4 heads only")
    print(f"  frozen:    {counts['frozen']:,}  ← DeBERTa backbone")

    batch_size = 4
    seq_len    = 512

    # Two separate inputs — question only and essay only
    q_input_ids      = torch.randint(0, 1000, (batch_size, seq_len))
    q_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    e_input_ids      = torch.randint(0, 1000, (batch_size, seq_len))
    e_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    print(f"\nRunning forward pass (batch={batch_size}, seq_len={seq_len})...")
    with torch.no_grad():
        scores = model(q_input_ids, q_attention_mask, e_input_ids, e_attention_mask)

    print(f"  output shape: {scores.shape}  ← should be ({batch_size}, 4)")
    assert scores.shape == (batch_size, NUM_LABELS), \
        f"Expected ({batch_size}, {NUM_LABELS}), got {scores.shape}"
    print(f"  output shape correct ✓")
    print(f"  sample output: {scores[0].tolist()}")

    print("\nParameter groups:")
    groups = model.get_parameter_groups()
    for i, g in enumerate(groups):
        n = sum(p.numel() for p in g["params"])
        print(f"  group {i}: lr={g['lr']}, params={n:,}")

    print("\n" + "=" * 50)
    print("All checks passed. model.py v4 ready.")
    print("=" * 50)