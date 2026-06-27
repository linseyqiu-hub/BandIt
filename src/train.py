import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from dataset import get_dataloaders, LABEL_COLUMNS
from model import BandItScorer


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

CONFIG = {
    # data
    "csv_path":        "data/ielts_relabeled_v3.csv",
    "max_length":      512,
    "val_split":       0.2,
    "seed":            42,

    # training
    "epochs":          30,
    "batch_size":      8,
    "warmup_epochs":   3,

    # optimizer
    "backbone_lr":     1e-6,
    "head_lr":         2e-4,
    "weight_decay":    0.05,

    # model
    "dropout":         0.4,

    # checkpointing
    "checkpoint_dir":  "checkpoints",
    "save_every":      2,
    "best_model_name": "best_model_v5.pt",
}


# ------------------------------------------------------------------
# Loss and metric
# ------------------------------------------------------------------

def mse_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(predictions, labels)


def mae_metric(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.no_grad():
        abs_errors    = (predictions - labels).abs()
        per_criterion = abs_errors.mean(dim=0)
        mean_mae      = abs_errors.mean().item()

    result = {"mean_mae": mean_mae}
    for i, col in enumerate(LABEL_COLUMNS):
        result[col] = per_criterion[i].item()
    return result


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()

    total_loss  = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        q_input_ids      = batch["q_input_ids"].to(device)
        q_attention_mask = batch["q_attention_mask"].to(device)
        e_input_ids      = batch["e_input_ids"].to(device)
        e_attention_mask = batch["e_attention_mask"].to(device)
        labels           = batch["labels"].to(device)

        predictions = model(
            q_input_ids, q_attention_mask,
            e_input_ids, e_attention_mask,
        )

        loss = mse_loss(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"  epoch {epoch} | batch {batch_idx+1}/{len(loader)} | loss {loss.item():.4f}")

    return total_loss / num_batches


# ------------------------------------------------------------------
# Validation loop
# ------------------------------------------------------------------

def validate(model, loader, device):
    model.eval()

    total_loss  = 0.0
    num_batches = 0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for batch in loader:
            q_input_ids      = batch["q_input_ids"].to(device)
            q_attention_mask = batch["q_attention_mask"].to(device)
            e_input_ids      = batch["e_input_ids"].to(device)
            e_attention_mask = batch["e_attention_mask"].to(device)
            labels           = batch["labels"].to(device)

            predictions = model(
                q_input_ids, q_attention_mask,
                e_input_ids, e_attention_mask,
            )
            loss = mse_loss(predictions, labels)

            total_loss  += loss.item()
            num_batches += 1
            all_preds.append(predictions)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    avg_loss = total_loss / num_batches
    mae      = mae_metric(all_preds, all_labels)
    return avg_loss, mae


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, val_mae, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":           epoch,
        "val_mae":         val_mae,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config":          CONFIG,
        "label_columns":   LABEL_COLUMNS,
    }, path)
    print(f"  [checkpoint] saved → {path}")


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    epoch   = checkpoint["epoch"]
    val_mae = checkpoint["val_mae"]
    print(f"  [checkpoint] loaded ← {path}  (epoch {epoch}, val MAE {val_mae:.4f})")
    return epoch, val_mae


def find_latest_checkpoint(ckpt_dir, prefix="v5_epoch_"):
    if not os.path.exists(ckpt_dir):
        return None
    candidates = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith(prefix) and f.endswith(".pt")
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda f: int(f.replace(prefix, "").replace(".pt", "")))
    return os.path.join(ckpt_dir, candidates[-1])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] device: {device}")

    print("[train] loading data...")
    train_loader, val_loader = get_dataloaders(
        csv_path   = CONFIG["csv_path"],
        batch_size = CONFIG["batch_size"],
        val_split  = CONFIG["val_split"],
        max_length = CONFIG["max_length"],
        seed       = CONFIG["seed"],
    )
    print(f"[train] train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    print("[train] building model...")
    model = BandItScorer(pretrained=True, dropout=CONFIG["dropout"])
    model = model.float()
    model.to(device)
    for param in model.deberta.parameters():
        param.requires_grad = True
    print("[train] backbone unfrozen from epoch 1")

    counts = model.count_parameters()
    print(f"[train] parameters: {counts['total']:,} total")
    print(f"[train]             {counts['trainable']:,} trainable")
    print(f"[train]             {counts['frozen']:,} frozen")

    # --- Optimizer ---
    optimizer = AdamW(
        [
            {"params": model.deberta.parameters(),      "lr": CONFIG["backbone_lr"]},
            {"params": list(model.cross_attention.parameters()) +
                       list(model.tr_head.parameters()) +
                       list(model.cc_head.parameters()) +
                       list(model.lr_head.parameters()) +
                       list(model.ra_head.parameters()), "lr": CONFIG["head_lr"]},
        ],
        weight_decay=CONFIG["weight_decay"],
    )

    # --- Scheduler — always built for full 50 epochs ---
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
        num_cycles         = 0.5,
    )
    print(f"[train] total steps: {total_steps} | warmup steps: {warmup_steps}")

    start_epoch  = 1
    best_val_mae = float("inf")
    best_ckpt    = os.path.join(CONFIG["checkpoint_dir"], CONFIG["best_model_name"])
    history      = []

    latest_ckpt = find_latest_checkpoint(CONFIG["checkpoint_dir"], prefix="v5_epoch_")

    if latest_ckpt:
        print(f"[train] found v5 checkpoint: {latest_ckpt}")
        resumed_epoch, resumed_mae = load_checkpoint(latest_ckpt, model)
        start_epoch  = resumed_epoch + 1
        best_val_mae = resumed_mae

        if os.path.exists(best_ckpt):
            best_data    = torch.load(best_ckpt, map_location="cpu")
            best_val_mae = min(best_val_mae, best_data["val_mae"])
            print(f"[train] best v5 val MAE so far: {best_val_mae:.4f}")

        # fast-forward scheduler to the correct position
        steps_done = resumed_epoch * len(train_loader)
        print(f"[train] fast-forwarding scheduler {steps_done} steps (epoch 1-{resumed_epoch})...")
        for _ in range(steps_done):
            scheduler.step()
        print(f"[train] scheduler ready at step {steps_done}/{total_steps}")

        if start_epoch > CONFIG["epochs"]:
            print(f"[train] already completed {CONFIG['epochs']} epochs.")
            return history
    else:
        print(f"[train] no v5 checkpoint found — starting from scratch")
        v3_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
        if os.path.exists(v3_path):
            print(f"[train] v3 best_model.pt preserved at {v3_path} ✓")

    print(f"[train] starting from epoch {start_epoch}/{CONFIG['epochs']}\n")

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):

        

        print(f"{'='*50}")
        print(f"epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*50}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_loss, val_mae = validate(model, val_loader, device)

        print(f"\n  train loss (MSE): {train_loss:.4f}")
        print(f"  val   loss (MSE): {val_loss:.4f}")
        print(f"  val MAE (mean):   {val_mae['mean_mae']:.4f} bands")
        print(f"  val MAE per criterion:")
        for col in LABEL_COLUMNS:
            print(f"    {col:<25} {val_mae[col]:.4f}")

        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"v5_epoch_{epoch:02d}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_mae["mean_mae"], ckpt_path)

        if val_mae["mean_mae"] < best_val_mae:
            best_val_mae = val_mae["mean_mae"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_mae["mean_mae"], best_ckpt)
            print(f"  *** new best — mean MAE {best_val_mae:.4f} → {best_ckpt} ***")

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mae":    val_mae,
        })
        print()

    print(f"\n{'='*50}")
    print(f"training complete")
    print(f"best val MAE: {best_val_mae:.4f} bands")
    print(f"best model:   {best_ckpt}")
    print(f"v3 model:     checkpoints/best_model.pt  (untouched)")
    print(f"{'='*50}\n")

    summary_path = os.path.join(CONFIG["checkpoint_dir"], "training_summary_v5.txt")
    save_summary(history, summary_path)
    return history


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

def save_summary(history, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("BANDIT TRAINING SUMMARY — v5\n")
        f.write("=" * 50 + "\n\n")

        f.write("CONFIG\n")
        f.write("-" * 30 + "\n")
        for k, v in CONFIG.items():
            f.write(f"  {k:<20} {v}\n")
        f.write("\n")

        f.write("EPOCH RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"  {'epoch':<8} {'train_loss':<14} {'val_loss':<14} {'mean_mae':<12} "
                f"{'Task_Resp':<12} {'Coh_Coh':<12} {'Lex_Res':<12} {'Rng_Acc':<12}\n")
        f.write("  " + "-" * 96 + "\n")
        for h in history:
            mae = h["val_mae"]
            f.write(
                f"  {h['epoch']:<8}"
                f"{h['train_loss']:<14.4f}"
                f"{h['val_loss']:<14.4f}"
                f"{mae['mean_mae']:<12.4f}"
                f"{mae['Task_Response']:<12.4f}"
                f"{mae['Coherence_Cohesion']:<12.4f}"
                f"{mae['Lexical_Resource']:<12.4f}"
                f"{mae['Range_Accuracy']:<12.4f}\n"
            )

        f.write("\n")
        f.write("BEST MODEL\n")
        f.write("-" * 30 + "\n")
        best = min(history, key=lambda h: h["val_mae"]["mean_mae"])
        f.write(f"  epoch:     {best['epoch']}\n")
        f.write(f"  mean MAE:  {best['val_mae']['mean_mae']:.4f} bands\n")
        f.write(f"  val loss:  {best['val_loss']:.4f}\n")

    print(f"[train] summary saved → {path}")


if __name__ == "__main__":
    history = train()