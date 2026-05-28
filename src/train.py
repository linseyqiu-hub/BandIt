import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import get_dataloaders, LABEL_COLUMNS
from model import BandItScorer


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
# All hyperparameters in one place.
# Change values here — never hardcode them inside functions.

CONFIG = {
    # data
    "csv_path":       "data/ielts_labeled.csv",
    "max_length":     512,
    "val_split":      0.2,
    "seed":           42,

    # training
    "epochs":         20,          # more epochs since we're starting frozen
    "batch_size":     8,
    "warmup_epochs":  3,           # was 1 — slower ramp

    # optimizer
    "backbone_lr":    2e-6,        # was 2e-5 — much gentler
    "head_lr":        3e-4,        # was 1e-3 — less aggressive
    "weight_decay":   0.05,

    # model
    "dropout":        0.4,         # was 0.1 — stronger regularisation

    # freezing strategy
    "freeze_epochs":  5,           # train head-only for first N epochs

    # checkpointing
    "checkpoint_dir": "checkpoints",
    "save_every":     2,
}


# ------------------------------------------------------------------
# Loss and metric
# ------------------------------------------------------------------

def mse_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error loss over all 5 criteria simultaneously.

    predictions: (batch, 5) — model output
    labels:      (batch, 5) — ground truth band scores

    MSE = mean((pred - truth)^2) across all elements in the batch.
    Differentiable — gradients flow back through this to the model.
    """
    return nn.MSELoss()(predictions, labels)


def mae_metric(predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """
    Mean Absolute Error — human-readable metric in band units.

    MAE = mean(|pred - truth|)

    We report:
      - mean_mae: single headline number, average across all 5 criteria combined
      - per-criterion MAE so we can see which criteria the model struggles with

    Naming:
      "mean_mae"        → average across ALL criteria AND all essays (our summary)
      "Overall"         → MAE for the Overall band score column specifically
      (these are different things — mean_mae summarises all 5, Overall is just 1)

    Not used for backprop — only for logging and model selection.
    predictions and labels are detached from the computation graph here.
    """
    with torch.no_grad():
        abs_errors = (predictions - labels).abs()   # (batch, 5)

        # per-criterion MAE — mean over batch dimension
        per_criterion = abs_errors.mean(dim=0)      # (5,)

        # mean_mae — single number summarising all criteria and all essays
        mean_mae = abs_errors.mean().item()

    result = {"mean_mae": mean_mae}
    for i, col in enumerate(LABEL_COLUMNS):
        result[col] = per_criterion[i].item()

    return result


# ------------------------------------------------------------------
# Training loop — one epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """
    Runs one full pass through the training set.

    For each batch:
        1. forward pass  → predictions (batch, 5)
        2. compute loss  → MSE scalar
        3. backward pass → gradients for every parameter
        4. clip gradients → prevent exploding gradients
        5. optimizer step → update weights
        6. scheduler step → update learning rate

    Returns average training loss for this epoch.
    """
    model.train()   # enables dropout, enables gradient computation

    total_loss  = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):

        # move tensors to device (CPU or GPU)
        input_ids      = batch["input_ids"].to(device)       # (batch, 512)
        attention_mask = batch["attention_mask"].to(device)  # (batch, 512)
        labels         = batch["labels"].to(device)          # (batch, 5)

        # --- forward pass ---
        # model maps tokenized essays → predicted band scores
        predictions = model(input_ids, attention_mask)       # (batch, 5)

        # --- loss ---
        # MSE between predicted and ground truth scores
        loss = mse_loss(predictions, labels)

        # --- backward pass ---
        # zero_grad first — PyTorch accumulates gradients by default
        # if you forget this, gradients from the previous batch
        # are added to this batch's gradients → wrong updates
        optimizer.zero_grad()
        loss.backward()

        # --- gradient clipping ---
        # caps gradient norm at 1.0
        # prevents a single bad batch from making catastrophically large updates
        # standard practice for transformer fine-tuning
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # --- optimizer step ---
        # uses gradients to update every parameter
        # applies correct lr to each parameter group (backbone vs head)
        optimizer.step()

        # --- scheduler step ---
        # updates the learning rate according to warmup + decay schedule
        # called once per batch, not once per epoch
        scheduler.step()

        total_loss  += loss.item()
        num_batches += 1

        # print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"  epoch {epoch} | batch {batch_idx + 1}/{len(loader)} | loss {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    return avg_loss


# ------------------------------------------------------------------
# Validation loop — one epoch
# ------------------------------------------------------------------

def validate(model, loader, device):
    """
    Runs one full pass through the validation set.

    No gradient computation — we are only measuring performance.
    model.eval() disables dropout so predictions are deterministic.

    Returns:
        avg_loss: average MSE loss on validation set
        mae:      dict of MAE scores per criterion + overall
    """
    model.eval()    # disables dropout, disables gradient tracking

    total_loss   = 0.0
    num_batches  = 0
    all_preds    = []
    all_labels   = []

    with torch.no_grad():   # no gradient computation — saves memory and time
        for batch in loader:

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            predictions = model(input_ids, attention_mask)  # (batch, 5)
            loss        = mse_loss(predictions, labels)

            total_loss  += loss.item()
            num_batches += 1

            # collect all predictions and labels for MAE computation
            all_preds.append(predictions)
            all_labels.append(labels)

    # concatenate all batches into one tensor for MAE computation
    all_preds  = torch.cat(all_preds,  dim=0)   # (val_size, 5)
    all_labels = torch.cat(all_labels, dim=0)   # (val_size, 5)

    avg_loss = total_loss / num_batches
    mae      = mae_metric(all_preds, all_labels)

    return avg_loss, mae


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def save_checkpoint(model, optimizer, scheduler, epoch, val_mae, path):
    """
    Saves model weights + training state to disk.

    Saves everything needed to resume training or run inference:
        - model state dict  (the weights)
        - optimizer state   (Adam moment estimates — needed to resume training)
        - scheduler state   (current lr position — needed to resume training)
        - epoch number
        - validation MAE    (so we know how good this checkpoint is)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":            epoch,
        "val_mae":          val_mae,
        "model_state":      model.state_dict(),
        "optimizer_state":  optimizer.state_dict(),
        "scheduler_state":  scheduler.state_dict(),
        "config":           CONFIG,
        "label_columns":    LABEL_COLUMNS,
    }, path)
    print(f"  [checkpoint] saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """
    Loads a checkpoint from disk.

    If optimizer and scheduler are provided, restores their state too
    (use this to resume training from a specific epoch).

    If only model is provided, just loads weights
    (use this for inference).

    Returns the epoch number and val MAE stored in the checkpoint.
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    epoch   = checkpoint["epoch"]
    val_mae = checkpoint["val_mae"]
    print(f"  [checkpoint] loaded ← {path}  (epoch {epoch}, val MAE {val_mae:.4f})")
    return epoch, val_mae


# ------------------------------------------------------------------
# Main training function
# ------------------------------------------------------------------

def train():

    # --- device ---
    # use GPU if available, otherwise CPU
    # for BandIt on a laptop this will be CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[train] device: {device}")

    # --- data ---
    print("[train] loading data...")
    train_loader, val_loader = get_dataloaders(
        csv_path   = CONFIG["csv_path"],
        batch_size = CONFIG["batch_size"],
        val_split  = CONFIG["val_split"],
        max_length = CONFIG["max_length"],
        seed       = CONFIG["seed"],
    )
    print(f"[train] train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    # --- model ---
    print("[train] building model...")
    model = BandItScorer(pretrained=True, dropout=CONFIG["dropout"])
    model = model.float()
    model.to(device)
    counts = model.count_parameters()
    print(f"[train] parameters: {counts['total']:,} total | {counts['trainable']:,} trainable")

    # --- optimizer ---
    # AdamW with two parameter groups — different lr for backbone vs head
    # weight_decay applies L2 regularisation to prevent overfitting
    optimizer = AdamW(
        model.get_parameter_groups(
            backbone_lr = CONFIG["backbone_lr"],
            head_lr     = CONFIG["head_lr"],
        ),
        weight_decay = CONFIG["weight_decay"],
    )

    # --- scheduler ---
    # linear warmup + linear decay
    # total_steps = number of optimizer steps across all epochs
    # warmup_steps = steps during which lr ramps from 0 to target lr
    total_steps  = len(train_loader) * CONFIG["epochs"]
    warmup_steps = len(train_loader) * CONFIG["warmup_epochs"]
    scheduler    = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
    )
    print(f"[train] total steps: {total_steps} | warmup steps: {warmup_steps}")

    # --- resume ---
    # look for the latest epoch_XX.pt checkpoint in checkpoint_dir
    # if found, restore model + optimizer + scheduler state and resume from there
    # if not found, start from scratch
    start_epoch  = 1
    best_val_mae = float("inf")
    best_ckpt    = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
    history      = []

    def find_latest_checkpoint(ckpt_dir):
        """Scans checkpoint_dir for epoch_XX.pt files, returns path of latest."""
        if not os.path.exists(ckpt_dir):
            return None
        candidates = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("epoch_") and f.endswith(".pt")
        ]
        if not candidates:
            return None
        # sort by epoch number embedded in filename: epoch_07.pt → 7
        candidates.sort(key=lambda f: int(f.replace("epoch_", "").replace(".pt", "")))
        return os.path.join(ckpt_dir, candidates[-1])

    latest_ckpt = find_latest_checkpoint(CONFIG["checkpoint_dir"])

    if latest_ckpt:
        print(f"[train] found checkpoint: {latest_ckpt}")
        print(f"[train] resuming training from checkpoint...")
        resumed_epoch, resumed_mae = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler
        )
        start_epoch  = resumed_epoch + 1   # resume from NEXT epoch
        best_val_mae = resumed_mae

        # also restore best_val_mae from best_model.pt if it exists
        # (it may be better than the latest epoch checkpoint)
        if os.path.exists(best_ckpt):
            best_ckpt_data = torch.load(best_ckpt, map_location="cpu")
            best_val_mae   = min(best_val_mae, best_ckpt_data["val_mae"])
            print(f"[train] best val MAE so far: {best_val_mae:.4f}")

        if start_epoch > CONFIG["epochs"]:
            print(f"[train] already completed {CONFIG['epochs']} epochs. nothing to do.")
            return history
    else:
        print(f"[train] no checkpoint found — starting from scratch")

    print(f"[train] starting from epoch {start_epoch}/{CONFIG['epochs']}\n")

    for epoch in range(start_epoch, CONFIG["epochs"] + 1):

    # --- gradual unfreezing ---
    # epochs 1-3: backbone frozen, only scoring head trains (3,845 params)
    # epoch 4+:   backbone unfrozen, full fine-tuning resumes
        if epoch == 1:
            model.freeze_backbone()
            print("[train] backbone frozen — training scoring head only")
        elif epoch == CONFIG["freeze_epochs"] + 1:
            model.unfreeze_backbone()
            print("[train] backbone unfrozen — full fine-tuning")
        print(f"{'='*50}")
        print(f"epoch {epoch}/{CONFIG['epochs']}")
        print(f"{'='*50}")

        # --- train ---
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )

        # --- validate ---
        val_loss, val_mae = validate(model, val_loader, device)

        # --- log ---
        print(f"\n  train loss (MSE):  {train_loss:.4f}")
        print(f"  val   loss (MSE):  {val_loss:.4f}")
        print(f"  val MAE (mean):    {val_mae['mean_mae']:.4f} bands  ← avg across all 5 criteria")
        print(f"  val MAE per criterion:")
        for col in LABEL_COLUMNS:
            print(f"    {col:<25} {val_mae[col]:.4f}")

        # --- checkpoint: save every N epochs ---
        if epoch % CONFIG["save_every"] == 0:
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch:02d}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_mae["mean_mae"], ckpt_path)

        # --- checkpoint: save best model ---
        if val_mae["mean_mae"] < best_val_mae:
            best_val_mae = val_mae["mean_mae"]
            save_checkpoint(model, optimizer, scheduler, epoch, val_mae["mean_mae"], best_ckpt)
            print(f"  *** new best model — mean MAE {best_val_mae:.4f} ***")

        # --- history ---
        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_mae":    val_mae,
        })

        print()

    # --- final summary ---
    print(f"\n{'='*50}")
    print(f"training complete")
    print(f"best val MAE: {best_val_mae:.4f} bands")
    print(f"best model saved to: {best_ckpt}")
    print(f"{'='*50}\n")
    summary_path = os.path.join(CONFIG["checkpoint_dir"], "training_summary.txt")
    save_summary(history, summary_path)
    return history
def save_summary(history, path):
    """
    Saves training history to a human-readable txt file.
    Call once at the end of train().
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("BANDIT TRAINING SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write("CONFIG\n")
        f.write("-" * 30 + "\n")
        for k, v in CONFIG.items():
            f.write(f"  {k:<20} {v}\n")
        f.write("\n")

        f.write("EPOCH RESULTS\n")
        f.write("-" * 30 + "\n")
        f.write(f"  {'epoch':<8} {'train_loss':<14} {'val_loss':<14} {'mean_mae':<12} {'Overall':<12} {'Task_Resp':<12} {'Coh_Coh':<12} {'Lex_Res':<12} {'Rng_Acc':<12}\n")
        f.write("  " + "-" * 104 + "\n")
        for h in history:
            mae = h["val_mae"]
            f.write(
                f"  {h['epoch']:<8}"
                f"{h['train_loss']:<14.4f}"
                f"{h['val_loss']:<14.4f}"
                f"{mae['mean_mae']:<12.4f}"
                f"{mae['Overall']:<12.4f}"
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

# ------------------------------------------------------------------
# Entry point
# python src/train.py
# ------------------------------------------------------------------

if __name__ == "__main__":
    history = train()