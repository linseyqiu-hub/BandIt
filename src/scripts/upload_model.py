"""
One-off: upload a trained checkpoint to the HuggingFace model repo and tag it.

Run from the repo root:
    python scripts/upload_model.py --checkpoint checkpoints/best_model_v5.pt --tag v5

This is a DEVELOPER script. It never runs inside the container — the
container only ever downloads. HF_TOKEN needs `write` scope here; the
container needs only `read`.

Why a constant filename (model.pt) instead of best_model_v5.pt:
the version lives in the git TAG, not the filename. That way shipping a
new model is one env var change (MODEL_VERSION=v6) and nothing else.
"""

import argparse
import os
import sys

from huggingface_hub import HfApi

REPO_ID     = os.environ.get("MODEL_REPO", "burgerFlipperF/bandit-scorer")
REPO_FILE   = "model.pt"          # constant on purpose — see docstring


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="local .pt file to upload")
    parser.add_argument("--tag", required=True,
                        help="version tag, e.g. v5 — this becomes MODEL_VERSION")
    parser.add_argument("--repo", default=REPO_ID)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("error: HF_TOKEN not set (needs `write` scope)", file=sys.stderr)
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"error: no such file: {args.checkpoint}", file=sys.stderr)
        return 1

    size_mb = os.path.getsize(args.checkpoint) / 1e6
    print(f"[upload] {args.checkpoint} ({size_mb:.0f} MB) "
          f"-> {args.repo}:{REPO_FILE} @ {args.tag}")

    api = HfApi(token=token)

    # Large files are routed through Git LFS automatically — no setup needed.
    api.upload_file(
        path_or_fileobj = args.checkpoint,
        path_in_repo    = REPO_FILE,
        repo_id         = args.repo,
        repo_type       = "model",
        commit_message  = f"upload {args.tag} ({os.path.basename(args.checkpoint)})",
    )
    print("[upload] file uploaded ✓")

    # The tag is the registry pointer: an immutable reference to this commit.
    # Without it we would be pinning to `main`, which moves under us.
    api.create_tag(
        repo_id = args.repo,
        tag     = args.tag,
        repo_type = "model",
    )
    print(f"[upload] tagged {args.tag} ✓")
    print(f"\nSet MODEL_VERSION={args.tag} in .env to serve this model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())