# BandIt — how it runs today

Pre-Docker inventory. This is the ground truth the Dockerfile gets built from.
Written before containerizing; update it when anything below changes.

---

## Start commands

**API**

```bash
cd src
uvicorn main:app --reload
```

- Serves on `http://127.0.0.1:8000`
- `--reload` is dev-only; drop it in the container
- Working directory is `src/`, not the repo root — this matters, see below

**Frontend**

```bash
cd frontend
npm run dev
```

- Serves on `http://localhost:5173`
- Vite dev server proxies `/api/*` → `http://localhost:8000/` (see `vite.config.js`)
- The proxy is a dev-server feature only. It does not exist in the production build.

---

## Ports

| Port | Service |
|------|---------|
| 8000 | FastAPI (uvicorn) |
| 5173 | Vite dev server |

---

## Environment variables

**Secret — injected at runtime, never committed or baked into an image**

| Variable | Read by | Notes |
|----------|---------|-------|
| `ANTHROPIC_API_KEY` | `src/services/feedback.py:209` | Implicit — `Anthropic()` with no args reads it from the environment. Invisible to grep. Currently set in `~/.bashrc`. |
| `HF_TOKEN` | HF libraries | Not set yet. Silences the unauthenticated warning and raises rate limits. Needed in block 6 for pushing the checkpoint. |

**Config — plain text, safe to commit**

| Variable | Current state | Target |
|----------|---------------|--------|
| `CHECKPOINT_PATH` | hardcoded in `core/config.py` | env read, default to current path |
| `CHROMA_HOST` | does not exist | env read; empty = local `PersistentClient` |
| `MODEL_VERSION` | does not exist | added in block 6 |

**Deliberately dropped in the container**

- `KMP_DUPLICATE_LIB_OK=TRUE` — Windows/Anaconda MKL workaround. No duplicate OpenMP runtime on Linux with pip-installed torch.
- `GROQ_API_KEY`, `GEMINI_API_KEY` — eval-only (`eval/smoke_test_apis.py`), not in the request path. Eval scripts don't ship.

---

## Files read from disk

| Path | Source | Size |
|------|--------|------|
| `checkpoints/best_model_v5.pt` | trained locally | ~350MB |
| `data/chroma/` | built by `src/scripts/ingest.py` | 1430 records × 2 collections |

Both resolved in `core/config.py` from `_PROJECT_ROOT` — three `dirname` calls up
from `core/config.py`. Portable, but assumes `checkpoints/` sits next to `src/`.

---

## Pulled from Hugging Face at startup

Three artifacts, not two. Only the first is ours.

| Artifact | Purpose |
|----------|---------|
| `best_model_v5.pt` | the scorer — ours, on disk |
| `microsoft/deberta-v3-base` tokenizer | text → integer token IDs for the scorer. Must match training exactly. |
| `all-MiniLM-L6-v2` | essay → 384-dim vector for Chroma retrieval |

The two HF artifacts cache to `~/.cache/huggingface` (Windows:
`C:\Users\18030\.cache\huggingface`) on first download.

**Startup still hits the network even with a warm cache** — `from_pretrained` pings
the Hub to check whether the cached copy is stale. That's the source of the
"unauthenticated requests" warning. `HF_HUB_OFFLINE=1` skips the check entirely.

Container implication: a fresh container has an empty cache and downloads both from
scratch. Bake them in at build time, or mount a volume as the cache.

---

## Dependencies

Direct dependencies of the API only. Everything else in `pip freeze` is Anaconda's
stack or training/eval-only.

```
fastapi==0.136.3
uvicorn==0.48.0
pydantic==2.13.4
anthropic==0.104.0
torch==2.12.0
transformers==5.8.1
sentencepiece==0.2.1
sentence-transformers==5.6.0
chromadb==1.5.9
numpy==2.3.5
```

Transitive, no need to list: `tokenizers`, `safetensors`, `huggingface_hub`,
`starlette`, `scipy`, `scikit-learn`.

**torch needs the CPU index** or pip pulls ~2GB of unused CUDA libraries:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.12.0
```

**Never include**: `winloop`, `pywin32`, `PyQt5`, `PySide6` — Windows-only, will
fail to install on Linux.

**Excluded deliberately** (training/eval, not shipped): `datasets`, `matplotlib`,
`tiktoken`, `openai`, `google-genai`.

**Watch**: `numpy`, `protobuf`, `scipy`, `scikit-learn` were installed by conda, not
pip. Versions above were read out of the embedded wheel filenames. Pip on Linux will
resolve its own builds — most likely fine, but this is where a version conflict would
surface first. Second place to look: the `transformers 5.8.1` / `sentence-transformers 5.6.0`
pairing.

---

## Known issues to fix (not blocking containerization)

1. **`/health` checks the wrong attribute.** Lifespan sets
   `app.state.inference_engine`; `main.py` checks `app.state.engine`. Reports
   unhealthy forever. Matters because hosting platforms poll this endpoint to
   decide whether to route traffic.

2. **CORS config is invalid for production.** `allow_origins=["*"]` with
   `allow_credentials=True` is rejected by browsers per spec. Works today only
   because the Vite proxy makes requests same-origin. Breaks the moment the
   frontend is deployed separately. Fix: name the actual origins.

3. **Anthropic client constructed per request** (`feedback.py:209`, inside the
   function). Rebuilds the connection pool every call. Move to `app.state` in
   lifespan alongside the models.

4. **Missing key fails late.** `Anthropic()` raises at call time, not startup. A
   container with no key boots, passes health checks, scores essays, then dies on
   the first feedback request. Add a startup check that refuses to boot.

5. **`sys.path.insert` in `main.py`** patches the import path because the app runs
   from `src/`. In the container, `WORKDIR` handles this — the patch becomes
   unnecessary.

6. **Frontend needs `VITE_API_URL`.** Production build has no proxy; the API's real
   URL must be injected at build time.

---

## Model reference

`best_model_v5.pt` — epoch 27, val MAE 0.7407, trained on `ielts_relabeled_v3.csv`
(1430 essays). CPU inference.
