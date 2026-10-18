"""
core/settings.py — environment-dependent configuration, grouped by subsystem.

Contract:
    - Importing this module reads NOTHING. It only defines classes.
    - A group reads and validates its env vars when it is instantiated,
      e.g. ChromaSettings(). Entrypoints do that once, at startup.
    - Each entrypoint loads only the groups it needs:

          serving : ServingSettings.load()   (chroma, registry, llm, redis, api)
          ingest  : load_groups(ChromaSettings)
          eval    : load_groups(ChromaSettings, EvalLLMSettings)

Where values come from (highest priority first):
    1. Real environment variables  — what compose / ECS inject into a container
    2. env/<group>.env files        — read on the host; silently skipped if absent,
                                      so the same code works inside a container
    3. Field defaults below

Secrets are typed SecretStr: they print as '**********' in logs, reprs and
tracebacks. Call .get_secret_value() only at the point of use.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


# ── Env file locations ──────────────────────────────────────────────────
# Anchored to the repo root (this file is src/core/settings.py), NOT the
# current working directory — so host runs find env/ no matter where you
# launch from. In a container env/ does not exist and the files are skipped.
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_DIR = REPO_ROOT / "env"

CONFIG_ENV = ENV_DIR / "config.env"   # non-secret, shared by every group
HF_ENV = ENV_DIR / "hf.env"           # HF_TOKEN
LLM_ENV = ENV_DIR / "llm.env"         # ANTHROPIC_API_KEY
EVAL_ENV = ENV_DIR / "eval.env"       # GROQ_API_KEY, GEMINI_API_KEY


# Required values must be non-empty. Without this, a blank line like
# "ANTHROPIC_API_KEY=" in an env file yields "" — which passes a plain
# "required" check and only fails later as a confusing 401 at request time.
NonEmptySecret = Annotated[SecretStr, Field(min_length=1)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class _GroupSettings(BaseSettings):
    """Common behaviour for every settings group."""

    model_config = SettingsConfigDict(
        env_file=(CONFIG_ENV,),
        env_file_encoding="utf-8",
        case_sensitive=False,
        # A group only claims the vars it declares; anything else in an env
        # file or the environment belongs to another group and is ignored.
        extra="ignore",
    )


# ── Groups ──────────────────────────────────────────────────────────────

class ChromaSettings(_GroupSettings):
    """Vector store connection. Used by serving, ingest and eval."""

    model_config = SettingsConfigDict(env_prefix="CHROMA_")

    host: str = "localhost"   # CHROMA_HOST
    port: int = 8000          # CHROMA_PORT — "8000" is coerced to int, "abc" fails


class ModelRegistrySettings(_GroupSettings):
    """Where the scoring checkpoint comes from. Serving only."""

    model_config = SettingsConfigDict(env_file=(CONFIG_ENV, HF_ENV))

    model_repo: str = "burgerFlipperF/bandit-scorer"   # MODEL_REPO
    model_version: NonEmptyStr                         # MODEL_VERSION — required:
                                                       # a wrong default is worse than crashing
    hf_token: NonEmptySecret                           # HF_TOKEN — required: a cold start
                                                       # (empty cache) needs it for the private repo


class LLMSettings(_GroupSettings):
    """Production feedback LLM. Serving only."""

    model_config = SettingsConfigDict(env_file=(CONFIG_ENV, LLM_ENV))

    anthropic_api_key: NonEmptySecret           # ANTHROPIC_API_KEY — required
    llm_model: str = "claude-sonnet-4-6"        # LLM_MODEL
    llm_max_tokens: int = 350                   # LLM_MAX_TOKENS


class EvalLLMSettings(_GroupSettings):
    """Generator + judge models for the eval harness. Eval only — never serving."""

    model_config = SettingsConfigDict(env_file=(CONFIG_ENV, EVAL_ENV))

    groq_api_key: NonEmptySecret      # GROQ_API_KEY — required
    gemini_api_key: NonEmptySecret    # GEMINI_API_KEY — required


class RedisSettings(_GroupSettings):
    """Rate limiter backend. Serving only."""

    redis_url: str = "redis://localhost:6379/0"   # REDIS_URL


class ApiSettings(_GroupSettings):
    """HTTP-layer behaviour. Serving only."""

    # NoDecode: by default pydantic-settings parses list fields as JSON, so
    # "a,b" would fail. We take the raw string and split it ourselves.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_comma_list(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


# ── Loading ─────────────────────────────────────────────────────────────

class SettingsError(RuntimeError):
    """Raised when one or more settings groups fail validation."""


_EMPTY_ERRORS = {"too_short", "string_too_short"}


def _env_var_name(group: type[BaseSettings], field: str) -> str:
    prefix = group.model_config.get("env_prefix", "")
    return f"{prefix}{field}".upper()


def load_groups(*groups: type[BaseSettings]) -> tuple[BaseSettings, ...]:
    """
    Instantiate each group, collecting EVERY validation error before failing,
    so one crash reports all missing/invalid vars instead of one per restart.

    The error message names env vars only — never their values. (pydantic's
    own error text includes the offending input, which could be a secret.)
    """
    loaded, problems = [], []

    for group in groups:
        try:
            loaded.append(group())
        except ValidationError as exc:
            for err in exc.errors(include_input=False, include_url=False):
                field = str(err["loc"][0]) if err["loc"] else "?"
                msg = "is set but empty" if err["type"] in _EMPTY_ERRORS else err["msg"]
                problems.append(
                    f"  {group.__name__}: {_env_var_name(group, field)} — {msg}"
                )

    if problems:
        raise SettingsError(
            "invalid configuration — fix these env vars and restart:\n"
            + "\n".join(problems)
        )
    return tuple(loaded)


@dataclass(frozen=True)
class ServingSettings:
    """Typed bundle for the api process, so code can write settings.llm.llm_model."""

    chroma: ChromaSettings
    registry: ModelRegistrySettings
    llm: LLMSettings
    redis: RedisSettings
    api: ApiSettings

    @classmethod
    def load(cls) -> "ServingSettings":
        return cls(*load_groups(
            ChromaSettings,
            ModelRegistrySettings,
            LLMSettings,
            RedisSettings,
            ApiSettings,
        ))