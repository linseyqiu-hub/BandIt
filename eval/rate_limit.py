"""
eval/rate_limit.py

Wraps judge_gemini / generate_groq calls with:
  1. A sliding-window limiter (proactive -- paces calls so you don't hit
     429s in the first place). Tracks RPM always, and optionally TPM too
     in the same rolling window.
  2. Exponential backoff with jitter on 429 (reactive -- handles the ones
     that get through anyway).

REAL numbers confirmed from AI Studio / GroqCloud consoles (free tier):
  Gemini 2.5 Flash: RPM=5, TPM=250,000
      -> RPM is the binding constraint by a huge margin (TPM converts to
         ~446 calls/min at this project's actual prompt sizes -- 90x
         headroom over RPM). TPM tracking is unnecessary here.
  Groq llama-3.3-70b-versatile: RPM=30, TPM=12,000
      -> TPM is the binding constraint (converts to ~12 calls/min,
         BELOW the RPM=30 ceiling). An RPM-only limiter would silently
         under-protect this side and let TPM 429s through anyway.

Usage in main_loop.py:
    from rate_limit import RateLimiter, with_backoff

    gemini_limiter = RateLimiter(rpm=5)                # TPM untracked -- not the binding constraint here
    groq_limiter = RateLimiter(rpm=30, tpm=12_000)      # TPM tracked -- IS the binding constraint here

    def judge_gemini_safe(prompt):
        gemini_limiter.wait()   # no token estimate needed -- RPM-only mode
        return with_backoff(judge_gemini, prompt)

    def generate_groq_safe(prompt_tuple):
        system, user = prompt_tuple
        estimated_tokens = (len(system) + len(user)) // 4 + 350  # rough input + MAX_TOKENS output ceiling
        groq_limiter.wait(estimated_tokens=estimated_tokens)
        return with_backoff(generate_groq, prompt_tuple)

Then swap judge_gemini -> judge_gemini_safe and generate_groq ->
generate_groq_safe everywhere in run() / score_faithfulness.
"""

import random
import time
from collections import deque


class RateLimiter:
    """
    Sliding-window limiter. Always tracks RPM (request count). Optionally
    also tracks TPM (token volume) in the same rolling 60s window, if
    `tpm` is provided -- pass tpm=None (default) for an RPM-only limiter.

    Call .wait() immediately before each API call. If tracking TPM, pass
    an estimated token cost for the upcoming call via
    wait(estimated_tokens=...) -- this has to be an ESTIMATE made before
    the call, since you don't know the real output size until after.
    Slightly over-estimating is safer than under-estimating (worst case
    you pace a bit more conservatively than strictly necessary; under-
    estimating risks a TPM 429 slipping through anyway).
    """

    def __init__(self, rpm, tpm=None):
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        if tpm is not None and tpm <= 0:
            raise ValueError("tpm must be positive if provided")
        self.rpm = rpm
        self.tpm = tpm
        self._call_times = deque()      # timestamps only -- RPM tracking
        self._token_events = deque()    # (timestamp, token_count) -- TPM tracking

    def wait(self, estimated_tokens=0):
        while True:
            now = time.monotonic()
            window_start = now - 60.0

            while self._call_times and self._call_times[0] < window_start:
                self._call_times.popleft()
            while self._token_events and self._token_events[0][0] < window_start:
                self._token_events.popleft()

            rpm_ok = len(self._call_times) < self.rpm

            tpm_ok = True
            if self.tpm is not None:
                current_tokens = sum(t for _, t in self._token_events)
                tpm_ok = (current_tokens + estimated_tokens) <= self.tpm

            if rpm_ok and tpm_ok:
                break

            # Not clear to proceed yet -- wait for whichever blocking
            # constraint's oldest entry ages out of the window first,
            # then re-check both from scratch (a single-shot wait can
            # under-wait for TPM if multiple prior token-events need to
            # expire before there's enough room).
            wait_candidates = []
            if not rpm_ok and self._call_times:
                wait_candidates.append(60.0 - (now - self._call_times[0]))
            if not tpm_ok and self._token_events:
                wait_candidates.append(60.0 - (now - self._token_events[0][0]))
            sleep_for = max(0.01, min(wait_candidates)) if wait_candidates else 0.5
            time.sleep(sleep_for)

        self._call_times.append(now)
        if self.tpm is not None:
            self._token_events.append((now, estimated_tokens))


def _looks_like_rate_limit_error(exc):
    """
    Best-effort detection across SDKs that don't share an exception
    hierarchy. Checks status_code/code attributes first (more reliable
    than string matching), falls back to searching the message for "429"
    or "rate limit". If your Groq/Gemini SDK exposes a specific exception
    class (e.g. groq.RateLimitError, google.api_core.exceptions.
    ResourceExhausted), prefer catching that directly instead of this
    heuristic -- swap it in where this function is called from
    with_backoff below.
    """
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "resource_exhausted" in msg or "resource exhausted" in msg


def with_backoff(fn, *args, max_retries=6, base_delay=2.0, max_delay=60.0, **kwargs):
    """
    Calls fn(*args, **kwargs), retrying with exponential backoff + jitter
    on rate-limit-like errors. Re-raises immediately on any error that
    doesn't look like a rate limit (e.g. auth failure, malformed request)
    -- retrying those just burns time and hides a real bug.

    After max_retries rate-limit failures in a row, re-raises the last
    exception rather than retrying forever -- the caller (main_loop.py's
    run()) should catch this, log it as a distinct failure mode from a
    parse failure, and either skip the case or abort, rather than silently
    treating an exhausted-retry case the same as a judge_parse failure.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _looks_like_rate_limit_error(exc):
                raise
            last_exc = exc
            delay = min(max_delay, base_delay * (2 ** attempt))
            delay += random.uniform(0, delay * 0.25)  # jitter, avoid thundering herd
            time.sleep(delay)

    raise last_exc