"""
eval/smoke_test_apis.py

Standalone connectivity check for Groq (generator) and Gemini (judge) before
running the full eval loop. Not part of the eval pipeline itself -- just a
quick "can I actually reach both APIs with my current keys" check.

Usage:
    python eval/smoke_test_apis.py

Exits 0 if both succeed, 1 if either fails (useful if you ever want to call
this from a Makefile/CI step as a pre-check before the real run).
"""

import os
import sys


def check_groq():
    print("[groq] checking connection...")
    try:
        from openai import OpenAI
    except ImportError:
        print("[groq] FAILED -- 'openai' package not installed. "
              "Run: pip install openai --break-system-packages")
        return False

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("[groq] FAILED -- GROQ_API_KEY not set in environment.")
        return False

    try:
        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Reply with exactly one word: hello"}],
            max_tokens=10,
        )
        text = resp.choices[0].message.content
        print(f"[groq] OK -- response: {text!r}")
        return True
    except Exception as e:
        print(f"[groq] FAILED -- {type(e).__name__}: {e}")
        return False


def check_gemini():
    print("[gemini] checking connection...")
    try:
        from google import genai
    except ImportError:
        print("[gemini] FAILED -- 'google-genai' package not installed. "
              "Run: pip install google-genai --break-system-packages")
        return False

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[gemini] FAILED -- GEMINI_API_KEY not set in environment.")
        return False

    try:
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Reply with exactly one word: hello",
        )
        print(f"[gemini] OK -- response: {resp.text!r}")
        return True
    except Exception as e:
        print(f"[gemini] FAILED -- {type(e).__name__}: {e}")
        return False


def main():
    groq_ok = check_groq()
    print()
    gemini_ok = check_gemini()
    print()
    print("=== summary ===")
    print(f"groq:   {'OK' if groq_ok else 'FAILED'}")
    print(f"gemini: {'OK' if gemini_ok else 'FAILED'}")

    if groq_ok and gemini_ok:
        print("\nBoth APIs reachable -- safe to run the full eval.")
        sys.exit(0)
    else:
        print("\nAt least one API check failed -- fix before running the full eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()