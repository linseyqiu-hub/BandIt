"""
eval/preflight.py

Run once, before run_ragas.py's main loop. Validates that every held-out test
row has held_out=true confirmed in BOTH the essays and questions ChromaDB
collections. Since ground truth is known here (every id in test_questions.json
must be held_out=true, by construction), any inconsistency found is safe to
auto-fix -- this is not a "which side do we trust" judgment call, just a
failed write from Pass 4 of build_test_set.py that needs redoing.

Anything that still fails after the fix attempt is genuinely anomalous and
gets logged for manual review; its id is returned so run_ragas.py can treat
it as a known-bad id during retrieval.

Usage:
    known_bad_ids = run_preflight(test_ids, essays_collection, questions_collection)
"""

from common import log_failure, log_fix, chroma_get_metadata, chroma_update_metadata


def run_preflight(all_held_out_ids, essays_collection, questions_collection):
    """
    all_held_out_ids: iterable of ids from test_questions.json
    essays_collection / questions_collection: real chromadb Collection
        objects (from build_test_set.py's get_collections())

    Returns: set of ids that remain inconsistent/bad after autofix attempt
             (should be empty or tiny in the common case).
    """
    known_bad_ids = set()
    n_already_ok = 0
    n_autofixed = 0
    n_autofix_failed = 0

    for row_id in all_held_out_ids:
        essays_meta = chroma_get_metadata(essays_collection, row_id)
        questions_meta = chroma_get_metadata(questions_collection, row_id)

        essays_flag = bool(essays_meta and essays_meta.get("held_out") is True)
        questions_flag = bool(questions_meta and questions_meta.get("held_out") is True)

        if essays_flag and questions_flag:
            n_already_ok += 1
            continue

        # Ground truth known: this id SHOULD be held_out=true in both.
        # chroma_update_metadata does a read-merge-write, so this won't
        # clobber band_bin/overall already stored on the record.
        if not essays_flag:
            chroma_update_metadata(essays_collection, row_id, held_out=True)
        if not questions_flag:
            chroma_update_metadata(questions_collection, row_id, held_out=True)

        essays_meta_after = chroma_get_metadata(essays_collection, row_id)
        questions_meta_after = chroma_get_metadata(questions_collection, row_id)
        essays_ok = bool(essays_meta_after and essays_meta_after.get("held_out") is True)
        questions_ok = bool(questions_meta_after and questions_meta_after.get("held_out") is True)

        if essays_ok and questions_ok:
            n_autofixed += 1
            log_fix(
                "held_out_preflight",
                row_id,
                "inconsistency detected and patched, confirmed on readback",
                {"essays_before": essays_flag, "questions_before": questions_flag},
            )
        else:
            n_autofix_failed += 1
            log_failure(
                "held_out_preflight",
                row_id,
                "autofix attempted but failed readback -- needs manual check",
                {
                    "essays_before": essays_flag,
                    "questions_before": questions_flag,
                    "essays_after": essays_ok,
                    "questions_after": questions_ok,
                },
            )
            known_bad_ids.add(row_id)

    print(f"[preflight] {n_already_ok} already consistent, "
          f"{n_autofixed} auto-fixed, {n_autofix_failed} still bad")
    if known_bad_ids:
        print(f"  -> {len(known_bad_ids)} ids need manual review: "
              f"see eval/logs/held_out_preflight_failures.jsonl")
        print(f"  -> these ids will be explicitly excluded during retrieval retries in run_ragas.py")

    return known_bad_ids