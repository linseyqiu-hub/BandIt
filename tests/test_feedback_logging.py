"""
Guards the truncation WARNING in services/feedback.py.

Runs generate_feedback() against FAKES (embedder, Chroma, Claude client):
no network, no models, no API cost. pytest's `caplog` adds its own handler
to the root logger for the duration of the test and collects every record,
so we can assert on what was logged.
"""

import logging
from types import SimpleNamespace

from schemas.responses import Scores
from services.feedback import generate_feedback


class FakeVectors:
    def __init__(self, n: int):
        self._n = n

    def tolist(self):
        return [[0.0] * 8 for _ in range(self._n)]


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        return FakeVectors(len(texts))


class FakeCollection:
    def query(self, **kwargs):
        return {
            "ids":       [["ex1"]],
            "documents": [["An examiner comment."]],
            "metadatas": [[{"overall": 6.0}]],
        }


class FakeChroma:
    def get_collection(self, name, embedding_function=None):
        return FakeCollection()


class FakeMessages:
    def __init__(self, stop_reason: str):
        self._stop_reason = stop_reason

    def create(self, **kwargs):
        return SimpleNamespace(
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=kwargs["max_tokens"]),
            content=[SimpleNamespace(text="word " * 50)],
        )


def make_app_state(stop_reason: str):
    return SimpleNamespace(
        embedding_model=FakeEmbedder(),
        chroma_client=FakeChroma(),
        llm_client=SimpleNamespace(messages=FakeMessages(stop_reason)),
        settings=SimpleNamespace(llm=SimpleNamespace(llm_model="fake-model", llm_max_tokens=100)),
    )


SCORES = Scores(
    task_response=6.0, coherence_cohesion=6.0, lexical_resource=6.0,
    grammatical_range_accuracy=6.0, overall=6.0,
)


def _truncation_warnings(caplog):
    return [
        r for r in caplog.records
        if r.name == "services.feedback"
        and r.levelno == logging.WARNING
        and "feedback truncated" in r.getMessage()
    ]


def test_truncation_is_logged_as_warning(caplog):
    caplog.set_level(logging.INFO)
    generate_feedback("A question?", "An essay.", SCORES, "coaching", make_app_state("max_tokens"))
    assert len(_truncation_warnings(caplog)) == 1


def test_no_truncation_warning_on_normal_finish(caplog):
    caplog.set_level(logging.INFO)
    generate_feedback("A question?", "An essay.", SCORES, "coaching", make_app_state("end_turn"))
    assert _truncation_warnings(caplog) == []
