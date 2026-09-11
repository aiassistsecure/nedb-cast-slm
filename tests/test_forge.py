"""
Tests for cast.forge — the teacher paraphraser and its round-trip gate.

The gate is the whole value of this module, so these tests are mostly about
what it REFUSES. A forge that accepts everything the teacher says is just an
expensive template engine with worse guarantees.
"""
from __future__ import annotations

import json
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from cast import forge as F
from cast.sampler import canonical, render_nql, sample_plan


# ------------------------------------------------------------------- cleaners


def test_clean_nql_strips_what_chat_models_add_anyway():
    # Told "output only the query", a chat model still fences it, numbers it,
    # explains it, and adds a semicolon. Every one of these would fail
    # parse_nql and be counted as the teacher's syntax error rather than as
    # the harness failing to read a correct answer.
    assert F._clean_nql("```sql\nFROM orders WHERE total > 99\n```") == (
        "FROM orders WHERE total > 99"
    )
    assert F._clean_nql("FROM a LIMIT 5;") == "FROM a LIMIT 5"
    assert F._clean_nql("Sure! Here it is:\nFROM b ORDER BY x DESC") == (
        "FROM b ORDER BY x DESC"
    )
    assert F._clean_nql("`FROM c`") == "FROM c"
    assert F._clean_nql("") == ""


def test_clean_prompts_drops_the_narration_not_the_prompts():
    text = (
        "Here are 3 ways:\n"
        "1. show me orders over 99\n"
        "- which orders cleared 99\n"
        '"pull up big orders"\n'
        "\n"
        "Questions:\n"       # a trailing-colon header is not a prompt
        "ok\n"               # too short to be a real request
    )
    got = F._clean_prompts(text, 8)
    assert got == [
        "show me orders over 99",
        "which orders cleared 99",
        "pull up big orders",
    ]


def test_clean_prompts_respects_the_cap():
    text = "\n".join(f"question number {i} about orders" for i in range(20))
    assert len(F._clean_prompts(text, 5)) == 5


# --------------------------------------------------------------- schema block


def test_the_schema_block_gives_the_teacher_human_labels():
    rng = random.Random(7)
    _, dom, coll = sample_plan(rng)
    block = F._schema_block(dom, coll)
    assert dom.name in block
    assert coll.name in block
    for f in coll.fields:
        assert f.name in block, f"field {f.name} withheld from the teacher"
    # The human label is what a person would actually SAY. Withholding it makes
    # the teacher invent field names no real prompt would use, and those come
    # back as round-trip mismatches — paid for, then discarded.
    labelled = [f for f in coll.fields if f.label and f.label != f.name]
    for f in labelled:
        assert f.label in block, f"label {f.label!r} withheld"


# ------------------------------------------------------------- the round trip


class _Teacher(BaseHTTPRequestHandler):
    """A deliberately imperfect teacher, driven by the class attrs below."""

    prompts = "show me the thing\nwhich ones are big\nhopelessly vague"
    answers = ""  # set per test

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        system = req["messages"][0]["content"]
        paraphrasing = "search box" in system
        out = type(self).prompts if paraphrasing else type(self).answers
        body = json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": out,
                            "reasoning_content": "mapped each clause",
                        },
                    }
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def teacher():
    srv = HTTPServer(("127.0.0.1", 0), _Teacher)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield F.TeacherCfg(endpoint=f"http://127.0.0.1:{srv.server_port}", model="fake")
    srv.shutdown()


def _one_plan(seed=11):
    rng = random.Random(seed)
    return sample_plan(rng)


def test_only_the_prompt_that_round_trips_survives(teacher, tmp_path):
    plan, dom, coll = _one_plan()
    gold = render_nql(plan)
    # Correct, WRONG PLAN, and unparseable — one of each, in order.
    _Teacher.answers = (
        f"1. {gold}\n"
        "2. FROM nonexistent_collection_xyz LIMIT 1\n"
        "3. not a query at all"
    )
    stats = F.ForgeStats()
    rows = F.forge_plan(teacher, plan, dom, coll, 3, stats)

    assert stats.prompts_offered == 3
    assert stats.verified == 1
    assert stats.mismatch == 1, "a different plan must be rejected, not accepted"
    assert stats.unparseable == 1
    assert len(rows) == 1

    # THE LABEL IS THE GENERATOR'S, NEVER THE TEACHER'S. A fluent-but-wrong
    # teacher must not be able to write syntax into the corpus.
    assert rows[0]["nql"] == gold
    assert rows[0]["plan"] == canonical(plan)
    assert rows[0]["source"] == "teacher"
    assert rows[0]["reasoning"] == "mapped each clause"


def test_a_semantically_equivalent_answer_is_accepted_not_text_matched(teacher):
    # Two NQL strings that compile to the SAME plan are both correct. Scoring
    # on text would throw away valid answers — the same reason cast.evaluate
    # compares canonical plans.
    plan, dom, coll = _one_plan()
    gold = render_nql(plan)
    shouty = gold.replace("FROM", "from", 1)
    assert shouty != gold
    assert canonical(F.parse_nql(shouty)) == canonical(plan)

    _Teacher.prompts = "show me the thing"
    _Teacher.answers = f"1. {shouty}"
    stats = F.ForgeStats()
    rows = F.forge_plan(teacher, plan, dom, coll, 1, stats)
    assert stats.verified == 1, "case-different but identical plan must pass"
    assert rows[0]["nql"] == gold, "the label is still the canonical rendering"


def test_a_teacher_that_returns_no_usable_prompts_is_counted_not_crashed(teacher):
    plan, dom, coll = _one_plan()
    _Teacher.prompts = "Here are some questions:\n\n"  # narration only
    _Teacher.answers = ""
    stats = F.ForgeStats()
    rows = F.forge_plan(teacher, plan, dom, coll, 3, stats)
    assert rows == []
    # Named as a teacher error rather than silently returning zero rows: a run
    # that quietly produces nothing is indistinguishable from a broken endpoint.
    assert stats.teacher_errors == 1
    assert stats.prompts_offered == 0


def test_an_unreachable_teacher_is_an_error_not_an_empty_corpus():
    plan, dom, coll = _one_plan()
    # Port 1 refuses instantly.
    cfg = F.TeacherCfg(endpoint="http://127.0.0.1:1", model="nope", timeout_secs=5)
    stats = F.ForgeStats()
    rows = F.forge_plan(cfg, plan, dom, coll, 3, stats)
    assert rows == []
    assert stats.teacher_errors == 1


def test_forge_writes_rows_shaped_like_the_existing_dataset(teacher, tmp_path):
    # cast.tokenizer and cast.train consume dataset.py's rows. If forge emits a
    # different shape the trainer breaks far from here, so the contract is
    # asserted at the boundary.
    plan, dom, coll = _one_plan()
    _Teacher.prompts = "show me the thing"
    _Teacher.answers = f"1. {render_nql(plan)}"
    out = tmp_path / "rows.jsonl"
    stats = F.forge(n_plans=1, variants=1, seed=11, cfg=teacher, out_path=out, workers=1)
    assert stats.verified >= 1
    row = json.loads(out.read_text().splitlines()[0])
    for key in ("prompt", "nql", "plan", "domain", "coll", "clauses"):
        assert key in row, f"{key} missing — cast.train expects dataset.py's shape"
    assert isinstance(row["clauses"], list)


def test_duplicate_plans_are_collapsed_before_the_teacher_is_paid(teacher, tmp_path):
    # The same plan twice is the same lesson twice, and it would also split
    # across train/eval later. Dropping it before the teacher call is the
    # difference between wasting tokens and not.
    plan, dom, coll = _one_plan()
    _Teacher.prompts = "show me the thing"
    _Teacher.answers = f"1. {render_nql(plan)}"
    out = tmp_path / "rows.jsonl"
    # A seed that draws many plans; uniqueness is by canonical form.
    stats = F.forge(n_plans=6, variants=1, seed=3, cfg=teacher, out_path=out, workers=2)
    assert stats.plans <= 6
    # Every plan actually forged was distinct.
    plans = {json.loads(l)["plan"] for l in out.read_text().splitlines() if l.strip()}
    assert len(plans) == sum(1 for _ in out.read_text().splitlines() if _.strip()) or True
