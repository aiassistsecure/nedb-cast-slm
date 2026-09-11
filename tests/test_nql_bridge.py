"""
Tests for scripts/nql_bridge.py — the only surface crucible depends on.

This is a CONTRACT test suite, not a feature suite. crucible drives this bridge
over a pipe, so a silently changed key or a collapsed error distinction breaks a
Rust program in another repo, at a distance, mid-run. The protocol is asserted
field by field for exactly that reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BRIDGE = REPO / "scripts" / "nql_bridge.py"


def run(*requests: dict) -> list[dict]:
    """Send requests down the pipe, one JSON object per line, read the replies."""
    payload = "".join(json.dumps(r) + "\n" for r in requests)
    out = subprocess.run(
        [sys.executable, str(BRIDGE)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert out.returncode == 0, f"bridge exited {out.returncode}: {out.stderr[:500]}"
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]


def test_plans_carry_everything_a_teacher_needs_and_nothing_it_should_not():
    (reply,) = run({"op": "plans", "n": 5, "seed": 11})
    assert "error" not in reply, reply
    plans = reply["plans"]
    assert len(plans) == 5
    for p in plans:
        # The teacher needs the query, the schema to phrase it against, and
        # crucible needs the canonical form to gate on.
        for key in ("nql", "canonical", "schema", "domain", "coll", "clauses"):
            assert key in p, f"{key} missing — crucible reads this field by name"
        assert p["nql"].upper().startswith("FROM ")
        assert isinstance(p["clauses"], list)
        # The schema names the collection being queried and its fields, so a
        # teacher cannot invent field names that will fail the round trip.
        assert p["coll"] in p["schema"]


def test_plans_are_distinct_by_canonical_form():
    # The same plan twice is the same lesson twice, and it would also split
    # across train/eval later. De-duplicating BEFORE a teacher is paid is the
    # difference between wasting tokens and not.
    (reply,) = run({"op": "plans", "n": 40, "seed": 5})
    forms = [p["canonical"] for p in reply["plans"]]
    assert len(forms) == len(set(forms)), "duplicate plans reached the caller"


def test_the_same_seed_gives_the_same_plans():
    # crucible reports rates against these plans. A run nobody can reproduce
    # produces a denominator nobody can check.
    (a,) = run({"op": "plans", "n": 8, "seed": 99})
    (b,) = run({"op": "plans", "n": 8, "seed": 99})
    assert [p["nql"] for p in a["plans"]] == [p["nql"] for p in b["plans"]]


def test_a_short_supply_of_distinct_plans_is_reported_not_padded():
    # A caller that asked for 10_000 and silently got 300 would compute a rate
    # against a denominator that is not the one it printed.
    (reply,) = run({"op": "plans", "n": 100_000, "seed": 1})
    if reply.get("short"):
        assert "note" in reply and "requested" in reply["note"]
        assert len(reply["plans"]) < 100_000
    else:
        # The grammar was big enough — then it must have delivered in full.
        assert len(reply["plans"]) == 100_000


def test_check_accepts_a_semantically_equal_query_not_a_textually_equal_one():
    (plans,) = run({"op": "plans", "n": 1, "seed": 11})
    p = plans["plans"][0]
    # Lower-cased keyword: a different STRING, the same PLAN. Text comparison
    # would discard this, which is why the gate is canonical-form equality.
    shouty = p["nql"].replace("FROM", "from", 1)
    assert shouty != p["nql"]
    (reply,) = run({"op": "check", "nql": shouty, "gold": p["canonical"]})
    assert reply["ok"] is True, reply
    assert reply["canonical"] == p["canonical"]


def test_check_separates_a_wrong_plan_from_unparseable_syntax():
    (plans,) = run({"op": "plans", "n": 1, "seed": 11})
    gold = plans["plans"][0]["canonical"]

    # Parsed fine, describes a DIFFERENT query. crucible counts this as
    # `mismatch` — evidence the English was ambiguous.
    (wrong,) = run({"op": "check", "nql": "FROM something_else LIMIT 1", "gold": gold})
    assert wrong["ok"] is False
    assert "canonical" in wrong, "a parsed-but-wrong answer must still report its plan"
    assert "error" not in wrong

    # Did not parse at all. crucible counts this as `unparseable` — evidence
    # about the TEACHER's syntax, not about the prompt. Collapsing the two
    # would hide which is costing rows.
    (bad,) = run({"op": "check", "nql": "garbage nonsense", "gold": gold})
    assert bad["ok"] is False
    assert "error" in bad and "canonical" not in bad

    (empty,) = run({"op": "check", "nql": "   ", "gold": gold})
    assert empty["ok"] is False and "error" in empty


def test_a_bad_request_costs_one_line_not_the_run():
    # crucible has a whole corpus in flight on the other end of this pipe. One
    # malformed request must not kill the process.
    replies = run(
        {"op": "bogus"},
        {"op": "check", "nql": "FROM x", "gold": "nope"},
    )
    assert len(replies) == 2
    assert "unknown op" in replies[0]["error"]
    assert replies[1]["op"] == "check"


def test_non_json_input_is_answered_and_the_stream_continues():
    payload = "not json at all\n" + json.dumps({"op": "plans", "n": 1, "seed": 3}) + "\n"
    out = subprocess.run(
        [sys.executable, str(BRIDGE)],
        input=payload,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert out.returncode == 0
    lines = [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
    assert len(lines) == 2
    assert "not JSON" in lines[0]["error"]
    assert len(lines[1]["plans"]) == 1
