#!/usr/bin/env python3
"""
nql_bridge — the only thing crucible needs from this repo.

crucible is the factory: it owns the teacher, the verdict discipline, the
scoreboard and the corpus contract. It does not own NQL semantics, and it should
not — the grammar, the plan canonicaliser and the parser live here, next to the
model that has to learn them, and they are Python.

crucible already shells out to Python for its mutation locator
(`python3 locators/python_locate.py {FILE}`). This is the same pattern for a
different language: a thin, dumb, side-effect-free process that answers two
questions and holds no opinions about training.

PROTOCOL — one JSON object per line on stdin, one per line on stdout.

  {"op":"plans","n":200,"seed":1337}
    -> {"op":"plans","plans":[{"nql":..., "canonical":..., "schema":...,
                               "domain":..., "coll":..., "clauses":[...]}, ...]}

  {"op":"check","nql":"FROM orders WHERE total > 99","gold":"<canonical>"}
    -> {"op":"check","ok":true,  "canonical":"<canonical>"}
    -> {"op":"check","ok":false, "canonical":"<other>"}          # parsed, wrong plan
    -> {"op":"check","ok":false, "error":"NQL: expected field"}  # did not parse

`ok` is PLAN equality, never string equality. Two different NQL strings that
compile to the same plan are both correct, and text comparison would throw away
valid answers — the same reason `cast.evaluate` scores on canonical form.

Deliberately NOT here: any HTTP, any teacher, any notion of a training row.
Those belong to crucible, and duplicating them on this side is how two
implementations of one gate drift apart.
"""
from __future__ import annotations

import json
import random
import sys
from typing import Any, Dict, List

from nedb.query import parse_nql

from cast.grammar import Collection, Domain
from cast.sampler import canonical, clauses_present, render_nql, sample_plan


def schema_block(dom: Domain, coll: Collection) -> str:
    """The schema a teacher may refer to, as prose it can actually use.

    Only the ONE collection the plan touches. Handing over a whole domain
    invites prompts that reference collections the query does not, and those
    come back as round-trip mismatches — paid for, then discarded.

    Human labels are included because they are what a person would SAY.
    Withholding them makes a teacher invent field names no real prompt would
    use, which is the same wasted round trip.
    """
    parts: List[str] = []
    for f in coll.fields:
        bit = f"{f.name}:{f.ftype}"
        if f.label and f.label != f.name:
            bit += f' (called "{f.label}")'
        if f.choices:
            bit += " one of [" + ", ".join(str(c) for c in f.choices[:8]) + "]"
        parts.append(bit)
    return (
        f"database: {dom.name}\n"
        f"collection: {coll.name} ({coll.singular}/{coll.plural})\n"
        "fields:\n  " + "\n  ".join(parts)
    )


def op_plans(req: Dict[str, Any]) -> Dict[str, Any]:
    n = int(req.get("n", 100))
    seed = int(req.get("seed", 1337))
    rng = random.Random(seed)

    # De-duplicated by canonical form here rather than in crucible, because the
    # same plan twice is the same lesson twice AND would split across
    # train/eval later. Dropping it before a teacher is paid is the difference
    # between wasting tokens and not.
    seen = set()
    out: List[Dict[str, Any]] = []
    # Bounded attempts: a tiny grammar can exhaust its distinct plans, and
    # looping until `n` unique ones appear would hang instead of saying so.
    for _ in range(n * 12):
        if len(out) >= n:
            break
        plan, dom, coll = sample_plan(rng)
        cf = canonical(plan)
        if cf in seen:
            continue
        seen.add(cf)
        nql = render_nql(plan)
        # Self-check before handing it over. The generator is already supposed
        # to be parser-clean (cast.dataset asserts this), so a failure here is
        # a generator bug and must be loud rather than becoming a corpus row
        # that teaches invalid syntax.
        recovered = canonical(parse_nql(nql))
        if recovered != cf:
            return {
                "op": "plans",
                "error": (
                    f"generator produced NQL that does not round-trip: {nql!r} "
                    f"parsed to a different plan"
                ),
            }
        out.append(
            {
                "nql": nql,
                "canonical": cf,
                "schema": schema_block(dom, coll),
                "domain": dom.name,
                "coll": coll.name,
                "clauses": clauses_present(plan),
            }
        )
    if len(out) < n:
        # Reported, not padded. A caller that asked for 500 and silently got 80
        # would compute rates against a denominator nobody can reproduce.
        return {
            "op": "plans",
            "plans": out,
            "short": True,
            "note": (
                f"only {len(out)} distinct plans available from this grammar at seed {seed}; "
                f"{n} were requested"
            ),
        }
    return {"op": "plans", "plans": out}


def op_check(req: Dict[str, Any]) -> Dict[str, Any]:
    nql = req.get("nql") or ""
    gold = req.get("gold") or ""
    if not nql.strip():
        return {"op": "check", "ok": False, "error": "empty NQL"}
    try:
        cf = canonical(parse_nql(nql))
    except Exception as e:
        # A teacher's bad syntax is not this bridge's problem to fix, but the
        # reason is returned so crucible can count `unparseable` separately
        # from `mismatch`. One "rejected" counter would hide which of the two
        # is actually costing rows.
        return {"op": "check", "ok": False, "error": f"{type(e).__name__}: {e}"}
    return {"op": "check", "ok": cf == gold, "canonical": cf}


OPS = {"plans": op_plans, "check": op_check}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"not JSON: {e}"}), flush=True)
            continue
        fn = OPS.get(req.get("op"))
        if fn is None:
            print(
                json.dumps({"error": f"unknown op {req.get('op')!r}; known: {sorted(OPS)}"}),
                flush=True,
            )
            continue
        try:
            print(json.dumps(fn(req)), flush=True)
        except Exception as e:
            # Never die mid-stream: crucible has a whole run in flight, and one
            # bad request should cost one row, not the corpus.
            print(json.dumps({"op": req.get("op"), "error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
