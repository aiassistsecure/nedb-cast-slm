"""
cast.sampler — sample random *valid* NQL plans, then render them to canonical NQL.

Two invariants make the whole project work:

  1. Every plan we emit must parse. We enforce this by rendering to NQL text and
     round-tripping through the REAL parser (nedb.query.parse_nql). Anything that
     fails to round-trip is a generator bug and gets dropped loudly, never
     silently shipped into the corpus.

  2. Clause coverage must be even and controllable. The model can only learn a
     clause it has seen. `CLAUSE_P` is the knob; eval reports per-clause accuracy
     against it so a rare clause can't hide inside a good average.

Grammar targeted (from nedb/query.py, verified against 2.7.2):

    FROM <coll> [AS OF <seq>] [VALID AS OF "<date>"]
      [WHERE <f> <op> <v> (AND ...)*] [SEARCH "<text>"]
      [ORDER BY <f> [ASC|DESC]] [TRAVERSE <rel>]
      [TRACE <field> [REVERSE]] [LIMIT <n>]
      [GROUP BY <f> [COUNT|SUM <f>|AVG <f>|MIN <f>|MAX <f>]]

Note the parser's clause ORDER is fixed: LIMIT is parsed *before* GROUP BY, so
canonical rendering must follow that sequence exactly.
"""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from .grammar import (BOOL, DATE, ENUM, FLOAT, MONEY, Collection, Domain, Field,
                      pick_domain)

# Probability each optional clause appears. Tuned so single-clause queries are
# the most common case (that's what people actually type) while multi-clause
# combinations still appear often enough to learn.
CLAUSE_P = {
    "where": 0.62,
    "where_second": 0.28,   # conditional on `where`
    "where_third": 0.07,    # conditional on `where_second`
    "as_of": 0.10,
    "valid_as_of": 0.09,
    "search": 0.16,
    "order_by": 0.30,
    "traverse": 0.10,
    "trace": 0.09,
    "trace_reverse": 0.40,  # conditional on `trace`
    "limit": 0.34,
    "group_by": 0.14,
    "aggregate": 0.75,      # conditional on `group_by`
}

NUM_OPS = ["=", "!=", "<", "<=", ">", ">="]
EQ_OPS = ["=", "!="]
AGGS = ["count", "sum", "avg", "min", "max"]

# Trace fields the engine understands. `caused_by` is the real provenance edge.
TRACE_FIELDS = ["caused_by"]


def _fmt_value(v: Any) -> str:
    """Render a Python value as NQL literal text."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v) if isinstance(v, float) else str(v)
    # strings always quoted — double quotes are canonical
    return '"%s"' % str(v)


def sample_plan(rng: random.Random) -> Tuple[Dict[str, Any], Domain, Collection]:
    """Sample one random valid plan. Returns (plan, domain, collection)."""
    dom = pick_domain(rng)
    coll = rng.choice(dom.collections)

    plan: Dict[str, Any] = {
        "from": coll.name, "as_of": None, "where": [], "search": None,
        "order_by": None, "traverse": None, "limit": None,
        "group_by": None, "aggregate": None,
        "trace": None, "trace_reverse": False, "valid_as_of": None,
    }

    # --- AS OF <seq>
    if rng.random() < CLAUSE_P["as_of"]:
        plan["as_of"] = rng.choice([
            rng.randint(1, 50), rng.randint(50, 999),
            rng.randint(1000, 99999), rng.choice([100, 500, 1000, 5000]),
        ])

    # --- VALID AS OF "<date>"
    if rng.random() < CLAUSE_P["valid_as_of"]:
        y, m, d = rng.randint(2023, 2026), rng.randint(1, 12), rng.randint(1, 28)
        plan["valid_as_of"] = f"{y:04d}-{m:02d}-{d:02d}"

    # --- WHERE (1..3 predicates, distinct fields)
    if rng.random() < CLAUSE_P["where"]:
        n_pred = 1
        if rng.random() < CLAUSE_P["where_second"]:
            n_pred = 2
            if rng.random() < CLAUSE_P["where_third"]:
                n_pred = 3
        used: set = set()
        for _ in range(n_pred):
            avail = [f for f in coll.fields if f.name not in used]
            if not avail:
                break
            fld = rng.choice(avail)
            used.add(fld.name)
            if fld.ftype == BOOL:
                op = "="
            elif fld.is_ordered():
                op = rng.choice(NUM_OPS)
                # Exact equality on a continuous value is unnatural — nobody
                # queries `rating != 1.6`. Push floats/money toward ranges.
                if fld.ftype in (FLOAT, MONEY) and op in ("=", "!="):
                    op = rng.choice([">", ">=", "<", "<="])
            else:
                op = rng.choice(EQ_OPS)
            val = fld.sample_value(rng)
            plan["where"].append((fld.name, op, val))

    # --- SEARCH "<text>"
    if rng.random() < CLAUSE_P["search"] and coll.searchable:
        plan["search"] = rng.choice(coll.searchable)

    # --- ORDER BY <f> [ASC|DESC]
    if rng.random() < CLAUSE_P["order_by"]:
        fld = coll.field(rng)
        plan["order_by"] = (fld.name, rng.choice(["ASC", "DESC"]))

    # --- TRAVERSE <rel> — only edges valid FROM this collection
    rels = coll.relations or dom.relations
    if rng.random() < CLAUSE_P["traverse"] and rels:
        plan["traverse"] = rng.choice(rels)

    # --- TRACE <field> [REVERSE]
    if rng.random() < CLAUSE_P["trace"]:
        plan["trace"] = rng.choice(TRACE_FIELDS)
        plan["trace_reverse"] = rng.random() < CLAUSE_P["trace_reverse"]

    # --- LIMIT <n>
    if rng.random() < CLAUSE_P["limit"]:
        plan["limit"] = rng.choice([1, 3, 5, 5, 10, 10, 20, 25, 50, 100,
                                    rng.randint(2, 99)])

    # --- GROUP BY <f> [agg]
    if rng.random() < CLAUSE_P["group_by"]:
        # group by a low-cardinality field where possible
        cats = [f for f in coll.fields if f.ftype in (ENUM, DATE)]
        gf = rng.choice(cats) if cats else coll.field(rng)
        plan["group_by"] = gf.name
        if rng.random() < CLAUSE_P["aggregate"]:
            agg = rng.choice(AGGS)
            if agg == "count":
                plan["aggregate"] = ("count", None)
            else:
                nums = [f for f in coll.fields if f.is_numeric()]
                if nums:
                    plan["aggregate"] = (agg, rng.choice(nums).name)
                else:
                    plan["aggregate"] = ("count", None)

    return plan, dom, coll


def render_nql(plan: Dict[str, Any]) -> str:
    """Render a plan to canonical NQL text.

    Clause order MUST match nedb/query.py's parse order:
    FROM, AS OF, VALID AS OF, WHERE, SEARCH, ORDER BY, TRAVERSE, TRACE, LIMIT, GROUP BY
    """
    parts: List[str] = ["FROM", plan["from"]]

    if plan.get("as_of") is not None:
        parts += ["AS", "OF", str(plan["as_of"])]

    if plan.get("valid_as_of") is not None:
        parts += ["VALID", "AS", "OF", '"%s"' % plan["valid_as_of"]]

    if plan.get("where"):
        parts.append("WHERE")
        preds = []
        for (f, op, v) in plan["where"]:
            preds.append(f"{f} {op} {_fmt_value(v)}")
        parts.append(" AND ".join(preds))

    if plan.get("search") is not None:
        parts += ["SEARCH", '"%s"' % plan["search"]]

    if plan.get("order_by") is not None:
        f, d = plan["order_by"]
        parts += ["ORDER", "BY", f, d]

    if plan.get("traverse") is not None:
        parts += ["TRAVERSE", plan["traverse"]]

    if plan.get("trace") is not None:
        parts += ["TRACE", plan["trace"]]
        if plan.get("trace_reverse"):
            parts.append("REVERSE")

    if plan.get("limit") is not None:
        parts += ["LIMIT", str(plan["limit"])]

    if plan.get("group_by") is not None:
        parts += ["GROUP", "BY", plan["group_by"]]
        agg = plan.get("aggregate")
        if agg:
            name, field = agg
            parts.append(name.upper())
            if field:
                parts.append(field)

    return " ".join(parts)


def canonical(plan: Dict[str, Any]) -> str:
    """Stable string form of a plan, for exact-match comparison.

    Normalises tuples->lists and ORDER BY direction casing so semantically
    identical plans compare equal regardless of how they were produced.
    """
    p = {
        "from": plan.get("from"),
        "as_of": plan.get("as_of"),
        "valid_as_of": plan.get("valid_as_of"),
        "where": [list(w) for w in (plan.get("where") or [])],
        "search": plan.get("search"),
        "order_by": (list(plan["order_by"]) if plan.get("order_by") else None),
        "traverse": plan.get("traverse"),
        "trace": plan.get("trace"),
        "trace_reverse": bool(plan.get("trace_reverse")),
        "limit": plan.get("limit"),
        "group_by": plan.get("group_by"),
        "aggregate": (list(plan["aggregate"]) if plan.get("aggregate") else None),
    }
    if p["order_by"]:
        p["order_by"][1] = str(p["order_by"][1]).upper()
    return json.dumps(p, sort_keys=True, separators=(",", ":"), default=str)


def clauses_present(plan: Dict[str, Any]) -> List[str]:
    """Which optional clauses this plan exercises — drives per-clause eval."""
    out = []
    if plan.get("where"):
        out.append("where")
        if len(plan["where"]) > 1:
            out.append("where_multi")
    if plan.get("as_of") is not None:
        out.append("as_of")
    if plan.get("valid_as_of") is not None:
        out.append("valid_as_of")
    if plan.get("search") is not None:
        out.append("search")
    if plan.get("order_by") is not None:
        out.append("order_by")
    if plan.get("traverse") is not None:
        out.append("traverse")
    if plan.get("trace") is not None:
        out.append("trace")
        if plan.get("trace_reverse"):
            out.append("trace_reverse")
    if plan.get("limit") is not None:
        out.append("limit")
    if plan.get("group_by") is not None:
        out.append("group_by")
        if plan.get("aggregate"):
            out.append("aggregate")
    if not out:
        out.append("bare")
    return out
