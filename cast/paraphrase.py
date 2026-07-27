"""
cast.paraphrase — turn a sampled plan into a short, human-sounding prompt.

This module is the real quality ceiling of the project. If every prompt for
`WHERE total > 99` reads "show me orders where total is greater than 99", the
model learns to pattern-match one template and collapses the moment a user
types "which orders cleared a hundred bucks".

So we attack surface diversity on several axes at once:
  - lead-in verb ("show me" / "list" / "pull up" / bare / "who are the")
  - operator phrasing ("over" / "more than" / "above" / "greater than" / ">")
  - field naming (raw field / human label / possessive form)
  - value phrasing ("$99.50" / "99.5" / "ninety-nine fifty" for small ints)
  - clause ORDER — humans don't say clauses in parser order
  - ellipsis (dropping the collection when it's implied)
  - register (terse "orders over 99" vs polite "could you show me...")
  - noise (lowercase, missing punctuation, occasional typo)

`HOLDOUT_STYLE` renders the SAME plan in phrasings deliberately absent from
training, so held-out accuracy measures generalisation, not memorisation.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from .grammar import (BOOL, DATE, ENUM, INT, MONEY, NAME, Collection, Domain,
                      Field)

# ------------------------------------------------------------------ lead-ins

LEADINS = [
    "show me", "show", "list", "list all", "get", "get me", "find", "find all",
    "pull up", "pull", "give me", "fetch", "query", "look up", "i want",
    "i need", "display", "return", "grab", "", "", "",  # bare is common
]
POLITE = ["could you show me", "can you list", "please show", "can i see",
          "would you pull up", "i'd like to see"]

# ------------------------------------------------------------- op phrasings

OP_WORDS = {
    ">":  ["over", "above", "more than", "greater than", "higher than",
           "exceeding", "north of", ">"],
    ">=": ["at least", "no less than", "or more", "minimum", ">=",
           "at or above"],
    "<":  ["under", "below", "less than", "lower than", "south of", "<"],
    "<=": ["at most", "or less", "up to", "maximum", "<=", "at or below"],
    "=":  ["is", "equals", "=", "set to", "marked", "that is", "where it's"],
    "!=": ["is not", "isn't", "not", "!=", "other than", "excluding",
           "anything but", "except"],
}
# words only valid for countable integer fields ("fewer than 5 orders")
COUNTABLE_OPS = {
    "<": ["fewer than"],
    ">": ["more than"],
}
# words only valid for money fields
MONEY_OPS = {
    "<": ["cheaper than", "under"],
    ">": ["pricier than", "over"],
}
# for date fields, comparison words differ
DATE_OP_WORDS = {
    ">":  ["after", "later than", "since", "newer than", "past"],
    ">=": ["on or after", "from", "starting"],
    "<":  ["before", "earlier than", "prior to", "older than", "up to"],
    "<=": ["on or before", "through", "no later than"],
    "=":  ["on", "dated", "from"],
    "!=": ["not on", "excluding"],
}

SORT_ASC = ["sorted by", "ordered by", "by", "sort by", "order by",
            "arranged by", "in order of", "ascending by", "lowest first by",
            "cheapest first by"]
SORT_DESC = ["sorted by {f} descending", "highest {f} first", "top {f} first",
             "by {f} descending", "ordered by {f} desc", "biggest {f} first",
             "most {f} first", "{f} high to low", "descending by {f}",
             "sort by {f} desc"]

LIMIT_PRE = ["top", "first", "just", "only", "the first", "give me",
             "limit to", "max"]
# NOTE: no "no more than {n}" here — it collides with the <= comparison words
# and produced garbage like "fee that is 2.87 no more than 20".
LIMIT_POST = ["only {n}", "limit {n}", "just {n}", "max {n}", "{n} of them",
              "cap at {n}", "limit to {n}"]

SEARCH_WORDS = ["mentioning", "matching", "containing", "about", "with",
                "that mention", "referencing", "talking about", "search for",
                "that say", "including the text", "with the words"]

GROUP_WORDS = ["grouped by", "by", "broken down by", "per", "grouped on",
               "split by", "for each", "aggregated by"]
AGG_WORDS = {
    "count": ["count", "how many", "number of", "total count", "counts",
              "tally"],
    "sum":   ["total", "sum of", "summed", "total up", "add up", "sum"],
    "avg":   ["average", "mean", "avg", "average of", "typical"],
    "min":   ["minimum", "lowest", "min", "smallest", "cheapest"],
    "max":   ["maximum", "highest", "max", "largest", "biggest", "peak"],
}

AS_OF_WORDS = ["as of seq {n}", "at sequence {n}", "as of sequence {n}",
               "at seq {n}", "as they were at seq {n}", "at version {n}",
               "rewind to seq {n}", "state at seq {n}", "as of {n}",
               "snapshot at seq {n}", "back at seq {n}"]
VALID_WORDS = ["valid as of {d}", "as they were valid on {d}",
               "effective {d}", "valid on {d}", "in effect on {d}",
               "as of the date {d}", "what was true on {d}",
               "valid at {d}"]

TRAVERSE_WORDS = ["traverse {r}", "follow {r}", "walk {r}", "through {r}",
                  "via {r}", "following the {r} relation", "hop {r}",
                  "along {r}"]
TRACE_WORDS = ["trace {f}", "show provenance", "trace lineage",
               "where did these come from", "show the causal chain",
               "trace {f} back", "what caused these", "show lineage",
               "trace provenance"]
TRACE_REV_WORDS = ["trace {f} reverse", "what did these cause",
                   "forward lineage", "downstream effects",
                   "what came from these", "trace {f} forward",
                   "show what these caused", "downstream of these"]

BOOL_TRUE = ["that are {f}", "which are {f}", "{f} ones", "the {f} ones",
             "where {f} is true", "{f} = true", "marked {f}"]
BOOL_FALSE = ["that aren't {f}", "not {f}", "which are not {f}",
              "where {f} is false", "{f} = false", "un{f}"]

TYPO_MAP = {"show": "shwo", "orders": "orderss", "where": "wher",
            "customers": "custommers", "total": "totl", "the": "teh",
            "list": "lsit", "find": "fnid"}


def _num_str(v: Any, fld: Field, rng: random.Random) -> str:
    """Phrase a value the way a human would type it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        if fld.ftype == MONEY and rng.random() < 0.45:
            # money often typed with a $ and sometimes "bucks"
            s = f"${v:g}"
            if rng.random() < 0.12:
                s = f"{v:g} bucks"
            return s
        if rng.random() < 0.08 and float(v).is_integer() and 0 < v <= 20:
            words = ["zero", "one", "two", "three", "four", "five", "six",
                     "seven", "eight", "nine", "ten", "eleven", "twelve",
                     "thirteen", "fourteen", "fifteen", "sixteen",
                     "seventeen", "eighteen", "nineteen", "twenty"]
            return words[int(v)]
        return f"{v:g}"
    # string value
    s = str(v)
    if rng.random() < 0.30:
        return f'"{s}"'
    return s


def _field_phrase(fld: Field, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.42:
        return fld.name
    if r < 0.80:
        return fld.label
    return fld.name.replace("_", " ")


def _pred_phrase(fld: Field, op: str, val: Any, rng: random.Random,
                 coll: Collection) -> str:
    """Render one WHERE predicate as natural language."""
    # boolean special-case reads much more naturally
    if fld.ftype == BOOL and op == "=":
        tmpl = rng.choice(BOOL_TRUE if val else BOOL_FALSE)
        return tmpl.format(f=fld.label)

    # Type-aware operator wording. "fewer than" is only valid for countable
    # ints; "cheaper than" only for money. Mixing them produced nonsense like
    # "lifetime value fewer than 4046.1".
    if fld.ftype == DATE:
        pool = list(DATE_OP_WORDS.get(op, OP_WORDS[op]))
    else:
        pool = list(OP_WORDS[op])
        if fld.ftype == MONEY:
            pool += MONEY_OPS.get(op, [])
        elif fld.ftype == INT:
            pool += COUNTABLE_OPS.get(op, [])
    word = rng.choice(pool)
    fp = _field_phrase(fld, rng)
    vp = _num_str(val, fld, rng)

    # enum/name equality often drops the field entirely: "orders that are paid"
    if fld.ftype in (ENUM, NAME) and op == "=" and rng.random() < 0.42:
        if fld.ftype == NAME and rng.random() < 0.5:
            return rng.choice([f"for {vp}", f"by {vp}", f"belonging to {vp}",
                               f"{fp} {vp}"])
        return rng.choice([f"that are {vp}", f"marked {vp}", f"in {vp}",
                           f"{vp}", f"with {fp} {vp}", f"that's {vp}"])

    # "or more"/"or less"-style suffix words read after the value
    if word in ("or more", "or less", "{n} of them"):
        return f"{fp} {vp} {word}"
    if word in ("minimum", "maximum", "at least", "at most", "up to", "from"):
        return f"{word} {vp} {fp}" if rng.random() < 0.5 else f"{fp} {word} {vp}"

    return f"{fp} {word} {vp}"


def paraphrase(plan: Dict[str, Any], dom: Domain, coll: Collection,
               rng: random.Random, holdout: bool = False) -> str:
    """Build one natural-language prompt for this plan.

    holdout=True uses phrasings deliberately excluded from the training pool,
    to measure generalisation rather than memorisation.
    """
    if holdout:
        return _holdout_paraphrase(plan, dom, coll, rng)

    fields = {f.name: f for f in coll.fields}

    # --- lead-in chosen first, so the subject can avoid stacking determiners
    #     ("find all" + "all events" -> "find all all events").
    lead = rng.choice(POLITE) if rng.random() < 0.07 else rng.choice(LEADINS)
    lead_has_det = any(w in lead for w in ("all", "the", "me"))

    subj_r = rng.random()
    if subj_r < 0.70:
        subject = coll.name
    elif subj_r < 0.86:
        subject = coll.plural
    elif lead_has_det:
        subject = rng.choice([coll.plural, coll.name])
    else:
        subject = rng.choice([coll.plural, f"all {coll.plural}",
                              f"every {coll.singular}", coll.name])

    # --- collect clause fragments
    where_frags: List[str] = []
    for (fname, op, val) in (plan.get("where") or []):
        fld = fields.get(fname)
        if fld is None:
            continue
        where_frags.append(_pred_phrase(fld, op, val, rng, coll))

    frags: List[str] = []

    if plan.get("search") is not None:
        frags.append(f'{rng.choice(SEARCH_WORDS)} "{plan["search"]}"'
                     if rng.random() < 0.55
                     else f'{rng.choice(SEARCH_WORDS)} {plan["search"]}')

    if plan.get("order_by") is not None:
        fname, direction = plan["order_by"]
        fld = fields.get(fname)
        fp = _field_phrase(fld, rng) if fld else fname
        if str(direction).upper() == "DESC":
            frags.append(rng.choice(SORT_DESC).format(f=fp))
        else:
            frags.append(f"{rng.choice(SORT_ASC)} {fp}")

    if plan.get("traverse") is not None:
        frags.append(rng.choice(TRAVERSE_WORDS).format(
            r=plan["traverse"].replace("_", " ") if rng.random() < 0.4
            else plan["traverse"]))

    if plan.get("trace") is not None:
        pool = TRACE_REV_WORDS if plan.get("trace_reverse") else TRACE_WORDS
        frags.append(rng.choice(pool).format(f=plan["trace"]))

    if plan.get("as_of") is not None:
        frags.append(rng.choice(AS_OF_WORDS).format(n=plan["as_of"]))

    if plan.get("valid_as_of") is not None:
        d = plan["valid_as_of"]
        dp = f'"{d}"' if rng.random() < 0.4 else d
        frags.append(rng.choice(VALID_WORDS).format(d=dp))

    if plan.get("group_by") is not None:
        gname = plan["group_by"]
        gfld = fields.get(gname)
        gp = _field_phrase(gfld, rng) if gfld else gname
        agg = plan.get("aggregate")
        if agg:
            aname, afield = agg
            aw = rng.choice(AGG_WORDS[aname])
            if afield:
                af = fields.get(afield)
                afp = _field_phrase(af, rng) if af else afield
                if rng.random() < 0.5:
                    frags.append(f"{aw} {afp} {rng.choice(GROUP_WORDS)} {gp}")
                else:
                    frags.append(f"{rng.choice(GROUP_WORDS)} {gp} with {aw} {afp}")
            else:
                if rng.random() < 0.5:
                    frags.append(f"{aw} {rng.choice(GROUP_WORDS)} {gp}")
                else:
                    frags.append(f"{rng.choice(GROUP_WORDS)} {gp} {aw}")
        else:
            frags.append(f"{rng.choice(GROUP_WORDS)} {gp}")

    # --- limit can go in front ("top 5 orders") or trail ("... limit 5")
    limit_front = None
    if plan.get("limit") is not None:
        if rng.random() < 0.5:
            limit_front = f"{rng.choice(LIMIT_PRE)} {plan['limit']}"
        else:
            frags.append(rng.choice(LIMIT_POST).format(n=plan["limit"]))

    # humans don't order clauses like a parser
    rng.shuffle(frags)

    # --- assemble (lead was chosen above, before the subject)
    head_parts: List[str] = []
    if lead:
        head_parts.append(lead)
    if limit_front:
        head_parts.append(limit_front)
    head_parts.append(subject)

    text = " ".join(head_parts)

    if where_frags:
        joiner = rng.choice(["where ", "with ", "", "that have ", "having "])
        conj = rng.choice([" and ", " and ", ", ", " & "])
        text += " " + joiner + conj.join(where_frags)

    if frags:
        text += " " + " ".join(frags)

    text = " ".join(text.split())

    # --- noise: casing, punctuation, occasional typo
    r = rng.random()
    if r < 0.06:
        text = text.capitalize()
    elif r < 0.09:
        text = text.upper()
    if rng.random() < 0.05:
        text += rng.choice(["?", ".", " please", "!"])
    if rng.random() < 0.03:
        for k, v in TYPO_MAP.items():
            if k in text:
                text = text.replace(k, v, 1)
                break

    return text


def _holdout_paraphrase(plan: Dict[str, Any], dom: Domain, coll: Collection,
                        rng: random.Random) -> str:
    """Phrasings intentionally NOT in the training pool.

    Different verbs, different operator words, different clause framing. If the
    model scores well here it generalised; if it only scores well on the
    training style it memorised templates.
    """
    fields = {f.name: f for f in coll.fields}
    HO_LEAD = ["which", "what are the", "surface", "dig up", "hand me",
               "round up", "scan for", "enumerate", "tell me the"]
    HO_OPS = {
        ">": ["that cleared", "beyond", "topping", "in excess of"],
        ">=": ["hitting", "reaching", "not under"],
        "<": ["shy of", "beneath", "trailing", "short of"],
        "<=": ["capped at", "not over", "within"],
        "=": ["flagged", "sitting at", "logged as", "reading"],
        "!=": ["anything besides", "skipping", "minus", "aside from"],
    }
    parts = [rng.choice(HO_LEAD), coll.name]
    for (fname, op, val) in (plan.get("where") or []):
        fld = fields.get(fname)
        if not fld:
            continue
        w = rng.choice(HO_OPS.get(op, ["is"]))
        parts.append(f"{fld.label} {w} {val}")
    if plan.get("order_by"):
        f, d = plan["order_by"]
        parts.append(f"ranked on {f} {'downward' if str(d).upper()=='DESC' else 'upward'}")
    if plan.get("limit") is not None:
        parts.append(f"capped to {plan['limit']} rows")
    if plan.get("search") is not None:
        parts.append(f'carrying the phrase "{plan["search"]}"')
    if plan.get("as_of") is not None:
        parts.append(f"wound back to seq {plan['as_of']}")
    if plan.get("valid_as_of") is not None:
        parts.append(f"true on the date {plan['valid_as_of']}")
    if plan.get("traverse") is not None:
        parts.append(f"hopping the {plan['traverse']} edge")
    if plan.get("trace") is not None:
        parts.append("unwind the causal chain" if not plan.get("trace_reverse")
                     else "chase the downstream chain")
    if plan.get("group_by") is not None:
        agg = plan.get("aggregate")
        seg = f"bucketed on {plan['group_by']}"
        if agg:
            seg += f" reporting {agg[0]}" + (f" of {agg[1]}" if agg[1] else "")
        parts.append(seg)
    return " ".join(" ".join(parts).split())
