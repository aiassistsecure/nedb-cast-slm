"""
cast.forge — teacher-written prompts, kept only when they ROUND-TRIP.

`cast.paraphrase` opens by naming itself "the real quality ceiling of the
project", and it is right. Every prompt in the current corpus comes from
hand-written templates, so the model learns the templates. The docstring's own
example is the failure: train only on "show me orders where total is greater
than 99" and the model folds the moment someone types "which orders cleared a
hundred bucks".

This module raises that ceiling by asking a large teacher model to write the
English instead — and then refuses to trust it.

THE GATE, which is the whole point
----------------------------------
`cast.dataset` already verifies that the generated NQL parses back to the
sampled plan. Nothing verifies the direction that actually matters: whether the
ENGLISH is recoverable. A template — or a teacher — can emit a prompt that is
ambiguous, under-specified, or plain wrong about its own query, and the corpus
swallows it. The student then learns to map an unanswerable question to a
specific plan, which is noise wearing a label.

So every candidate prompt is round-tripped:

  1. sample a plan            -> gold_plan
  2. teacher writes English   -> prompt
  3. teacher reads ONLY the prompt (no plan, no gold NQL) -> nql
  4. parse_nql(nql)           -> recovered plan
  5. keep the pair iff canonical(recovered) == canonical(gold_plan)

Step 5 is PLAN equality, not string equality. Two different NQL strings that
compile to the same plan are both correct, and text comparison would throw away
valid answers — the same reason `cast.evaluate` scores on canonical form.

A prompt that fails the round trip is not a bad teacher; it is usually an
ambiguous prompt, and discarding it is the point. If a 271 GiB model cannot
recover the query from its own description, a 3.33M student never will.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
The teacher's NQL is never used as the label. The label is always
`render_nql(gold_plan)` — canonical, generator-produced, already parser-checked.
The teacher's only jobs are to invent English and to act as its own judge. That
keeps a fluent-but-wrong teacher from writing syntax into the corpus.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nedb.query import parse_nql

from .grammar import Collection, Domain
from .sampler import canonical, clauses_present, render_nql, sample_plan

# ------------------------------------------------------------------ the teacher


@dataclass
class TeacherCfg:
    """How to reach the teacher, and how patient to be with it."""

    endpoint: str = "http://127.0.0.1:11434"
    model: str = "glm-5.3-flash"
    # TOTAL budget: reasoning plus answer. A thinking model spends most of it
    # before writing a single visible character. At 400 the real teacher
    # returned HTTP 200 with EMPTY content and finish_reason `length` as the
    # only clue; it needed 875. Starving it looks exactly like it failing.
    max_tokens: int = 3000
    # 0.9 for paraphrasing: DIVERSITY is the product here. This is the one call
    # in the pipeline where sampling away from the mode is the goal.
    paraphrase_temperature: float = 0.9
    # 0.0 for translation: there is one right plan and creativity is a defect.
    translate_temperature: float = 0.0
    timeout_secs: int = 300
    api_key: Optional[str] = None

    def __post_init__(self) -> None:
        self.endpoint = self.endpoint.rstrip("/")


class TeacherError(RuntimeError):
    """A teacher request that did not produce usable text. Never silent."""


def _post(cfg: TeacherCfg, system: str, user: str, temperature: float) -> Tuple[str, Optional[str]]:
    """One chat completion. Returns (content, reasoning).

    curl subprocess rather than `requests`, to keep this repo's runtime deps as
    thin as they already are — and the body goes through a temp FILE, not argv,
    because prompts carry newlines and quotes and a generator should not die on
    row 4000 with a shell quoting error.
    """
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": temperature,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(body, fh)
        body_path = fh.name

    argv = [
        "curl", "--silent", "--show-error", "--fail-with-body",
        "--max-time", str(cfg.timeout_secs),
        "-H", "Content-Type: application/json",
        "--data-binary", "@" + body_path,
    ]
    if cfg.api_key:
        argv += ["-H", f"Authorization: Bearer {cfg.api_key}"]
    argv.append(f"{cfg.endpoint}/v1/chat/completions")

    try:
        out = subprocess.run(argv, capture_output=True, text=True)
    finally:
        Path(body_path).unlink(missing_ok=True)

    if out.returncode != 0:
        # Name the code AND stderr AND the body: a gateway refusing a request
        # answers in the body, and reporting only the exit code turns "model
        # not found" into "curl failed with 22".
        raise TeacherError(
            f"curl exit {out.returncode}: {out.stderr.strip() or 'no stderr'}"
            + (f" — body: {out.stdout[:300]}" if out.stdout.strip() else "")
        )

    try:
        v = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise TeacherError(f"not JSON ({e}): {out.stdout[:300]}") from e

    if isinstance(v, dict) and v.get("error"):
        raise TeacherError(f"teacher refused: {json.dumps(v['error'])[:300]}")

    try:
        choice = v["choices"][0]
        msg = choice["message"]
    except (KeyError, IndexError) as e:
        raise TeacherError(f"no choices: {out.stdout[:300]}") from e

    content = (msg.get("content") or "").strip()
    # llama.cpp emits `reasoning_content`; some gateways emit `thinking`.
    reasoning = msg.get("reasoning_content") or msg.get("thinking") or None
    if reasoning is not None and not str(reasoning).strip():
        reasoning = None

    if not content:
        # THE THINKING-MODEL TRAP, named rather than returned as "". An empty
        # string looks like the model failing the task; the real cause is
        # usually a budget spent entirely on reasoning.
        raise TeacherError(
            f"EMPTY content (finish_reason={choice.get('finish_reason')}, "
            f"{len(reasoning or '')} chars reasoning) — raise max_tokens above {cfg.max_tokens}"
        )
    return content, reasoning


# ------------------------------------------------------------------- prompting


def _schema_block(dom: Domain, coll: Collection) -> str:
    """The schema the teacher may refer to.

    Only the ONE collection being queried, with field names and types. Handing
    over the whole domain invites prompts that reference collections the plan
    does not touch, which round-trip as ambiguous and get discarded — wasted
    tokens either way.
    """
    parts = []
    for f in coll.fields:
        # The human label is included because it is what a person would SAY.
        # Withholding it makes the teacher invent field names that no prompt
        # would ever use, and those round-trip as mismatches.
        bit = f"{f.name}:{f.ftype}"
        if f.label and f.label != f.name:
            bit += f' (called "{f.label}")'
        if f.choices:
            bit += " one of [" + ", ".join(str(c) for c in f.choices[:8]) + "]"
        parts.append(bit)
    return (
        f"database: {dom.name}\ncollection: {coll.name} "
        f"({coll.singular}/{coll.plural})\nfields:\n  " + "\n  ".join(parts)
    )


_PARAPHRASE_SYSTEM = """You write the kinds of questions real people type into a database search box.

You will be shown a collection's schema and one NQL query. Write DIFFERENT ways a person might ask for exactly that data.

Rules:
- Output ONE question per line. Nothing else — no numbering, no quotes, no commentary.
- Every question must be answerable by that exact query. Do not add or drop a filter, a sort, or a limit.
- Vary hard: verb ("show me" / "list" / "pull up" / none), operator wording ("over" / "more than" / "above"), field naming (raw field vs human phrasing), clause order, register (terse vs polite), and casing.
- Some should be sloppy the way real input is: lowercase, no punctuation, contractions.
- Never mention NQL, SQL, syntax, fields you were not given, or the word "query"."""

_TRANSLATE_SYSTEM = """You convert a natural-language request into a single NQL query.

NQL grammar (keywords case-insensitive):
    FROM <collection>
      [ AS OF <seq> ]
      [ VALID AS OF <date> ]
      [ WHERE <field> <op> <value> (AND <field> <op> <value>)* ]
      [ SEARCH "<text>" ]
      [ ORDER BY <field> [ASC|DESC] ]
      [ TRAVERSE <relation> ]
      [ LIMIT <n> ]
    op    := = | != | < | <= | > | >=
    value := number | "string" | 'string' | true | false | null

Output ONLY the query on one line. No explanation, no code fences, no trailing punctuation."""


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _clean_nql(text: str) -> str:
    """Strip the wrappers a chat model adds even when told not to."""
    t = text.strip()
    t = _FENCE.sub("", t).strip()
    # Take the first line that starts a query. A model that explains anyway
    # usually still puts the query on its own line.
    for line in t.splitlines():
        s = line.strip().strip("`").rstrip(";")
        if s.lower().startswith("from "):
            return s
    return t.splitlines()[0].strip().strip("`").rstrip(";") if t else ""


def _clean_prompts(text: str, want: int) -> List[str]:
    """One prompt per line, with the model's decorations removed."""
    out: List[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        # Numbered / bulleted lists, despite the instruction not to.
        s = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", s)
        s = s.strip().strip('"').strip("'").strip()
        # A model narrating ("Here are 8 ways:") is not a prompt.
        if not s or s.endswith(":") or len(s) < 4:
            continue
        if s.lower().startswith(("here are", "sure", "certainly")):
            continue
        out.append(s)
        if len(out) >= want:
            break
    return out


# ---------------------------------------------------------------------- result


@dataclass
class ForgeStats:
    """What a run actually did. Every discard is counted and named."""

    plans: int = 0
    prompts_offered: int = 0
    verified: int = 0
    mismatch: int = 0          # round-tripped to a DIFFERENT plan
    unparseable: int = 0       # teacher's NQL did not parse at all
    teacher_errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def rate(self) -> float:
        return self.verified / self.prompts_offered if self.prompts_offered else 0.0


def forge_plan(
    cfg: TeacherCfg,
    plan: Dict[str, Any],
    dom: Domain,
    coll: Collection,
    variants: int,
    stats: ForgeStats,
) -> List[Dict[str, Any]]:
    """Paraphrase one plan and keep only the prompts that round-trip.

    Two teacher calls per plan regardless of `variants`: one to write all the
    English, one to translate all of it back. Per-prompt calls would be
    `variants`x the cost for the same information.
    """
    gold_nql = render_nql(plan)
    gold_cf = canonical(plan)
    schema = _schema_block(dom, coll)

    ask = (
        f"{schema}\n\nNQL query:\n{gold_nql}\n\n"
        f"Write {variants} different questions a person might type to get exactly this."
    )
    try:
        text, para_reasoning = _post(cfg, _PARAPHRASE_SYSTEM, ask, cfg.paraphrase_temperature)
    except TeacherError as e:
        with stats.lock:
            stats.teacher_errors += 1
        print(f"  teacher failed on paraphrase: {e}", file=sys.stderr)
        return []

    prompts = _clean_prompts(text, variants)
    if not prompts:
        with stats.lock:
            stats.teacher_errors += 1
        print(
            f"  teacher returned no usable prompts for {gold_nql!r} "
            f"({len(text)} chars of output)",
            file=sys.stderr,
        )
        return []

    # Round trip: the teacher sees ONLY the English. No plan, no gold NQL, no
    # sight of what it wrote a moment ago beyond the prompt itself.
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    back_ask = (
        f"{schema}\n\nConvert each request to one NQL query. "
        f"Output exactly {len(prompts)} lines, in order, each `N. <query>`.\n\n{numbered}"
    )
    try:
        back, trans_reasoning = _post(cfg, _TRANSLATE_SYSTEM, back_ask, cfg.translate_temperature)
    except TeacherError as e:
        with stats.lock:
            stats.teacher_errors += 1
        print(f"  teacher failed on translate-back: {e}", file=sys.stderr)
        return []

    # Map "N. query" back to its prompt. A line the teacher failed to number is
    # matched positionally as a fallback, but never guessed across indices.
    recovered: Dict[int, str] = {}
    positional: List[str] = []
    for line in back.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^\s*(\d+)\s*[.)]\s*(.+)$", s)
        if m:
            recovered[int(m.group(1)) - 1] = _clean_nql(m.group(2))
        elif s.lower().startswith("from "):
            positional.append(_clean_nql(s))

    rows: List[Dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        nql = recovered.get(i)
        if nql is None and len(positional) == len(prompts):
            nql = positional[i]
        with stats.lock:
            stats.prompts_offered += 1
        if not nql:
            with stats.lock:
                stats.unparseable += 1
            continue
        try:
            got = canonical(parse_nql(nql))
        except Exception:
            # The teacher's syntax is not the corpus's problem — the prompt is
            # discarded, and the reason is counted rather than lumped in with
            # semantic mismatches.
            with stats.lock:
                stats.unparseable += 1
            continue
        if got != gold_cf:
            with stats.lock:
                stats.mismatch += 1
            continue

        with stats.lock:
            stats.verified += 1
        rows.append(
            {
                "prompt": prompt,
                # The LABEL is always the generator's canonical NQL, never the
                # teacher's. The teacher invents English and judges itself; it
                # does not get to write syntax into the corpus.
                "nql": gold_nql,
                "plan": gold_cf,
                "domain": dom.name,
                "coll": coll.name,
                "clauses": clauses_present(plan),
                "source": "teacher",
                "teacher_model": cfg.model,
                # Kept because it is the expensive part of the run and it is
                # what a future student could learn to imitate.
                "reasoning": trans_reasoning or para_reasoning or None,
            }
        )
    return rows


def forge(
    n_plans: int,
    variants: int = 8,
    seed: int = 1337,
    cfg: Optional[TeacherCfg] = None,
    out_path: Optional[Path] = None,
    workers: int = 4,
) -> ForgeStats:
    """Forge a teacher-written, round-trip-verified corpus.

    Writes JSONL rows shaped exactly like `cast.dataset` emits, so the existing
    tokenizer and trainer consume them unchanged.
    """
    cfg = cfg or TeacherCfg()
    stats = ForgeStats()
    rng = random.Random(seed)

    # Sampled up front and on ONE rng, so a run is reproducible from its seed
    # no matter how the workers interleave.
    plans = [sample_plan(rng) for _ in range(n_plans)]
    # De-duplicate by canonical form: the same plan twice is the same lesson
    # twice, and it would also split across train/eval later.
    seen: set = set()
    unique: List[Tuple[Dict[str, Any], Domain, Collection]] = []
    for p, d, c in plans:
        k = canonical(p)
        if k in seen:
            continue
        seen.add(k)
        unique.append((p, d, c))

    print(
        f"forge: {len(unique)} unique plans ({n_plans} sampled) x up to {variants} prompts "
        f"· teacher {cfg.model} at {cfg.endpoint} · {workers} worker(s)",
        file=sys.stderr,
    )

    fh = out_path.open("w") if out_path else None
    write_lock = threading.Lock()
    started = time.time()
    idx = threading.Semaphore(0)  # placeholder to keep imports honest
    del idx

    next_i = [0]
    take_lock = threading.Lock()

    def work() -> None:
        while True:
            with take_lock:
                i = next_i[0]
                next_i[0] += 1
            if i >= len(unique):
                return
            plan, dom, coll = unique[i]
            rows = forge_plan(cfg, plan, dom, coll, variants, stats)
            with stats.lock:
                stats.plans += 1
                done = stats.plans
                ver = stats.verified
                off = stats.prompts_offered
            if fh is not None and rows:
                with write_lock:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                    fh.flush()
            el = max(time.time() - started, 0.001)
            print(
                f"  [{done}/{len(unique)}] +{len(rows):>2} verified · "
                f"{ver}/{off} kept ({(ver/off*100 if off else 0):.1f}%) · "
                f"{done/el*60:.1f} plans/min",
                file=sys.stderr,
            )

    threads = [threading.Thread(target=work, daemon=True) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if fh is not None:
        fh.close()

    el = time.time() - started
    print(
        "\nforge complete\n"
        f"  plans            {stats.plans}\n"
        f"  prompts offered  {stats.prompts_offered}\n"
        f"  VERIFIED         {stats.verified}  ({stats.rate()*100:.1f}%)\n"
        f"  mismatch         {stats.mismatch}  (round-tripped to a different plan — ambiguous English)\n"
        f"  unparseable      {stats.unparseable}  (teacher NQL did not parse)\n"
        f"  teacher errors   {stats.teacher_errors}\n"
        f"  wall clock       {el:.1f}s\n"
        + (f"  rows -> {out_path}\n" if out_path else ""),
        file=sys.stderr,
    )
    return stats
