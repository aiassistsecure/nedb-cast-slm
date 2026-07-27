#!/usr/bin/env python3
"""
smoketest.py — verify an INSTALLED nedb-cast-slm on a real machine.

Run it anywhere the package is pip-installed. No repo clone, no pytest, no
network except the one-time weight download.

    pip install nedb-cast-slm
    python smoketest.py

Flags:
    --offline      skip anything that needs the network (env + parser only)
    --model PATH   use a local model.cast / ckpt.pt instead of downloading
    --quick        fewer prompts, skip the timing benchmark
    --verbose      print every prediction, not just failures

Exit code is 0 only if every executed check passed, so it drops straight into
CI or a scheduled job.

Design notes, because this file is meant to be trusted:
  * ASCII only. Windows consoles default to cp1252 and a stray box-drawing
    character turns a passing test into a UnicodeEncodeError.
  * Every check is independent and reports PASS / FAIL / SKIP separately. A
    missing optional piece (weights not published yet) SKIPS rather than fails,
    so the exit code stays meaningful.
  * Correctness is graded by the REAL engine parser on a plan-dict basis, never
    by string match. `FROM orders WHERE total > 99` and `from orders where
    total>99` are the same query and both count.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import traceback

# Prefer an installed `cast`, but fall back to a source checkout so this file
# also works when run straight out of a cloned repo (scripts/smoketest.py).
try:
    import cast as _probe  # noqa: F401
except ImportError:
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(_repo, "cast")):
        sys.path.insert(0, _repo)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = []

# ANSI colour, but only on a real terminal. Windows 10+ consoles understand
# these; MINGW/Git-Bash does too. Piped output stays plain so logs are clean.
_TTY = sys.stdout.isatty() and os.environ.get("TERM") != "dumb" \
    and not os.environ.get("NO_COLOR")


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def green(t): return _c("32;1", t)
def red(t): return _c("31;1", t)
def yellow(t): return _c("33;1", t)
def dim(t): return _c("2", t)
def bold(t): return _c("1", t)
def cyan(t): return _c("36;1", t)


BANNER = r"""
                 888
  .d8888b .d88b. 888 .d8888b  888888
 d88P"   d88""88b 888 88K     888
 888     888  888 888 "Y8888b. 888
 Y88b.   Y88..88P 888      X88 Y88b.
  "Y8888P "Y88P"  888  88888P'  "Y888
"""


_MARK = {PASS: "OK  ", FAIL: "FAIL", SKIP: "SKIP"}
_PAINT = {PASS: green, FAIL: red, SKIP: yellow}


def record(name, status, detail=""):
    results.append((name, status, detail))
    tag = _PAINT[status]("[%s]" % _MARK[status])
    line = "%s %s" % (tag, name)
    if detail:
        line += " " + dim("-- " + detail)
    print(line, flush=True)


_step = [0]


def section(title):
    _step[0] += 1
    print("")
    print(cyan("=" * 70))
    print(cyan(" %d. %s" % (_step[0], title)))
    print(cyan("=" * 70), flush=True)


# --------------------------------------------------------------- environment
def check_environment():
    section("Environment")
    record("python >= 3.9", PASS if sys.version_info >= (3, 9) else FAIL,
           "%s on %s %s" % (platform.python_version(), platform.system(),
                            platform.machine()))
    try:
        import torch
        record("torch importable", PASS,
               "%s, threads=%d" % (torch.__version__, torch.get_num_threads()))
    except Exception as e:
        record("torch importable", FAIL, repr(e))
        return False
    try:
        import cast
        record("cast importable", PASS, "version %s" %
               getattr(cast, "__version__", "unknown"))
    except Exception as e:
        record("cast importable", FAIL, repr(e))
        return False
    try:
        import nedb
        from nedb.query import parse_nql  # noqa: F401
        record("nedb-engine importable", PASS,
               "version %s" % getattr(nedb, "__version__", "unknown"))
    except Exception as e:
        record("nedb-engine importable", FAIL, repr(e))
        return False
    return True


# --------------------------------------------------------------- parser only
def check_parser_offline():
    """The grader itself, with no model involved. If this fails, nothing
    downstream can be trusted."""
    section("NQL parser (no model needed)")
    from nedb.query import parse_nql

    cases = [
        ('FROM orders WHERE total > 99 ORDER BY placed_at DESC',
         {"from": "orders"}),
        ('FROM sales VALID AS OF "2024-02-15" GROUP BY region SUM total',
         {"group_by": "region"}),
        ('FROM docs SEARCH "invoice" TRACE caused_by REVERSE LIMIT 5',
         {"trace": "caused_by", "trace_reverse": True}),
    ]
    ok = True
    for nql, expect in cases:
        try:
            plan = parse_nql(nql)
            bad = [k for k, v in expect.items() if plan.get(k) != v]
            if bad:
                record("parse %s" % nql[:40], FAIL, "wrong %s" % bad)
                ok = False
            else:
                record("parse %s" % nql[:40], PASS)
        except Exception as e:
            record("parse %s" % nql[:40], FAIL, repr(e))
            ok = False

    # garbage must be rejected -- a parser that accepts anything grades nothing
    try:
        parse_nql("FROM x WHERE bogus ~~ 3")
        record("rejects invalid NQL", FAIL, "accepted garbage")
        ok = False
    except Exception:
        record("rejects invalid NQL", PASS)
    return ok


# --------------------------------------------------------------- load model
def load_model(args):
    section("Load the model")
    from cast import Cast

    if args.model:
        try:
            m = Cast.from_pretrained(args.model)
            record("load from %s" % args.model, PASS, repr(m))
            return m
        except Exception as e:
            record("load from %s" % args.model, FAIL, repr(e))
            return None

    if args.offline:
        record("download weights", SKIP, "--offline")
        return None

    try:
        t0 = time.time()
        m = Cast.pretrained()
        record("Cast.pretrained()", PASS,
               "%s in %.1fs" % (repr(m), time.time() - t0))
        return m
    except Exception as e:
        # Assets may not be attached to the release yet -- that is a SKIP, not
        # a failure of the installed package.
        msg = str(e)
        if "404" in msg or "could not download" in msg:
            record("Cast.pretrained()", SKIP,
                   "release assets not published yet: %s" % msg[:90])
        else:
            record("Cast.pretrained()", FAIL, repr(e))
        return None


# --------------------------------------------------------------- accuracy
GOLDEN = [
    # (prompt, required plan fields). Deliberately hand-written, NOT drawn from
    # the training corpus -- these are the phrasings a real user would type.
    ("top 5 stylists in winter park",
     {"from": "stylists", "limit": 5}),
    ("invoices that are overdue limit 10",
     {"from": "invoices", "limit": 10}),
    ("show me all services",
     {"from": "services"}),
    ("what caused these checkpoints",
     {"from": "checkpoints", "trace": "caused_by"}),
    ("appointments for marisa",
     {"from": "appointments"}),
    ("blocks sorted by difficulty descending",
     {"from": "blocks"}),
    ("memories with importance 5 grouped by category count",
     {"from": "memories", "group_by": "category"}),
    ("runs as of seq 500",
     {"from": "runs", "as_of": 500}),
    ("events mentioning timeout",
     {"from": "events", "search": "timeout"}),
    ("deploys that are green",
     {"from": "deploys"}),
]

QUICK = GOLDEN[:4]


def check_predictions(m, args):
    section("Predictions (graded by the real parser)")
    from nedb.query import parse_nql

    cases = QUICK if args.quick else GOLDEN
    valid = 0
    field_ok = 0
    for prompt, expect in cases:
        try:
            nql = m.cast(prompt)
        except Exception as e:
            record('cast "%s"' % prompt[:34], FAIL, repr(e))
            continue
        try:
            plan = parse_nql(nql)
            valid += 1
        except Exception as e:
            record('cast "%s"' % prompt[:34], FAIL,
                   "INVALID NQL: %s (%s)" % (nql[:60], str(e)[:40]))
            continue
        wrong = [k for k, v in expect.items() if plan.get(k) != v]
        if wrong:
            record('cast "%s"' % prompt[:34], FAIL,
                   "got %s | wrong fields %s" % (nql[:60], wrong))
        else:
            field_ok += 1
            record('cast "%s"' % prompt[:34], PASS,
                   nql[:64] if args.verbose else "")

    n = len(cases)
    record("valid NQL rate", PASS if valid == n else FAIL,
           "%d/%d" % (valid, n))
    # The model is ~92%% exact-match on in-distribution eval, so a couple of
    # misses on hand-written prompts is expected, not a regression. Anything
    # below 70%% means something is actually wrong with the install.
    thresh = max(1, int(0.7 * n))
    record("key fields correct (>=70%)", PASS if field_ok >= thresh else FAIL,
           "%d/%d" % (field_ok, n))
    return valid == n and field_ok >= thresh


# --------------------------------------------------------------- invariants
def check_invariants(m):
    section("Invariants that caught real bugs")
    ok = True

    # Batched decoding must equal single decoding. The model uses learned
    # positional embeddings with no pad mask, so padding a mixed-length batch
    # corrupts short sequences. That bug cost 55 accuracy points once and looked
    # exactly like a model capacity limit.
    prompts = [p for p, _ in GOLDEN]
    try:
        batched = m.cast_many(prompts)
        single = [m.cast_many([p])[0] for p in prompts]
        if batched == single:
            record("batched == single decoding", PASS,
                   "%d prompts" % len(prompts))
        else:
            diff = [i for i, (a, b) in enumerate(zip(batched, single)) if a != b]
            record("batched == single decoding", FAIL,
                   "%d of %d differ (idx %s)" % (len(diff), len(prompts), diff[:4]))
            ok = False
    except Exception as e:
        record("batched == single decoding", FAIL, repr(e))
        ok = False

    # Determinism: greedy decoding must be reproducible run to run.
    try:
        a = m.cast("top 5 stylists in winter park")
        b = m.cast("top 5 stylists in winter park")
        record("greedy decode is deterministic", PASS if a == b else FAIL,
               "" if a == b else "%r != %r" % (a, b))
        ok = ok and (a == b)
    except Exception as e:
        record("greedy decode is deterministic", FAIL, repr(e))
        ok = False

    # try_plan must never raise; plan() must raise on invalid output.
    try:
        out = m.try_plan("show me orders")
        has = isinstance(out, dict) and "ok" in out and "nql" in out
        record("try_plan returns a report", PASS if has else FAIL,
               "keys %s" % sorted(out) if isinstance(out, dict) else repr(out))
        ok = ok and has
    except Exception as e:
        record("try_plan returns a report", FAIL, repr(e))
        ok = False

    # Dates must survive the tokenizer round trip (regression: "2026-10-18"
    # decoded as "2026 -10 -18" because the digit merge dropped '-').
    try:
        q = 'FROM products VALID AS OF "2026-10-18" ORDER BY title ASC'
        back = m.tok.decode(m.tok.encode(q))
        record("dates survive tokenizer", PASS if back == q else FAIL,
               "" if back == q else "got %r" % back)
        ok = ok and (back == q)
    except Exception as e:
        record("dates survive tokenizer", FAIL, repr(e))
        ok = False

    # Keywords inside quoted literals must NOT be uppercased (regression:
    # SEARCH "rate limit" became SEARCH "rate LIMIT", corrupting user data).
    try:
        q = 'FROM events SEARCH "rate limit" LIMIT 5'
        back = m.tok.decode(m.tok.encode(q))
        good = '"rate limit"' in back
        record("quoted literals preserved", PASS if good else FAIL,
               "" if good else "got %r" % back)
        ok = ok and good
    except Exception as e:
        record("quoted literals preserved", FAIL, repr(e))
        ok = False
    return ok


# --------------------------------------------------------------- end to end
def check_live_engine(m):
    """Cast a prompt and RUN it against a real NEDB database."""
    section("End to end against a live NEDB database")
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cast-smoke-")
    try:
        from nedb import NEDB
        db = NEDB(tmp)
        db.put("orders", "o1", {"total": 150.0, "status": "paid"})
        db.put("orders", "o2", {"total": 40.0, "status": "pending"})
        db.put("orders", "o3", {"total": 900.0, "status": "paid"})
        record("wrote 3 docs to a fresh db", PASS, "verify=%s" % db.verify())

        nql = m.cast("orders over 100")
        rows = db.execute(m.plan("orders over 100"))
        record("cast + execute", PASS, "%s -> %d rows" % (nql[:52], len(rows)))

        # provenance primitives the project is actually about
        ds = db.put("datasets", "d1", {"name": "smoke"})
        run = db.put("runs", "r1", {"steps": 1}, caused_by=[ds["_seq"]])
        lineage = db.query("FROM runs TRACE caused_by")
        good = len(lineage) >= 1
        record("TRACE caused_by works", PASS if good else FAIL,
               "%d ancestor row(s)" % len(lineage))
        record("db.verify() after writes", PASS if db.verify() else FAIL)
        return good and db.verify()
    except Exception as e:
        record("live engine round trip", FAIL, repr(e))
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------- showcase
SHOWCASE = [
    "top 5 stylists in winter park",
    "invoices that are overdue limit 10",
    "paid orders sorted by total descending",
    "what caused these checkpoints",
    "events mentioning timeout in the last batch",
    "memories with importance 5 grouped by category count",
    "runs as of seq 500",
    "blocks above height 4000 sorted by difficulty descending",
]


def showcase(m):
    """Not a test. This is the part where you see what it does."""
    section("Watch it work")
    from nedb.query import parse_nql
    print(dim("  3.34M parameters. Trained on two CPU cores in 40.9 minutes."))
    print(dim("  No GPU, no API key, no network call. Just a sentence in, a plan out."))
    print("")
    for prompt in SHOWCASE:
        t0 = time.time()
        nql = m.cast(prompt)
        ms = (time.time() - t0) * 1000
        try:
            parse_nql(nql)
            mark = green("valid")
        except Exception:
            mark = red("INVALID")
        print("  " + bold('"%s"' % prompt))
        print("      " + cyan(nql))
        print("      " + dim("%s  %.0f ms" % (mark, ms)))
        print("")


# --------------------------------------------------------------- speed
def check_speed(m):
    section("Speed (informational -- no pass/fail threshold)")
    # Deliberately NO threshold. Hardware varies; a slow machine is not a
    # broken install. Numbers are printed so they can be compared over time.
    prompts = [p for p, _ in GOLDEN]
    try:
        m.cast(prompts[0])  # warm up
        t0 = time.time()
        for p in prompts:
            m.cast(p)
        single = (time.time() - t0) / len(prompts)

        t0 = time.time()
        m.cast_many(prompts)
        batch = (time.time() - t0) / len(prompts)
        record("latency", PASS,
               "single %.0f ms/prompt | batched %.0f ms/prompt" %
               (single * 1000, batch * 1000))
        return True
    except Exception as e:
        record("latency", FAIL, repr(e))
        return False


# --------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip anything needing the network")
    ap.add_argument("--model", default=None,
                    help="path to a local model.cast or ckpt.pt / run dir")
    ap.add_argument("--quick", action="store_true", help="fewer checks")
    ap.add_argument("--verbose", action="store_true",
                    help="print every prediction")
    args = ap.parse_args(argv)

    if _TTY:
        print(cyan(BANNER))
    print(bold("  nedb-cast-slm smoke test"))
    try:
        import cast as _c
        ver = getattr(_c, "__version__", "?")
    except Exception:
        ver = "?"
    print(dim("  package %s  |  python %s  |  %s %s" % (
        ver, platform.python_version(), platform.system(), platform.machine())))

    if not check_environment():
        print("\nEnvironment is not usable; stopping here.")
        return summarize()

    check_parser_offline()

    m = load_model(args)
    if m is None:
        section("Model-dependent checks")
        record("predictions", SKIP, "no model loaded")
        record("invariants", SKIP, "no model loaded")
        record("end to end", SKIP, "no model loaded")
        return summarize()

    check_predictions(m, args)
    check_invariants(m)
    check_live_engine(m)
    if not args.quick:
        showcase(m)
    if not args.quick:
        check_speed(m)
    return summarize()


def summarize():
    section("Summary")
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    total = n_pass + n_fail + n_skip

    # a simple bar, honest about proportions
    if total:
        width = 46
        gp = max(0, round(width * n_pass / total))
        gf = max(0, round(width * n_fail / total))
        gs = max(0, width - gp - gf)
        print("  " + green("#" * gp) + red("!" * gf) + yellow("-" * gs))
    print("")
    print("  " + green("passed  %3d" % n_pass))
    print("  " + (red("failed  %3d" % n_fail) if n_fail
                  else dim("failed    0")))
    print("  " + (yellow("skipped %3d" % n_skip) if n_skip
                  else dim("skipped   0")))

    if n_fail:
        print("")
        print(red("  FAILURES"))
        for name, s, detail in results:
            if s == FAIL:
                print("    - %s%s" % (name, (" -- " + detail) if detail else ""))
    print("")
    if n_fail:
        print(red(bold("  RESULT: FAIL")))
    else:
        print(green(bold("  RESULT: OK")))
        print(dim("  Everything that ran, passed."))
    print("")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
