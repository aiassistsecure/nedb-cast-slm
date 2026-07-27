"""
nedb-cast-slm — a small language model that casts short prompts into NEDB query plans.

Quick start
-----------

    from cast import Cast

    caster = Cast.from_pretrained("runs/v1")

    caster.cast("paid orders over $99, newest first")
    # 'FROM orders WHERE status = "paid" AND total > 99 ORDER BY placed_at DESC'

    caster.plan("top 5 stylists in winter park")
    # {'from': 'stylists', 'where': [['city', '=', 'winter park']], 'limit': 5, ...}

The model emits NQL text; `plan()` additionally parses it with the real engine
parser (`nedb.query.parse_nql`), so an invalid generation raises rather than
silently handing back something the engine cannot run.
"""
__version__ = "10.30.92"
__all__ = ["Cast"]


def __getattr__(name):
    """Import Cast lazily so `import cast` works without torch installed.

    torch is an optional extra (`pip install "nedb-cast-slm[torch]"`). The CLI's
    generate/tokenizer/lineage commands and the NQL parser need no torch at all,
    so importing it eagerly would break them on any platform lacking a wheel —
    new Python releases, MSYS2/MINGW Python, 32-bit. Reported from a real
    MINGW64 install.
    """
    if name == "Cast":
        try:
            from .inference import Cast
        except ImportError as e:
            if "torch" in str(e).lower():
                raise ImportError(
                    "nedb-cast-slm inference needs PyTorch, which is an "
                    "optional extra.\n\n"
                    "  pip install \"nedb-cast-slm[torch]\"\n\n"
                    "If pip reports \"no matching distribution\" for torch, your "
                    "Python has no torch wheel. Check with `python -VV`:\n"
                    "  * torch supports CPython 3.9-3.13 on win_amd64 / "
                    "manylinux / macOS\n"
                    "  * MSYS2 / MINGW Python is NOT supported by torch — use a "
                    "python.org Windows build\n"
                    "  * try: pip install torch "
                    "--index-url https://download.pytorch.org/whl/cpu\n\n"
                    "The pure-Rust core (crates.io: nedb-cast-core) needs no "
                    "Python or torch at all."
                ) from e
            raise
        return Cast
    raise AttributeError("module 'cast' has no attribute %r" % name)
