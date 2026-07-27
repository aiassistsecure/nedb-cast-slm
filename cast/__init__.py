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
from .inference import Cast  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Cast"]
