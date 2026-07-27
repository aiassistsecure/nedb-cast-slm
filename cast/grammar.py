"""
cast.grammar — the synthetic schema universe the model learns to query.

We deliberately train across MANY domains (shop, salon, chain, agent, crm, ops)
so the model learns the *shape* of a query rather than memorising one schema's
field names. A model that only works on `orders.total` is useless in Studio,
where every user has their own collections.

Field types drive both value generation and paraphrasing: knowing `total` is
money lets us say "over $99.50", and knowing `age` is an int lets us say
"older than 30".
"""
from __future__ import annotations

import random
from typing import Any, Dict, List

# ---------------------------------------------------------------- field types

INT = "int"
FLOAT = "float"
MONEY = "money"
ENUM = "enum"
TEXT = "text"
NAME = "name"
DATE = "date"
BOOL = "bool"

# Words that are NQL keywords — never usable as bare-word values, and avoided
# as field/collection names so generated queries stay unambiguous.
RESERVED = {
    "from", "as", "of", "where", "and", "search", "order", "by",
    "asc", "desc", "traverse", "trace", "reverse", "limit",
    "valid", "true", "false", "null", "group", "count", "sum", "avg", "min", "max",
}


class Field:
    def __init__(self, name: str, ftype: str, choices: List[str] | None = None,
                 lo: float = 0, hi: float = 100, label: str | None = None):
        assert name.lower() not in RESERVED, f"field {name} collides with NQL keyword"
        self.name = name
        self.ftype = ftype
        self.choices = choices or []
        self.lo = lo
        self.hi = hi
        # human phrase used in paraphrases, e.g. "order total" for `total`
        self.label = label or name.replace("_", " ")

    def sample_value(self, rng: random.Random) -> Any:
        t = self.ftype
        if t == INT:
            return rng.randint(int(self.lo), int(self.hi))
        if t == FLOAT:
            return round(rng.uniform(self.lo, self.hi), rng.choice([1, 2]))
        if t == MONEY:
            return round(rng.uniform(self.lo, self.hi), rng.choice([0, 1, 2]))
        if t in (ENUM, TEXT, NAME):
            return rng.choice(self.choices)
        if t == DATE:
            y = rng.randint(2023, 2026)
            m = rng.randint(1, 12)
            d = rng.randint(1, 28)
            return f"{y:04d}-{m:02d}-{d:02d}"
        if t == BOOL:
            return rng.choice([True, False])
        raise ValueError(t)

    def is_numeric(self) -> bool:
        return self.ftype in (INT, FLOAT, MONEY)

    def is_ordered(self) -> bool:
        """Can this field take <, >, <=, >= sensibly?"""
        return self.ftype in (INT, FLOAT, MONEY, DATE)


class Collection:
    def __init__(self, name: str, fields: List[Field], singular: str, plural: str,
                 searchable: List[str] | None = None,
                 relations: List[str] | None = None):
        assert name.lower() not in RESERVED
        self.name = name
        self.fields = fields
        self.singular = singular
        self.plural = plural
        # realistic free-text search terms for this collection
        self.searchable = searchable or []
        # relations valid FROM this collection. Domain-wide sampling produced
        # nonsense like `FROM services TRAVERSE booked_by`; scoping the edge to
        # the collection keeps generated queries semantically sane, not just
        # syntactically valid.
        self.relations = relations or []

    def field(self, rng: random.Random, pred=None) -> Field:
        pool = [f for f in self.fields if pred(f)] if pred else self.fields
        return rng.choice(pool)


class Domain:
    def __init__(self, name: str, collections: List[Collection], relations: List[str]):
        self.name = name
        self.collections = collections
        self.relations = relations


# ---------------------------------------------------------------- value pools

STATUS = ["paid", "pending", "refunded", "shipped", "cancelled", "failed"]
TIERS = ["free", "pro", "enterprise", "trial"]
FIRST = ["marisa", "alex", "jordan", "priya", "sam", "chen", "dana", "luis",
         "nina", "omar", "riley", "tess"]
CITIES = ["winter park", "orlando", "maitland", "oviedo", "apopka", "sanford"]
SERVICES = ["balayage", "color", "cut", "gloss", "treatment", "blowout"]
CHAINS = ["confirmed", "orphan", "pending", "stale"]
AGENTS = ["vex", "porter", "scribe", "trader", "vendor"]
STAGES = ["lead", "qualified", "proposal", "won", "lost"]
SEVERITY = ["info", "warn", "error", "fatal"]


def _f(*a, **kw) -> Field:
    return Field(*a, **kw)


DOMAINS: List[Domain] = [
    Domain("shop", [
        Collection("orders", [
            _f("total", MONEY, lo=5, hi=900, label="order total"),
            _f("status", ENUM, STATUS),
            _f("quantity", INT, lo=1, hi=40),
            _f("customer", NAME, FIRST),
            _f("placed_at", DATE, label="placed date"),
            _f("discounted", BOOL, label="discounted"),
        ], "order", "orders", ["invoice", "refund request", "gift card", "late delivery"],
           relations=["purchased", "belongs_to"]),
        Collection("products", [
            _f("price", MONEY, lo=2, hi=500),
            _f("stock", INT, lo=0, hi=1000, label="stock level"),
            _f("category", ENUM, ["shampoo", "tools", "color", "styling"]),
            _f("rating", FLOAT, lo=1, hi=5, label="rating"),
            _f("title", TEXT, ["repair mask", "clay pomade", "gloss kit"]),
        ], "product", "products", ["sulfate free", "travel size", "best seller"],
           relations=["reviewed"]),
        Collection("customers", [
            _f("age", INT, lo=18, hi=80),
            _f("city", ENUM, CITIES),
            _f("tier", ENUM, TIERS),
            _f("lifetime_value", MONEY, lo=0, hi=5000, label="lifetime value"),
            _f("name", NAME, FIRST),
        ], "customer", "customers", ["vip", "complaint", "referral"],
           relations=["purchased", "reviewed"]),
    ], ["purchased", "reviewed", "belongs_to"]),

    Domain("salon", [
        Collection("appointments", [
            _f("price", MONEY, lo=25, hi=400),
            _f("service", ENUM, SERVICES),
            _f("stylist", NAME, FIRST),
            _f("booked_at", DATE, label="booking date"),
            _f("duration", INT, lo=15, hi=240, label="duration in minutes"),
            _f("confirmed", BOOL),
        ], "appointment", "appointments", ["first visit", "consultation", "no show"],
           relations=["booked_by", "performed_by"]),
        Collection("stylists", [
            _f("level", INT, lo=1, hi=5, label="seniority level"),
            _f("city", ENUM, CITIES),
            _f("name", NAME, FIRST),
            _f("rating", FLOAT, lo=1, hi=5),
        ], "stylist", "stylists", ["master colorist", "aveda certified"],
           relations=["performed_by", "offers"]),
        Collection("services", [
            _f("price", MONEY, lo=20, hi=350),
            _f("category", ENUM, ["color", "cut", "treatment", "styling"]),
            _f("duration", INT, lo=15, hi=180, label="duration"),
        ], "service", "services", ["balayage", "keratin", "toner"],
           relations=["offers"]),
    ], ["booked_by", "performed_by", "offers"]),

    Domain("chain", [
        Collection("blocks", [
            _f("height", INT, lo=1, hi=900000, label="block height"),
            _f("difficulty", FLOAT, lo=1, hi=99999, label="difficulty"),
            _f("txcount", INT, lo=0, hi=5000, label="transaction count"),
            _f("state", ENUM, CHAINS),
            _f("mined_at", DATE, label="mined date"),
        ], "block", "blocks", ["coinbase", "orphaned", "reorg"],
           relations=["mined_by"]),
        Collection("transactions", [
            _f("amount", MONEY, lo=0, hi=10000),
            _f("fee", MONEY, lo=0, hi=5),
            _f("confirmations", INT, lo=0, hi=1000),
            _f("sender", NAME, FIRST),
        ], "transaction", "transactions", ["op return", "multisig", "dust"],
           relations=["spent_in", "funds"]),
        Collection("wallets", [
            _f("balance", MONEY, lo=0, hi=50000),
            _f("label", TEXT, ["cold storage", "governance", "faucet"]),
            _f("watch_only", BOOL, label="watch only"),
        ], "wallet", "wallets", ["governance", "treasury"],
           relations=["funds"]),
    ], ["spent_in", "mined_by", "funds"]),

    Domain("agent", [
        Collection("checkpoints", [
            _f("step", INT, lo=100, hi=200000, label="training step"),
            _f("loss", FLOAT, lo=0, hi=6, label="loss"),
            _f("params", INT, lo=1, hi=400, label="parameter count in millions"),
        ], "checkpoint", "checkpoints", ["best", "diverged", "resumed"],
           relations=["produced_by", "cites"]),
        Collection("runs", [
            _f("duration", INT, lo=1, hi=600, label="duration in minutes"),
            _f("status", ENUM, ["running", "done", "failed", "killed"]),
            _f("owner", NAME, AGENTS),
            _f("started_at", DATE, label="start date"),
        ], "run", "runs", ["overnight", "smoke test", "sweep"],
           relations=["produced_by"]),
        Collection("memories", [
            _f("importance", INT, lo=1, hi=5),
            _f("category", ENUM, ["preference", "project", "people", "domain"]),
            _f("agent", NAME, AGENTS),
        ], "memory", "memories", ["release flow", "guardrail", "handoff"],
           relations=["recalled_by", "cites"]),
    ], ["produced_by", "recalled_by", "cites"]),

    Domain("crm", [
        Collection("leads", [
            _f("score", INT, lo=0, hi=100, label="lead score"),
            _f("stage", ENUM, STAGES),
            _f("owner", NAME, FIRST),
            _f("value", MONEY, lo=100, hi=90000, label="deal value"),
            _f("city", ENUM, CITIES),
        ], "lead", "leads", ["inbound", "referral", "cold outreach"],
           relations=["owned_by"]),
        Collection("invoices", [
            _f("amount", MONEY, lo=50, hi=25000),
            _f("status", ENUM, ["draft", "sent", "paid", "overdue"]),
            _f("issued_at", DATE, label="issue date"),
            _f("client", NAME, FIRST),
        ], "invoice", "invoices", ["net 30", "retainer", "disputed"],
           relations=["billed_to", "owned_by"]),
    ], ["owned_by", "billed_to"]),

    Domain("ops", [
        Collection("events", [
            _f("severity", ENUM, SEVERITY),
            _f("latency", INT, lo=1, hi=5000, label="latency in ms"),
            _f("service", ENUM, ["api", "worker", "web", "daemon"]),
            _f("occurred_at", DATE, label="event date"),
        ], "event", "events", ["timeout", "oom", "cert expiry", "rate limit"],
           relations=["triggered_by"]),
        Collection("deploys", [
            _f("version", TEXT, ["2.7.2", "2.4.468", "0.6.1"]),
            _f("status", ENUM, ["green", "red", "rolled_back"]),
            _f("duration", INT, lo=10, hi=1800, label="deploy duration"),
            _f("author", NAME, AGENTS),
        ], "deploy", "deploys", ["hotfix", "rollback", "canary"],
           relations=["deployed_by", "triggered_by"]),
    ], ["triggered_by", "deployed_by"]),
]


def pick_domain(rng: random.Random) -> Domain:
    return rng.choice(DOMAINS)
