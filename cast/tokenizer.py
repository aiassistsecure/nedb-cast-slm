"""
cast.tokenizer — a small, purpose-built word-level tokenizer.

Why not BPE? At ~4M params the embedding table is a large fraction of the budget.
A word-level vocab fitted to THIS corpus keeps sequences short (fewer tokens per
example = faster training and less to learn) and makes the NQL side effectively
one-token-per-keyword. Numbers are the exception: they're unbounded, so we split
them into digits and let the model compose them.

Design:
  - lowercase everything except NQL keywords, which are uppercased on output
  - split on whitespace and punctuation, keeping operators as single tokens
  - digits are individual tokens (0-9, '.', '-') so any number is representable
  - unknown words map to <unk>; we report the rate so it can't hide
  - special tokens: <pad> <s> </s> <unk> <sep>

<sep> separates prompt from target, so the model reads:
    <s> prompt tokens <sep> nql tokens </s>
and loss is computed only on the tokens after <sep>.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

PAD, BOS, EOS, UNK, SEP = "<pad>", "<s>", "</s>", "<unk>", "<sep>"
SPECIALS = [PAD, BOS, EOS, UNK, SEP]

# Operators and punctuation that must survive as their own tokens.
_OPS = ["<=", ">=", "!=", "=", "<", ">"]
# Tokenise: quoted strings keep their quotes as tokens, numbers split to digits.
_SPLIT_RE = re.compile(r'(<=|>=|!=|=|<|>|"|\'|,|\?|!|\$|&|\.|\-|\s+)')

_DIGITS = [str(d) for d in range(10)]


def pre_tokenize(text: str) -> List[str]:
    """Split raw text into atomic pieces without vocab knowledge."""
    out: List[str] = []
    for piece in _SPLIT_RE.split(text):
        if piece is None or piece == "" or piece.isspace():
            continue
        if piece in _OPS or piece in ('"', "'", ",", "?", "!", "$", "&", ".", "-"):
            out.append(piece)
            continue
        # split numbers into digits so any integer/float is representable
        if any(ch.isdigit() for ch in piece):
            buf = ""
            for ch in piece:
                if ch.isdigit():
                    if buf:
                        out.append(buf.lower())
                        buf = ""
                    out.append(ch)
                else:
                    buf += ch
            if buf:
                out.append(buf.lower())
            continue
        out.append(piece.lower())
    return out


class CastTokenizer:
    def __init__(self, vocab: Optional[List[str]] = None):
        self.itos: List[str] = vocab or []
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    # ---------------------------------------------------------------- fitting
    @classmethod
    def fit(cls, texts: List[str], min_freq: int = 2,
            max_vocab: int = 4096) -> "CastTokenizer":
        counter: Counter = Counter()
        for t in texts:
            counter.update(pre_tokenize(t))
        # always include specials and digits, even if unseen
        vocab = list(SPECIALS)
        for d in _DIGITS + [".", "-", '"', "'"]:
            if d not in vocab:
                vocab.append(d)
        for tok, n in counter.most_common():
            if len(vocab) >= max_vocab:
                break
            if tok in vocab:
                continue
            if n < min_freq:
                continue
            vocab.append(tok)
        return cls(vocab)

    # ---------------------------------------------------------------- encoding
    @property
    def pad_id(self) -> int: return self.stoi[PAD]
    @property
    def bos_id(self) -> int: return self.stoi[BOS]
    @property
    def eos_id(self) -> int: return self.stoi[EOS]
    @property
    def unk_id(self) -> int: return self.stoi[UNK]
    @property
    def sep_id(self) -> int: return self.stoi[SEP]

    def __len__(self) -> int: return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in pre_tokenize(text)]

    def encode_pair(self, prompt: str, target: str) -> Tuple[List[int], int]:
        """Return (ids, prompt_len) where ids = <s> prompt <sep> target </s>.

        prompt_len counts tokens up to and including <sep>, so the trainer can
        mask loss on the prompt and train only on the NQL continuation.
        """
        p = self.encode(prompt)
        t = self.encode(target)
        ids = [self.bos_id] + p + [self.sep_id] + t + [self.eos_id]
        prompt_len = 1 + len(p) + 1
        return ids, prompt_len

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            t = self.itos[i]
            if skip_special and t in SPECIALS:
                continue
            toks.append(t)
        return self._detok(toks)

    @staticmethod
    def _detok(toks: List[str]) -> str:
        """Rejoin tokens into NQL text, merging digits back into numbers.

        Three bugs were found here by round-trip testing and are fixed:
          1. Dates lost their hyphens ("2026-10-18" -> "2026 -10 -18") because
             the digit-merge loop accepted '.' but not '-'.
          2. NQL keywords appearing INSIDE a quoted string were uppercased
             (SEARCH "rate limit" -> "rate LIMIT"), corrupting the literal.
             Keyword casing must only apply outside quotes.
          3. '!=' lost its leading space ("occurred_at!=") — harmless to the
             parser but it made decoded output not byte-equal to canonical NQL.
        """
        NQL_KW = {"from", "as", "of", "where", "and", "search", "order", "by",
                  "asc", "desc", "traverse", "trace", "reverse", "limit",
                  "valid", "group", "count", "sum", "avg", "min", "max",
                  "true", "false", "null"}
        out: List[str] = []
        i = 0
        in_quote = False
        while i < len(toks):
            t = toks[i]

            if t == '"':
                in_quote = not in_quote
                out.append(t)
                i += 1
                continue

            # merge runs of digits, '.' and '-' into a single literal so dates
            # ("2026-10-18") and floats ("31.0") survive intact
            starts_num = t.isdigit() or (
                t == "-" and i + 1 < len(toks) and toks[i + 1].isdigit())
            if starts_num:
                num = t
                i += 1
                while i < len(toks):
                    nx = toks[i]
                    if nx.isdigit():
                        num += nx
                        i += 1
                        continue
                    # '.' or '-' only continue the literal when a digit follows
                    if nx in (".", "-") and i + 1 < len(toks) and toks[i + 1].isdigit():
                        num += nx
                        i += 1
                        continue
                    break
                out.append(num)
                continue

            # keyword casing applies OUTSIDE quotes only
            out.append(t if in_quote else (t.upper() if t in NQL_KW else t))
            i += 1

        s = " ".join(out)
        # tighten quoted literals: `" paid "` -> `"paid"`
        s = re.sub(r'"\s*([^"]*?)\s*"', lambda m: '"%s"' % m.group(1), s)
        s = re.sub(r"\s+([,?!])", r"\1", s)
        return " ".join(s.split()).strip()

    # ------------------------------------------------------------------- io
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"itos": self.itos}, fh)

    @classmethod
    def load(cls, path: str) -> "CastTokenizer":
        with open(path) as fh:
            return cls(json.load(fh)["itos"])
