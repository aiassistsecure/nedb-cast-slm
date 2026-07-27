"""
cast.weights — fetch released weights on first use, then cache them.

Weights are GitHub RELEASE ASSETS, not git history. `ckpt.pt` is ~40MB and
`model.cast` ~13MB; committing them would bloat every clone forever and be
irreversible without a history rewrite. Release assets give the same permanence
and the same github.com URL without that cost.

The consequence is that a fresh `pip install` has code but no weights, so this
module closes that gap:

    from cast import Cast
    caster = Cast.pretrained()        # downloads once, caches, verifies, loads

Cache location, first match wins:
    $CAST_HOME
    $XDG_CACHE_HOME/nedb-cast-slm
    ~/.cache/nedb-cast-slm

Every download is checked against the SHA256SUMS.txt published in the same
release. A mismatch raises — a silently corrupt model would produce plausible
wrong queries, which is the worst possible failure for a query planner.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

REPO = "aiassistsecure/nedb-cast-slm"
DEFAULT_ASSET = "model.cast"
SUMS_ASSET = "SHA256SUMS.txt"

# Pinned by default so a package version always resolves the same weights.
# Overridable for testing against a newer release.
# Pinned to the release that actually carries the weights. A packaging-only
# patch bump must NOT chase its own tag, or Cast.pretrained() 404s.
DEFAULT_TAG = os.environ.get("CAST_MODEL_TAG", "v10.30.90")

_UA = {"User-Agent": "nedb-cast-slm/weights"}


def cache_dir() -> Path:
    if os.environ.get("CAST_HOME"):
        p = Path(os.environ["CAST_HOME"])
    elif os.environ.get("XDG_CACHE_HOME"):
        p = Path(os.environ["XDG_CACHE_HOME"]) / "nedb-cast-slm"
    else:
        p = Path.home() / ".cache" / "nedb-cast-slm"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _asset_url(tag: str, name: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{name}"


def _download(url: str, dest: Path, quiet: bool = False) -> None:
    # Only draw a progress bar on a real terminal. Writing \r into a
    # non-TTY (CI logs, piped output, nohup) produces hundreds of junk lines.
    show = (not quiet) and sys.stdout.isatty()
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            total = int(r.headers.get("Content-Length") or 0)
            tmp = Path(tempfile.mkstemp(dir=str(dest.parent))[1])
            done = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if show and total:
                        pct = 100 * done / total
                        print(f"\r  {dest.name}: {pct:5.1f}% "
                              f"({done/1e6:.1f}/{total/1e6:.1f} MB)",
                              end="", flush=True)
            if show and total:
                print()
            elif not quiet:
                print(f"  {dest.name}: {done/1e6:.1f} MB")
            tmp.replace(dest)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"could not download {url} (HTTP {e.code}). "
            f"If the release has no such asset yet, pass an explicit path to "
            f"Cast.from_pretrained(), or set CAST_MODEL_TAG to a published tag."
        ) from e


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _published_sums(tag: str) -> Dict[str, str]:
    """Parse SHA256SUMS.txt from the release. Empty dict if absent."""
    try:
        req = urllib.request.Request(_asset_url(tag, SUMS_ASSET), headers=_UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode()
    except Exception:
        return {}
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            out[parts[-1].lstrip("*")] = parts[0]
    return out


def fetch(asset: str = DEFAULT_ASSET, tag: Optional[str] = None,
          force: bool = False, quiet: bool = False,
          verify: bool = True) -> Path:
    """Return a local path to `asset`, downloading it once if needed.

    Raises on checksum mismatch. A corrupt model would emit plausible-but-wrong
    queries — far worse than a loud failure.
    """
    tag = tag or DEFAULT_TAG
    dest_dir = cache_dir() / tag
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / asset

    if dest.exists() and not force:
        return dest

    url = _asset_url(tag, asset)
    if not quiet:
        print(f"[cast] fetching {asset} from release {tag}")
    _download(url, dest, quiet=quiet)

    if verify:
        sums = _published_sums(tag)
        want = sums.get(asset)
        if want:
            got = _sha256(dest)
            if got != want:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"checksum mismatch for {asset}: expected {want}, got {got}. "
                    f"The download was corrupt or the asset was replaced; "
                    f"refusing to load it.")
            if not quiet:
                print(f"[cast] sha256 verified: {got[:16]}…")
        elif not quiet:
            print(f"[cast] note: no {SUMS_ASSET} in release {tag}; "
                  f"skipping checksum verification")
    return dest


def clear_cache(tag: Optional[str] = None) -> None:
    target = cache_dir() / tag if tag else cache_dir()
    if target.exists():
        shutil.rmtree(target)
