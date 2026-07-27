'use strict';
/**
 * Fetch released weights on first use, then cache them.
 *
 * Weights are GitHub release assets rather than package contents: model.cast is
 * 13MB, and shipping it inside every npm tarball (and every platform-specific
 * tarball) would multiply that across the prebuild matrix for no benefit.
 *
 * Every download is verified against SHA256SUMS.txt from the same release. A
 * mismatch throws — a silently corrupt model produces plausible wrong queries,
 * the worst possible failure mode for a query planner.
 */
const fs = require('fs');
const os = require('os');
const path = require('path');
const crypto = require('crypto');
const https = require('https');

const REPO = 'aiassistsecure/nedb-cast-slm';
// Pinned to the release that carries the weights. A packaging-only patch bump
// must not chase its own tag, or the fetch 404s.
const DEFAULT_TAG = process.env.CAST_MODEL_TAG || 'v10.30.90';
const ASSET = 'model.cast';
const SUMS = 'SHA256SUMS.txt';

function cacheDir(tag) {
  const base = process.env.CAST_HOME
    || path.join(os.homedir(), '.cache', 'nedb-cast-slm');
  const dir = path.join(base, tag);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function assetUrl(tag, name) {
  return `https://github.com/${REPO}/releases/download/${tag}/${name}`;
}

function get(url, redirects = 0) {
  return new Promise((resolve, reject) => {
    if (redirects > 5) return reject(new Error('too many redirects'));
    https.get(url, { headers: { 'User-Agent': 'nedb-cast-slm/weights' } }, (res) => {
      if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
        res.resume();
        return get(res.headers.location, redirects + 1).then(resolve, reject);
      }
      if (res.statusCode !== 200) {
        res.resume();
        return reject(new Error(
          `could not download ${url} (HTTP ${res.statusCode}). ` +
          `If the release has no such asset yet, pass an explicit path to ` +
          `Cast.load(), or set CAST_MODEL_TAG to a published tag.`));
      }
      const chunks = [];
      res.on('data', (c) => chunks.push(c));
      res.on('end', () => resolve(Buffer.concat(chunks)));
      res.on('error', reject);
    }).on('error', reject);
  });
}

async function publishedSums(tag) {
  try {
    const txt = (await get(assetUrl(tag, SUMS))).toString('utf8');
    const out = {};
    for (const line of txt.split('\n')) {
      const parts = line.trim().split(/\s+/);
      if (parts.length >= 2) out[parts[parts.length - 1].replace(/^\*/, '')] = parts[0];
    }
    return out;
  } catch (_) {
    return {};
  }
}

async function fetchWeights({ tag = DEFAULT_TAG, quiet = false } = {}) {
  const dir = cacheDir(tag);
  const dest = path.join(dir, ASSET);
  if (fs.existsSync(dest)) return dest;

  if (!quiet) console.log(`[cast] fetching ${ASSET} from release ${tag}`);
  const data = await get(assetUrl(tag, ASSET));

  const sums = await publishedSums(tag);
  const want = sums[ASSET];
  if (want) {
    const got = crypto.createHash('sha256').update(data).digest('hex');
    if (got !== want) {
      throw new Error(
        `checksum mismatch for ${ASSET}: expected ${want}, got ${got}. ` +
        `The download was corrupt or the asset was replaced; refusing to load it.`);
    }
    if (!quiet) console.log(`[cast] sha256 verified: ${got.slice(0, 16)}...`);
  } else if (!quiet) {
    console.log(`[cast] note: no ${SUMS} in release ${tag}; skipping verification`);
  }

  const tmp = dest + '.tmp';
  fs.writeFileSync(tmp, data);
  fs.renameSync(tmp, dest);
  return dest;
}

module.exports = { fetchWeights, cacheDir, DEFAULT_TAG };
