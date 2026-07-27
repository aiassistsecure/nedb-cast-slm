'use strict';
/**
 * @interchained/cast
 *
 * Thin JS surface over the napi binding. The Rust core does the work; this file
 * only adds the convenience `pretrained()` helper that fetches and caches the
 * released weights, mirroring `Cast.pretrained()` in the Python package.
 */
const { existsSync, readdirSync } = require('fs');
const { join } = require('path');
const { fetchWeights, DEFAULT_TAG } = require('./weights');

// napi build --platform emits nedb-cast.<platform>.node next to this file.
function loadNative() {
  const local = readdirSync(__dirname).filter((f) => f.endsWith('.node'));
  if (local.length) return require(join(__dirname, local[0]));
  // fall back to the per-platform packages napi publishes alongside
  const { platform, arch } = process;
  const candidates = [
    `nedb-cast-slm-${platform}-${arch}`,
    `nedb-cast-slm-${platform}-${arch}-gnu`,
    `nedb-cast-slm-${platform}-${arch}-msvc`,
  ];
  for (const c of candidates) {
    try { return require(c); } catch (_) { /* try next */ }
  }
  throw new Error(
    `no native binding found for ${platform}-${arch}.\n` +
    `Prebuilt binaries ship for linux-x64-gnu, darwin-x64, darwin-arm64, ` +
    `win32-x64-msvc. On other targets, build from source:\n` +
    `  git clone https://github.com/aiassistsecure/nedb-cast-slm\n` +
    `  cd nedb-cast-slm/npm/nedb-cast && npm i && npm run build`);
}

const native = loadNative();

/**
 * Download the released model.cast once, cache it, verify its sha256, and load.
 */
async function pretrained(opts = {}) {
  const path = await fetchWeights({ tag: opts.tag || DEFAULT_TAG, quiet: opts.quiet });
  return native.Cast.load(path);
}

module.exports = { Cast: native.Cast, pretrained, DEFAULT_TAG };
