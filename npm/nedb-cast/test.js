'use strict';
/**
 * Smoke test for the npm package. Mirrors what `cast smoke` does in Python:
 * load the released weights, cast real prompts, and check the output shape.
 *
 * Run: node test.js
 */
const assert = require('assert');
const { pretrained } = require('./index');

const GOLDEN = [
  ['top 5 stylists in winter park', /^FROM stylists\b/],
  ['invoices that are overdue limit 10', /^FROM invoices\b/],
  ['what caused these checkpoints', /TRACE caused_by/],
  ['show me all services', /^FROM services/],
];

(async () => {
  const cast = await pretrained();
  console.log(`loaded: ${cast.nParams} params, vocab ${cast.vocabSize}, ` +
              `checksum ${cast.checksumVerified ? 'verified' : 'UNVERIFIED'}`);
  assert.ok(cast.checksumVerified, 'container checksum did not verify');

  let pass = 0;
  for (const [prompt, want] of GOLDEN) {
    const t0 = Date.now();
    const nql = cast.cast(prompt);
    const ms = Date.now() - t0;
    const ok = want.test(nql);
    console.log(`  [${ok ? 'OK  ' : 'FAIL'}] "${prompt}"`);
    console.log(`         ${nql}  (${ms} ms)`);
    if (ok) pass++;
  }

  // determinism: greedy decoding must be reproducible
  const a = cast.cast(GOLDEN[0][0]);
  const b = cast.cast(GOLDEN[0][0]);
  assert.strictEqual(a, b, 'greedy decode is not deterministic');
  console.log('  [OK  ] greedy decode is deterministic');

  console.log(`\n${pass}/${GOLDEN.length} prompts matched`);
  process.exit(pass === GOLDEN.length ? 0 : 1);
})().catch((e) => { console.error('FAILED:', e.message); process.exit(1); });
