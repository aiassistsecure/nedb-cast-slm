/**
 * @interchained/cast — cast short prompts into NEDB query plans.
 *
 * The heavy lifting is a zero-dependency Rust core (`nedb-cast-core`) held to
 * numerical parity with the PyTorch reference in CI: max |delta logit| 7.629e-6
 * and 20/20 decoded strings byte-identical. Node, Python, and Rust all load the
 * same `model.cast` bytes, so they agree by construction rather than by luck.
 */

export declare class Cast {
  /** Load a `model.cast` container from disk. */
  static load(path: string): Cast;

  /** Load from a buffer, for callers that fetch the weights themselves. */
  static fromBuffer(buf: Buffer): Cast;

  /** Cast a short prompt into NQL text. Greedy — one right answer. */
  cast(prompt: string): string;

  castWithLimit(prompt: string, maxNewTokens: number): string;

  /** Raw logits at the final position. */
  logits(prompt: string): number[];

  readonly vocabSize: number;
  readonly nParams: number;

  /**
   * True when the container carried a checksum and it verified. Check this: a
   * corrupt model emits plausible-but-wrong queries, which is worse than a
   * loud failure.
   */
  readonly checksumVerified: boolean;
}

/**
 * Download `model.cast` from the GitHub release once and cache it, then load.
 * Cache dir: $CAST_HOME, or ~/.cache/nedb-cast-slm/<tag>/.
 * The download is checked against SHA256SUMS.txt published in the same release.
 */
export declare function pretrained(opts?: {
  tag?: string;
  quiet?: boolean;
}): Promise<Cast>;
