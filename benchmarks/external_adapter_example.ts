/**
 * Template for `qa_eval.py --external NAME=COMMAND`.
 *
 * Tests someone else's compressor on the same points as this library. The
 * contract is deliberately tiny so nothing about the implementation has to
 * change to be measured:
 *
 *   stdin:  {"messages": [{"role": ..., "content": ...}], "query": ..., "maxTokens": N}
 *   stdout: {"text": "<compressed context>", "ratio": 0.31}
 *
 * This copy is written against a TypeScript harness that exposes
 * `compressContext(messages, query, options)` — point HARNESS_DIR at that
 * project and nothing in it is modified or copied. It calls the same entry
 * point the running service calls, not the strategy underneath it, so the
 * recent-message window, the strategy routing and the fallback path are all
 * part of what gets scored.
 *
 *   cd /path/to/harness && HARNESS_DIR=$PWD HARNESS_STRATEGY=auto \
 *     npx tsx /path/to/this/adapter.ts
 *
 * Run it from inside the harness directory: that is where `npx tsx` and its
 * node_modules live. The adapter itself imports nothing from the harness at
 * build time — only at run time, from the path you give it — so the code being
 * scored is always the code that is actually deployed, not a copy that drifts.
 *
 * Env:
 *   HARNESS_DIR       project root (required)
 *   HARNESS_STRATEGY  "auto" (default, matches production) | simple | acs | hcs
 *   HARNESS_USE_LLM   "1" to let the strategy call a model (default off, which
 *                     is also what production does — nothing passes a gateway)
 */

import path from "node:path";
import { pathToFileURL } from "node:url";

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks).toString("utf8");
}

async function main(): Promise<void> {
  const harnessDir = process.env.HARNESS_DIR;
  if (!harnessDir) {
    console.error("HARNESS_DIR is not set (point it at the project root)");
    process.exit(2);
  }

  const input = JSON.parse(await readStdin());
  const modulePath = path.join(
    harnessDir,
    "src/lib/harness/context-compressor.ts"
  );
  const { compressContext } = await import(pathToFileURL(modulePath).href);

  const compressed = await compressContext(input.messages, input.query ?? "", {
    maxTokens: input.maxTokens ?? 4000,
    strategy: process.env.HARNESS_STRATEGY ?? "auto",
    useLlm: process.env.HARNESS_USE_LLM === "1",
  });

  process.stdout.write(
    JSON.stringify({
      text: compressed
        .map((message: { role: string; content: string }) =>
          `[${message.role}] ${message.content}`
        )
        .join("\n"),
      // Reported for eyeballing only; qa_eval.py re-measures the ratio with its
      // own tokenizer so that every strategy is compared in the same units.
      ratio: 1,
    })
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
