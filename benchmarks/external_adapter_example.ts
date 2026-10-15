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
 * `createStrategy(name).compress(messages, query, { maxTokens })`. Point
 * HARNESS_DIR at that project; nothing in it is modified or copied.
 *
 *   cd /path/to/harness && HARNESS_DIR=$PWD HARNESS_STRATEGY=simple \
 *     npx tsx /path/to/this/adapter.ts
 *
 * Run it from inside the harness directory: that is where `npx tsx` and its
 * node_modules live. The adapter itself imports nothing from the harness at
 * build time — only at run time, from the path you give it — so the code being
 * scored is always the code that is actually deployed, not a copy that drifts.
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
    "src/lib/harness/compression/registry.ts"
  );
  const { createStrategy } = await import(pathToFileURL(modulePath).href);

  const strategy = createStrategy(process.env.HARNESS_STRATEGY ?? "simple");
  const result = await strategy.compress(input.messages, input.query ?? "", {
    maxTokens: input.maxTokens ?? 4000,
    useLlm: false,
  });

  process.stdout.write(
    JSON.stringify({
      text: result.compressedText,
      ratio: result.compressionRatio,
    })
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
