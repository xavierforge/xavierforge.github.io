#!/usr/bin/env node
//
// Translation drift checker for the blog's in-page bilingual posts.
//
// English translations live beside their Chinese originals as `<slug>.en.md`
// (repo-only; excluded from the Obsidian sync). Each `.en.md` records a
// `sourceHash` of the Chinese body it was translated from. This script
// recomputes that hash from the current `<slug>.md` and reports which
// translations have drifted out of date — run it before pushing so you know
// which originals changed and need a re-translation.
//
// Usage:
//   node scripts/check-translations.mjs [dir]            # report (warn-only)
//   node scripts/check-translations.mjs [dir] --strict   # exit 1 if any issue
//   node scripts/check-translations.mjs [dir] --update    # stamp current hashes
//
// `--update` rewrites every `.en.md`'s sourceHash to match its current source —
// use it right after translating (or after deliberately re-syncing a change).

import { createHash } from "node:crypto";
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join } from "node:path";

const args = process.argv.slice(2);
const strict = args.includes("--strict");
const update = args.includes("--update");
const dir = args.find((a) => !a.startsWith("--")) ?? "src/content/post";

/** Strip a leading `--- … ---` frontmatter block and normalize the body. */
function bodyOf(raw) {
	const m = raw.match(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/);
	const body = m ? raw.slice(m[0].length) : raw;
	return body.replace(/\r\n/g, "\n").trim();
}

/** Short, stable content hash of a markdown file's body. */
function hashBody(raw) {
	return createHash("sha256").update(bodyOf(raw)).digest("hex").slice(0, 16);
}

/** Read the `sourceHash:` value from a frontmatter block, if present. */
function readSourceHash(raw) {
	const fm = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
	if (!fm) return null;
	const line = fm[1].match(/^sourceHash:\s*["']?([0-9a-f]+)["']?\s*$/m);
	return line ? line[1] : null;
}

/** Return `raw` with its frontmatter `sourceHash:` set to `hash` (insert if absent). */
function setSourceHash(raw, hash) {
	const fm = raw.match(/^(---\r?\n)([\s\S]*?)(\r?\n---)/);
	if (!fm) return raw;
	let body = fm[2];
	if (/^sourceHash:.*$/m.test(body)) {
		body = body.replace(/^sourceHash:.*$/m, `sourceHash: ${hash}`);
	} else {
		body = `${body}\nsourceHash: ${hash}`;
	}
	return raw.replace(fm[0], `${fm[1]}${body}${fm[3]}`);
}

let entries;
try {
	entries = readdirSync(dir).filter((f) => f.endsWith(".en.md"));
} catch (err) {
	console.error(`check-translations: cannot read ${dir}: ${err.message}`);
	process.exit(1);
}

if (entries.length === 0) {
	console.log("check-translations: no *.en.md translations found.");
	process.exit(0);
}

let problems = 0;
const lines = [];

for (const file of entries.sort()) {
	const slug = basename(file, ".en.md");
	const enPath = join(dir, file);
	const srcPath = join(dir, `${slug}.md`);

	let srcRaw;
	try {
		srcRaw = readFileSync(srcPath, "utf8");
	} catch {
		problems++;
		lines.push(`  ✗  ${slug}  — no source ${slug}.md (orphaned translation)`);
		continue;
	}

	const srcHash = hashBody(srcRaw);
	const enRaw = readFileSync(enPath, "utf8");

	if (update) {
		writeFileSync(enPath, setSourceHash(enRaw, srcHash));
		lines.push(`  ⟳  ${slug}  — stamped ${srcHash}`);
		continue;
	}

	const stored = readSourceHash(enRaw);
	if (!stored) {
		problems++;
		lines.push(`  ⚠  ${slug}  — no sourceHash recorded (run --update after translating)`);
	} else if (stored !== srcHash) {
		problems++;
		lines.push(`  ⚠  ${slug}  — STALE: source changed (${stored} → ${srcHash})`);
	} else {
		lines.push(`  ✓  ${slug}  — up to date`);
	}
}

console.log(`check-translations: ${entries.length} translation(s) in ${dir}`);
console.log(lines.join("\n"));

if (update) process.exit(0);

if (problems > 0) {
	console.log(`\n${problems} translation(s) need attention.`);
	process.exit(strict ? 1 : 0);
}
console.log("\nAll translations up to date.");
