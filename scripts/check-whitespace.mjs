#!/usr/bin/env node
//
// Scan Markdown for invisible / unusual whitespace that tends to sneak in via
// copy-paste (LLM output, web pages, Word) and looks fine until it doesn't —
// e.g. hair spaces (U+200A) around em dashes, no-break spaces, zero-width chars.
//
// It is tuned for this blog's conventions: a *single* thin space hugging a dash
// (`字—字`) is an intentional typographic style and is NOT reported. Only
// anomalies are flagged. Fenced code blocks are skipped (the pasted Python is
// full of trailing spaces that aren't the point).
//
// Usage:
//   node scripts/check-whitespace.mjs [--strict] [--verbose] <file|dir> ...
//
//   --strict    exit 1 if any issue is found (for CI / hard gating)
//   --verbose   also list the low-severity "info" findings (single NBSP, etc.)
//
// Default behaviour is warn-only (exit 0) so it never blocks a sync.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join } from "node:path";

const args = process.argv.slice(2);
const strict = args.includes("--strict");
const verbose = args.includes("--verbose") || args.includes("-v");
const paths = args.filter((a) => !a.startsWith("-"));

if (paths.length === 0) {
	console.error("usage: check-whitespace.mjs [--strict] [--verbose] <file|dir> ...");
	process.exit(2);
}

// --- character classes -------------------------------------------------------

// Always-suspicious: invisible, zero-width, never typed on purpose.
const ZERO_WIDTH = new Set([0x200b, 0x200c, 0x200d, 0x2060, 0xfeff]);

// "Special" spaces — anything that isn't a plain ASCII space or tab.
const SPECIAL = new Set([
	0x00a0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009,
	0x200a, 0x202f, 0x205f, 0x3000,
]);

// Thin/hair spaces that are legitimately used to pad dashes in this blog.
const THIN = new Set([0x2009, 0x200a, 0x202f]);

// Dashes those thin spaces are allowed to hug.
const DASH = new Set([0x2013, 0x2014]);

// Keys are Unicode codepoints in hex to mirror the U+ notation in the comments
// (biome's useSimpleNumberKeys is disabled for this file in biome.json).
const NAMES = {
	0x00a0: "NO-BREAK SPACE",
	0x2002: "EN SPACE",
	0x2003: "EM SPACE",
	0x2009: "THIN SPACE",
	0x200a: "HAIR SPACE",
	0x202f: "NARROW NO-BREAK SPACE",
	0x205f: "MEDIUM MATHEMATICAL SPACE",
	0x3000: "IDEOGRAPHIC SPACE",
	0x200b: "ZERO WIDTH SPACE",
	0x200c: "ZERO WIDTH NON-JOINER",
	0x200d: "ZERO WIDTH JOINER",
	0x2060: "WORD JOINER",
	0xfeff: "ZERO WIDTH NO-BREAK SPACE / BOM",
};

const cp = (c) => `U+${c.toString(16).toUpperCase().padStart(4, "0")}`;
const named = (c) => `${cp(c)} ${NAMES[c] ?? ""}`.trim();

// Render a snippet with every invisible char made visible.
function viz(str) {
	let out = "";
	for (const ch of str) {
		const c = ch.codePointAt(0);
		if (ZERO_WIDTH.has(c) || SPECIAL.has(c)) out += `«${cp(c)}»`;
		else out += ch;
	}
	return out;
}

// --- per-line scan -----------------------------------------------------------

function scanLine(line) {
	const codes = Array.from(line, (ch) => ch.codePointAt(0));
	const flagged = new Array(codes.length).fill(false);
	const findings = [];

	// 1. zero-width chars (anywhere)
	for (let i = 0; i < codes.length; i++) {
		if (ZERO_WIDTH.has(codes[i])) {
			findings.push({ i, len: 1, sev: "high", rule: "zero-width char", codes: [codes[i]] });
			flagged[i] = true;
		}
	}

	// 2. runs of 2+ special spaces (the "doubled hair space" bug)
	for (let i = 0; i < codes.length; ) {
		if (SPECIAL.has(codes[i])) {
			let j = i;
			while (j < codes.length && SPECIAL.has(codes[j])) j++;
			if (j - i >= 2) {
				findings.push({
					i,
					len: j - i,
					sev: "high",
					rule: "consecutive special spaces",
					codes: codes.slice(i, j),
				});
				for (let k = i; k < j; k++) flagged[k] = true;
			}
			i = j;
		} else i++;
	}

	// 3. stray thin space — not part of a run, not hugging a dash
	for (let i = 0; i < codes.length; i++) {
		if (flagged[i] || !THIN.has(codes[i])) continue;
		const prev = i > 0 ? codes[i - 1] : 0;
		const next = i + 1 < codes.length ? codes[i + 1] : 0;
		if (!DASH.has(prev) && !DASH.has(next)) {
			findings.push({
				i,
				len: 1,
				sev: "med",
				rule: "stray thin space (not hugging a dash)",
				codes: [codes[i]],
			});
			flagged[i] = true;
		}
	}

	// 4. single NBSP left over — informational (often intentional keep-together)
	for (let i = 0; i < codes.length; i++) {
		if (flagged[i] || codes[i] !== 0x00a0) continue;
		findings.push({ i, len: 1, sev: "info", rule: "NBSP (no-break space)", codes: [codes[i]] });
	}

	return findings;
}

function scanFile(file) {
	const lines = readFileSync(file, "utf8").split("\n");
	const results = [];
	let inFence = false;
	for (let n = 0; n < lines.length; n++) {
		const line = lines[n];
		if (/^\s*(```|~~~)/.test(line)) {
			inFence = !inFence;
			continue;
		}
		if (inFence) continue;
		for (const f of scanLine(line)) {
			const start = Math.max(0, f.i - 18);
			const arr = Array.from(line);
			const snippet = viz(arr.slice(start, f.i + f.len + 18).join(""));
			results.push({
				file,
				line: n + 1,
				col: f.i + 1,
				sev: f.sev,
				rule: f.rule,
				chars: f.codes.map(named).join(", "),
				snippet,
			});
		}
	}
	return results;
}

// --- collect target files ----------------------------------------------------

function collect(p) {
	const st = statSync(p);
	if (st.isFile()) return extname(p) === ".md" ? [p] : [];
	return readdirSync(p, { withFileTypes: true }).flatMap((d) => {
		if (d.name.startsWith(".")) return [];
		return collect(join(p, d.name));
	});
}

const files = paths.flatMap(collect);
const all = files.flatMap(scanFile);

// --- report ------------------------------------------------------------------

const TAG = { high: "🔴 HIGH", med: "🟡 WARN", info: "·  info" };
const shown = all.filter((r) => r.sev !== "info" || verbose);
const issues = all.filter((r) => r.sev !== "info").length;
const infos = all.length - issues;

let lastFile = "";
for (const r of shown) {
	if (r.file !== lastFile) {
		console.log(`\n${r.file}`);
		lastFile = r.file;
	}
	console.log(`  ${TAG[r.sev]}  ${r.line}:${r.col}  ${r.rule}`);
	console.log(`        ${r.chars}`);
	console.log(`        ${r.snippet}`);
}

console.log();
if (issues === 0) {
	console.log(
		`✓ whitespace check: clean across ${files.length} file(s)` +
			(infos && !verbose ? ` (${infos} info hidden — use --verbose)` : ""),
	);
} else {
	console.log(
		`whitespace check: ${issues} issue(s) across ${files.length} file(s)` +
			(infos && !verbose ? `, ${infos} info hidden (--verbose)` : ""),
	);
}

if (strict && issues > 0) process.exit(1);
