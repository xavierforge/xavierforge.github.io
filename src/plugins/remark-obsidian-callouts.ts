import type { Blockquote, Root } from "mdast";
import type { Plugin } from "unified";
import { SKIP, visit } from "unist-util-visit";
import type { AdmonitionType } from "@/types";
import { h } from "../utils/remark";

/**
 * Maps Obsidian callout types (`> [!warning]`) onto the five admonition types
 * Cactus styles (`src/plugins/remark-admonitions.ts`). Unknown types fall back
 * to "note" so a callout never renders as a bare blockquote.
 */
const CALLOUT_TO_ADMONITION: Record<string, AdmonitionType> = {
	tip: "tip",
	hint: "tip",
	success: "tip",
	check: "tip",
	done: "tip",
	note: "note",
	info: "note",
	abstract: "note",
	summary: "note",
	tldr: "note",
	example: "note",
	quote: "note",
	cite: "note",
	important: "important",
	todo: "important",
	warning: "warning",
	attention: "warning",
	caution: "caution",
	danger: "caution",
	error: "caution",
	failure: "caution",
	fail: "caution",
	missing: "caution",
	bug: "caution",
};

// Obsidian callout marker on the first line of a blockquote:
//   [!WARNING] optional title  /  [!note]-  /  [!tip]+ Title
const CALLOUT_RE = /^\[!([^\]]+)\][+-]?[ \t]*(.*)/;

/**
 * Rewrites Obsidian callouts into the same `<aside class="admonition">` markup
 * that Cactus's `:::note` directive admonitions produce, so both share one set
 * of styles. Must run before `remark-breaks`, which would otherwise split the
 * marker line off the first text node. Plain blockquotes pass through untouched.
 */
export const remarkObsidianCallouts: Plugin<[], Root> = () => (tree) => {
	visit(tree, "blockquote", (node: Blockquote, index, parent) => {
		if (!parent || index === undefined) return;

		const firstChild = node.children[0];
		if (firstChild?.type !== "paragraph") return;
		const firstText = firstChild.children[0];
		if (firstText?.type !== "text") return;

		const newlineIdx = firstText.value.indexOf("\n");
		const firstLine = newlineIdx === -1 ? firstText.value : firstText.value.slice(0, newlineIdx);
		const match = firstLine.match(CALLOUT_RE);
		if (!match) return;

		const admonitionType = CALLOUT_TO_ADMONITION[(match[1] ?? "").trim().toLowerCase()] ?? "note";
		const title = (match[2] ?? "").trim() || admonitionType;

		// Drop the marker line; keep everything after it as the callout body.
		firstText.value = newlineIdx === -1 ? "" : firstText.value.slice(newlineIdx + 1);
		if (firstText.value === "" && firstChild.children.length === 1) {
			node.children.shift();
		}

		const admonition = h(
			"aside",
			{ "aria-label": title, class: "admonition", "data-admonition-type": admonitionType },
			[
				h("p", { "aria-hidden": "true", class: "admonition-title" }, [
					{ type: "text", value: title },
				]),
				h("div", { class: "admonition-content" }, node.children),
			],
		);

		parent.children[index] = admonition;
		return [SKIP, index];
	});
};
