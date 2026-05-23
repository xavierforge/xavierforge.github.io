import { visit } from "unist-util-visit";

/**
 * Bridges the user's Obsidian image conventions to Astro:
 *
 *   1. URL-encoded characters in relative paths (e.g. `assets/OpenClaw%20on%20Jetson/foo.gif`)
 *      are decoded so the literal filename on disk is resolved.
 *   2. Bare relative paths (e.g. `assets/Hello/foo.png`) are prefixed with
 *      `./` so Astro's image pipeline treats them as collocated assets and
 *      optimises them via `astro:assets`.
 *   3. Obsidian-style alt + width syntax is parsed:
 *        - `![alt|400](...)`    → alt = "alt", width = 400
 *        - `![400](...)`        → alt = "",    width = 400  (bare numeric alt)
 *        - `![alt](...)`        → alt = "alt", width = unset
 *      The width is lifted onto the rendered `<img>` as a `width` attribute.
 *
 * URL-like references (http(s)://, protocol-relative, absolute paths
 * starting with `/`) are left untouched.
 */
export function remarkObsidianImages() {
	return (tree: unknown) => {
		visit(tree as never, "image", (node: ImageNode) => {
			if (node.url && !isExternal(node.url)) {
				let url = decodeUrlSafe(node.url);
				if (!url.startsWith("./") && !url.startsWith("/")) {
					url = `./${url}`;
				}
				node.url = url;
			}

			if (node.alt) {
				const parsed = parseAlt(node.alt);
				node.alt = parsed.alt;
				if (parsed.width !== undefined) {
					node.data ??= {};
					node.data.hProperties ??= {};
					node.data.hProperties.width = parsed.width;
				}
			}
		});
	};
}

function isExternal(url: string): boolean {
	return /^([a-z]+:)?\/\//i.test(url);
}

function decodeUrlSafe(url: string): string {
	try {
		return decodeURIComponent(url);
	} catch {
		return url;
	}
}

function parseAlt(raw: string): { alt: string; width?: number } {
	const trimmed = raw.trim();

	// `![alt|400](...)` — split on last `|`
	const pipeIndex = trimmed.lastIndexOf("|");
	if (pipeIndex !== -1) {
		const left = trimmed.slice(0, pipeIndex).trim();
		const right = trimmed.slice(pipeIndex + 1).trim();
		if (/^\d+$/.test(right)) {
			return { alt: left, width: Number.parseInt(right, 10) };
		}
	}

	// `![400](...)` — bare numeric alt, treat as width
	if (/^\d+$/.test(trimmed)) {
		return { alt: "", width: Number.parseInt(trimmed, 10) };
	}

	return { alt: trimmed };
}
