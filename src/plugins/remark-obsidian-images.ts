import { visit } from "unist-util-visit";

/**
 * Bridges the user's Obsidian image conventions to Astro:
 *
 *   1. Bare relative paths (e.g. `assets/Hello/foo.png`) are prefixed with
 *      `./` so Astro's image pipeline treats them as collocated assets and
 *      optimises them via `astro:assets`.
 *   2. Numeric alt text (e.g. `![400](...)`) is interpreted as a pixel
 *      width and lifted onto the rendered `<img>` as a `width` attribute.
 *      The alt is cleared so screen readers don't announce "400".
 *
 * URL-like references (http(s)://, protocol-relative, absolute paths
 * starting with `/`) are left untouched.
 */
export function remarkObsidianImages() {
	return (tree: unknown) => {
		visit(tree as never, "image", (node: ImageNode) => {
			if (node.url && !isExternal(node.url) && !node.url.startsWith("./")) {
				node.url = `./${node.url.replace(/^\/+/, "")}`;
			}

			if (node.alt && /^\d+$/.test(node.alt.trim())) {
				const width = Number.parseInt(node.alt.trim(), 10);
				node.data ??= {};
				node.data.hProperties ??= {};
				node.data.hProperties.width = width;
				node.alt = "";
			}
		});
	};
}

function isExternal(url: string): boolean {
	return /^([a-z]+:)?\/\//i.test(url);
}

interface ImageNode {
	type: "image";
	url: string;
	alt?: string | null;
	data?: {
		hProperties?: Record<string, unknown>;
	};
}
