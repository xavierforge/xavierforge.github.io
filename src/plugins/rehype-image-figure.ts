import type { Element, ElementContent, Root, RootContent } from "hast";
import { visit } from "unist-util-visit";

/**
 * Wraps Markdown images in `<figure class="post-figure">` and turns their alt
 * text into a `<figcaption>` (empty alt → no caption, for decorative images).
 *
 * Posts here are written without blank lines around images, so an image ends up
 * INSIDE a paragraph, flanked by `<br>`s (from remark-breaks) — never unwrapped
 * by rehypeUnwrapImages. A `<figure>` can't live inside a `<p>`, so for those we
 * split the paragraph: text runs stay as `<p>`, each image becomes a sibling
 * `<figure>`, and the `<br>`s hugging the image are dropped. Already-standalone
 * images (e.g. unwrapped sole-image paragraphs) are wrapped in place.
 */

const isImg = (n: RootContent | ElementContent): n is Element =>
	n.type === "element" && n.tagName === "img";

const isDropAtEdge = (n: RootContent | ElementContent | undefined): boolean =>
	n !== undefined &&
	((n.type === "element" && n.tagName === "br") || (n.type === "text" && n.value.trim() === ""));

function toFigure(img: Element): Element {
	const alt = typeof img.properties?.alt === "string" ? img.properties.alt.trim() : "";
	const children: ElementContent[] = [img];
	if (alt) {
		children.push({
			type: "element",
			tagName: "figcaption",
			properties: {},
			children: [{ type: "text", value: alt }],
		});
	}
	return {
		type: "element",
		tagName: "figure",
		properties: { className: ["post-figure"] },
		children,
	};
}

export function rehypeImageFigure() {
	return (tree: Root) => {
		visit(tree, "element", (node: Element, index, parent) => {
			if (!parent || index === undefined) return;

			// A paragraph that contains image(s): split into text-paragraphs + figures.
			if (node.tagName === "p" && node.children.some(isImg)) {
				const out: RootContent[] = [];
				let run: ElementContent[] = [];
				const flushRun = () => {
					while (run.length && isDropAtEdge(run[0])) run.shift();
					while (run.length && isDropAtEdge(run[run.length - 1])) run.pop();
					if (run.length) {
						out.push({ type: "element", tagName: "p", properties: {}, children: run });
					}
					run = [];
				};
				for (const child of node.children) {
					if (isImg(child)) {
						flushRun();
						out.push(toFigure(child));
					} else {
						run.push(child);
					}
				}
				flushRun();
				parent.children.splice(index, 1, ...out);
				// Continue past the nodes we just inserted (their imgs are already in figures).
				return index + out.length;
			}

			// A standalone <img> not already inside a figure → wrap it.
			if (node.tagName === "img" && !(parent.type === "element" && parent.tagName === "figure")) {
				parent.children[index] = toFigure(node);
			}
			return undefined;
		});
	};
}
