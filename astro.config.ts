import fs from "node:fs";
// Rehype plugins
import { rehypeHeadingIds } from "@astrojs/markdown-remark";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwind from "@tailwindcss/vite";
import { defineConfig, envField } from "astro/config";
import expressiveCode from "astro-expressive-code";
import icon from "astro-icon";
import robotsTxt from "astro-robots-txt";
import webmanifest from "astro-webmanifest";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeExternalLinks from "rehype-external-links";
import rehypeKatex from "rehype-katex"; /* render math nodes via KaTeX */
import rehypeUnwrapImages from "rehype-unwrap-images";
import { rehypeImageFigure } from "./src/plugins/rehype-image-figure"; /* <img> -> <figure> + <figcaption> */
// Remark plugins
import remarkBreaks from "remark-breaks"; /* single newline -> <br>, matching Obsidian */
import remarkDirective from "remark-directive"; /* Handle ::: directives as nodes */
import remarkMath from "remark-math"; /* parse $…$ / $$…$$ into math nodes */
import { remarkAdmonitions } from "./src/plugins/remark-admonitions"; /* Add admonitions */
import { remarkGithubCard } from "./src/plugins/remark-github-card";
import { remarkObsidianCallouts } from "./src/plugins/remark-obsidian-callouts"; /* > [!warning] -> admonition */
import { remarkObsidianImages } from "./src/plugins/remark-obsidian-images";
import { remarkReadingTime } from "./src/plugins/remark-reading-time";
import { expressiveCodeOptions, siteConfig } from "./src/site.config";

// https://astro.build/config
export default defineConfig({
	site: siteConfig.url,
	image: {
		domains: ["webmention.io"],
		// Responsive images: generate a srcset (incl. 2x) so Markdown images stay
		// sharp on HiDPI / mobile screens instead of upscaling a single width.
		layout: "constrained",
	},
	integrations: [
		expressiveCode(expressiveCodeOptions),
		icon(),
		sitemap(),
		mdx(),
		robotsTxt(),
		webmanifest({
			// See: https://github.com/alextim/astro-lib/blob/main/packages/astro-webmanifest/README.md
			name: siteConfig.title,
			short_name: "Xavier Forge", // optional
			description: siteConfig.description,
			lang: siteConfig.lang,
			icon: "public/logo.png", // the source for generating favicon & icons
			icons: [
				{
					src: "icons/apple-touch-icon.png", // used in src/components/BaseHead.astro L:26
					sizes: "180x180",
					type: "image/png",
				},
				{
					src: "icons/icon-192.png",
					sizes: "192x192",
					type: "image/png",
				},
				{
					src: "icons/icon-512.png",
					sizes: "512x512",
					type: "image/png",
				},
			],
			start_url: "/",
			background_color: "#1d1f21",
			theme_color: "#1c6fc9",
			display: "standalone",
			config: {
				insertFaviconLinks: false,
				insertThemeColorMeta: false,
				insertManifestLink: false,
			},
		}),
	],
	markdown: {
		rehypePlugins: [
			rehypeHeadingIds,
			[rehypeAutolinkHeadings, { behavior: "wrap", properties: { className: ["not-prose"] } }],
			[
				rehypeExternalLinks,
				{
					rel: ["noreferrer", "noopener"],
					target: "_blank",
				},
			],
			rehypeUnwrapImages,
			rehypeImageFigure,
			rehypeKatex,
		],
		remarkPlugins: [
			remarkReadingTime,
			remarkMath,
			remarkObsidianImages,
			remarkObsidianCallouts, // > [!type] callouts -> Cactus admonitions (before remarkBreaks)
			remarkDirective,
			remarkGithubCard,
			remarkAdmonitions,
			remarkBreaks, // last: structural plugins parse first, then soft breaks -> <br>
		],
		remarkRehype: {
			footnoteLabelProperties: {
				className: [""],
			},
		},
	},
	vite: {
		plugins: [tailwind(), rawFonts([".ttf", ".woff"])],
	},
	env: {
		schema: {
			WEBMENTION_API_KEY: envField.string({ context: "server", access: "secret", optional: true }),
			WEBMENTION_URL: envField.string({ context: "client", access: "public", optional: true }),
			WEBMENTION_PINGBACK: envField.string({ context: "client", access: "public", optional: true }),
		},
	},
});

function rawFonts(ext: string[]) {
	return {
		name: "vite-plugin-raw-fonts",
		// @ts-expect-error:next-line
		transform(_, id) {
			if (ext.some((e) => id.endsWith(e))) {
				const buffer = fs.readFileSync(id);
				return {
					code: `export default ${JSON.stringify(buffer)}`,
					map: null,
				};
			}
		},
	};
}
