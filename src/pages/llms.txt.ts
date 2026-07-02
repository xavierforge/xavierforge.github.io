import { getAllPosts } from "@/data/post";
import { canonicalSlug, postPath } from "@/i18n";
import { siteConfig } from "@/site.config";

/** `/llms.txt` — an LLM-friendly Markdown index of the site, per https://llmstxt.org/.
 *  Dynamically generated from the (default-locale, Chinese) posts so it stays in
 *  sync with content on every build. English mirrors live under /en/; each post's
 *  page carries its own hreflang, so we list the canonical zh URLs here. */
export const GET = async () => {
	const posts = (await getAllPosts()).sort(
		(a, b) => b.data.publishDate.valueOf() - a.data.publishDate.valueOf(),
	);

	const abs = (path: string) => new URL(path, import.meta.env.SITE).href;

	const lines = [
		`# ${siteConfig.title}`,
		"",
		`> ${siteConfig.description}`,
		"",
		`Author: ${siteConfig.author}. Posts are in Traditional Chinese by default; English translations, when available, live under \`/en/\`.`,
		"",
		"## Posts",
		"",
		...posts.map((post) => {
			const url = abs(postPath(canonicalSlug(post.id), "zh-Hant"));
			return `- [${post.data.title}](${url}): ${post.data.description}`;
		}),
		"",
		"## Optional",
		"",
		`- [About](${abs("/about/")}): About the author.`,
		`- [RSS](${abs("/rss.xml")}): Full post feed.`,
		"",
	];

	return new Response(lines.join("\n"), {
		headers: { "Content-Type": "text/plain; charset=utf-8" },
	});
};
