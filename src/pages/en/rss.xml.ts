import rss from "@astrojs/rss";
import { getPostsByLocale } from "@/data/post";
import { canonicalSlug } from "@/i18n";
import { siteConfig } from "@/site.config";

export const GET = async () => {
	const posts = await getPostsByLocale("en");

	return rss({
		title: siteConfig.title,
		description: siteConfig.description,
		site: import.meta.env.SITE,
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.publishDate,
			link: `en/posts/${canonicalSlug(post.id)}/`,
		})),
	});
};
