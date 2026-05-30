import { type CollectionEntry, getCollection } from "astro:content";

/** filter out draft posts based on the environment
 *
 * Also excludes `*.en.md` translations: they are not standalone posts but
 * alternate-language bodies rendered in-page on their original post via a
 * toggle (see getPostTranslation + BlogPost.astro). Excluding them here keeps
 * them out of every listing/route at once — index, /posts, tags, RSS, og-image,
 * and the [...slug] page builder all go through getAllPosts.
 */
export async function getAllPosts(): Promise<CollectionEntry<"post">[]> {
	return await getCollection("post", ({ id, data }) => {
		if (id.endsWith(".en")) return false;
		return import.meta.env.PROD ? !data.draft : true;
	});
}

/** Look up the English translation companion for a post id, if one exists. */
export async function getPostTranslation(id: string): Promise<CollectionEntry<"post"> | undefined> {
	const matches = await getCollection("post", (entry) => entry.id === `${id}.en`);
	return matches[0];
}

/** Get tag metadata by tag name */
export async function getTagMeta(tag: string): Promise<CollectionEntry<"tag"> | undefined> {
	const tagEntries = await getCollection("tag", (entry) => {
		return entry.id === tag;
	});
	return tagEntries[0];
}

/** groups posts by year (based on option siteConfig.sortPostsByUpdatedDate), using the year as the key
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 */
export function groupPostsByYear(posts: CollectionEntry<"post">[]) {
	return Object.groupBy(posts, (post) => post.data.publishDate.getFullYear().toString());
}

/** returns all tags created from posts (inc duplicate tags)
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getAllTags(posts: CollectionEntry<"post">[]) {
	return posts.flatMap((post) => [...post.data.tags]);
}

/** returns all unique tags created from posts
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getUniqueTags(posts: CollectionEntry<"post">[]) {
	return [...new Set(getAllTags(posts))];
}

/** returns a count of each unique tag - [[tagName, count], ...]
 *  Note: This function doesn't filter draft posts, pass it the result of getAllPosts above to do so.
 *  */
export function getUniqueTagsWithCount(posts: CollectionEntry<"post">[]): [string, number][] {
	return [
		...getAllTags(posts).reduce(
			(acc, t) => acc.set(t, (acc.get(t) ?? 0) + 1),
			new Map<string, number>(),
		),
	].sort((a, b) => b[1] - a[1]);
}
