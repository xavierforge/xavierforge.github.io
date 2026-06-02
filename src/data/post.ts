import { type CollectionEntry, getCollection } from "astro:content";
import { canonicalSlug, type Locale } from "@/i18n";

/** A post's locale, from either signal:
 *  - filename `.en.md` suffix → an English *translation* of a `<slug>.md` original
 *  - frontmatter `lang: en`   → an English-only *original* (authored in the vault)
 *  Everything else is the default locale (Chinese). */
function localeOf({ id, data }: CollectionEntry<"post">): Locale {
	return id.endsWith(".en") || data.lang === "en" ? "en" : "zh-Hant";
}

/** Posts for a given locale, with draft/PROD filtering. */
export async function getPostsByLocale(locale: Locale): Promise<CollectionEntry<"post">[]> {
	return await getCollection("post", (entry) => {
		if (localeOf(entry) !== locale) return false;
		return import.meta.env.PROD ? !entry.data.draft : true;
	});
}

/** The Chinese (default-locale) posts — what every zh listing/route consumes
 *  (index, /posts, tags, RSS, og-image, the [...slug] builder). Excludes the
 *  `.en.md` translations and any `lang: en` English-only originals. */
export async function getAllPosts(): Promise<CollectionEntry<"post">[]> {
	return await getPostsByLocale("zh-Hant");
}

/** Look up the English translation companion for a (Chinese) post id, if one exists. */
export async function getPostTranslation(id: string): Promise<CollectionEntry<"post"> | undefined> {
	const matches = await getCollection("post", (entry) => entry.id === `${id}.en`);
	return matches[0];
}

/** The companion entry of `post` in the other locale, if it exists.
 *  zh original `foo` → its translation `foo.en`; en translation `foo.en` → its
 *  source `foo`. An English-only original (`lang: en`, no `.en` suffix) has no
 *  companion → undefined. Drives the conditional `hreflang` + the language toggle. */
export async function getAlternateLocalePost(
	post: CollectionEntry<"post">,
): Promise<CollectionEntry<"post"> | undefined> {
	let otherId: string;
	if (post.id.endsWith(".en")) {
		otherId = canonicalSlug(post.id); // en translation → its zh source
	} else if (post.data.lang === "en") {
		return undefined; // en-only original → no Chinese companion
	} else {
		otherId = `${post.id}.en`; // zh original → its en translation
	}
	const matches = await getCollection("post", (entry) => entry.id === otherId);
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
