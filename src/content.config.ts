import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

function removeDupsAndLowerCase(array: string[]) {
	return [...new Set(array.map((str) => str.toLowerCase()))];
}

const titleSchema = z.string().max(60);

const baseSchema = z.object({
	title: titleSchema,
});

const post = defineCollection({
	loader: glob({
		base: "./src/content/post",
		pattern: "**/*.{md,mdx}",
		// Preserve the file path as the id. The default generateId slugifies, which
		// drops the dot and collapses `foo.en.md` → `fooen`; we rely on the `.en`
		// suffix to detect/route-exclude in-page translations (see src/data/post.ts).
		// Existing posts are already slug-clean, so their ids/URLs are unchanged.
		generateId: ({ entry }) => entry.replace(/\.(md|mdx)$/, ""),
	}),
	schema: ({ image }) =>
		baseSchema.extend({
			description: z.string(),
			coverImage: z
				.object({
					alt: z.string(),
					src: image(),
				})
				.optional(),
			draft: z.boolean().default(false),
			// Locale of an *original* post. Omit (default zh-Hant) for Chinese posts and
			// for `.en.md` translation companions (their locale comes from the suffix).
			// Set `lang: en` on a normal `<slug>.md` to author an English-only original in
			// the vault — it routes to /en/ and stays off the Chinese site. See post.ts.
			lang: z.enum(["zh-Hant", "en"]).optional(),
			ogImage: z.string().optional(),
			tags: z.array(z.string()).default([]).transform(removeDupsAndLowerCase),
			publishDate: z
				.string()
				.or(z.date())
				.transform((val) => new Date(val)),
			updatedDate: z
				.string()
				.or(z.date())
				.optional()
				.transform((val) => (val ? new Date(val) : undefined)),
			pinned: z.boolean().default(false),
		}),
});

const tag = defineCollection({
	loader: glob({ base: "./src/content/tag", pattern: "**/*.{md,mdx}" }),
	schema: z.object({
		title: titleSchema.optional(),
		description: z.string().optional(),
	}),
});

const project = defineCollection({
	loader: glob({ base: "./src/content/project", pattern: "**/*.{md,mdx}" }),
	schema: ({ image }) =>
		baseSchema.extend({
			description: z.string(),
			link: z.string().url(),
			coverImage: z
				.object({
					alt: z.string(),
					src: image(),
				})
				.optional(),
			order: z.number().default(0),
		}),
});

export const collections = { post, tag, project };
