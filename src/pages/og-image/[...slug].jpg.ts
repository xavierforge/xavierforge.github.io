import fs from "node:fs";
import type { APIContext, ImageMetadata, InferGetStaticPropsType } from "astro";
import satori, { type SatoriOptions } from "satori";
import sharp from "sharp";
import RobotoMonoBold from "@/assets/roboto-mono-700.ttf";
import RobotoMono from "@/assets/roboto-mono-regular.ttf";
import { getPostsByLocale } from "@/data/post";
import { getFormattedDate } from "@/utils/date";
import { readCache, writeToCache } from "./_cacheUtil";
import { loadCJKFonts } from "./_loadGoogleFont";
import { ogMarkup } from "./_ogMarkup";

const ogOptions: SatoriOptions = {
	// debug: true,
	fonts: [
		{
			data: Buffer.from(RobotoMono),
			name: "Roboto Mono",
			style: "normal",
			weight: 400,
		},
		{
			data: Buffer.from(RobotoMonoBold),
			name: "Roboto Mono",
			style: "normal",
			weight: 700,
		},
	],
	height: 630,
	width: 1200,
};

type Props = InferGetStaticPropsType<typeof getStaticPaths>;

// Render the post's cover, cropped to the 1200×630 OG canvas, as a JPEG.
// Covers are photographs — lossless PNG ran ~2 MB (over the 1 MB social-scraper
// warning threshold); mozjpeg q80 lands ~200 KB with no visible loss at OG size.
async function renderCover(coverPath: string): Promise<Buffer> {
	return sharp(coverPath)
		.resize(1200, 630, { fit: "cover", position: "centre" })
		.jpeg({ quality: 80, mozjpeg: true })
		.toBuffer();
}

// The Cactus title-card fallback (Satori → SVG → JPEG) for cover-less posts.
// Flat card with crisp text → keep quality high so edges stay clean.
async function renderTitleCard(title: string, pubDate: Date): Promise<Buffer> {
	const postDate = getFormattedDate(pubDate, {
		month: "long",
		weekday: "long",
	});
	// Only the (bold) title can contain CJK; load a matching glyph subset.
	const cjkFonts = await loadCJKFonts(title, 700);
	const svg = await satori(ogMarkup(title, postDate), {
		...ogOptions,
		fonts: [...ogOptions.fonts, ...cjkFonts],
	});
	return sharp(Buffer.from(svg)).jpeg({ quality: 92, mozjpeg: true }).toBuffer();
}

export async function GET(context: APIContext) {
	const { coverPath, pubDate, title } = context.props as Props;

	// A cover-based image gets a variant key (path + mtime) so it never collides
	// with the title-card cache and re-renders when the cover file changes.
	const variant = coverPath ? `cover:${coverPath}:${fs.statSync(coverPath).mtimeMs}` : "";

	let imageBuffer = readCache(title, pubDate, variant);
	if (!imageBuffer) {
		console.info(`Generating new OG image for: ${title}`);
		imageBuffer = coverPath ? await renderCover(coverPath) : await renderTitleCard(title, pubDate);
		writeToCache(title, pubDate, imageBuffer, variant);
	}

	return new Response(new Uint8Array(imageBuffer), {
		headers: {
			"Cache-Control": "public, max-age=31536000, immutable",
			"Content-Type": "image/jpeg",
		},
	});
}

export async function getStaticPaths() {
	// Both locales: zh posts key off `<slug>` and English companions off `<slug>.en`,
	// so each /en/ page gets a card rendered with its own (English) title. Only the
	// getStaticPaths input set changes — the card markup (_ogMarkup) is untouched.
	const posts = [...(await getPostsByLocale("zh-Hant")), ...(await getPostsByLocale("en"))];
	return posts
		.values()
		.filter(({ data }) => !data.ogImage)
		.map((post) => ({
			params: { slug: post.id },
			props: {
				// `image()` populates `fsPath` (the source file on disk) at runtime;
				// it's omitted from the public ImageMetadata type, hence the cast.
				coverPath: (post.data.coverImage?.src as (ImageMetadata & { fsPath?: string }) | undefined)
					?.fsPath,
				pubDate: post.data.updatedDate ?? post.data.publishDate,
				title: post.data.title,
			},
		}))
		.toArray();
}
