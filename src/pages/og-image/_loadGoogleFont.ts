import type { SatoriOptions } from "satori";

type Font = SatoriOptions["fonts"][number];

// Matches CJK ideographs, kana, and full-width punctuation — enough to decide
// whether a post title needs a CJK font loaded for the OG image.
const CJK_RE = /[　-〿㐀-䶿一-鿿豈-﫿＀-￯]/;

export function hasCJK(text: string): boolean {
	return CJK_RE.test(text);
}

// Roboto Mono (used for the OG images) has no CJK glyphs, so Chinese titles
// render as blanks. Fetch just the glyphs a title uses from Google Fonts as a
// satori-compatible woff subset. An old User-Agent makes Google serve woff/ttf
// instead of woff2 (which satori can't parse). Returns [] for Latin-only text.
export async function loadCJKFonts(
	text: string,
	weight: NonNullable<Font["weight"]>,
): Promise<Font[]> {
	if (!hasCJK(text)) return [];

	const family = "Noto Sans TC";
	const url = `https://fonts.googleapis.com/css2?family=${encodeURIComponent(family)}:wght@${weight}&text=${encodeURIComponent(text)}`;
	const css = await fetch(url, {
		headers: {
			"User-Agent": "Mozilla/5.0 (Windows NT 10.0; rv:10.0) Gecko/20100101 Firefox/10.0",
		},
	}).then((r) => r.text());

	const fontUrl = css.match(/src:\s*url\((https:\/\/[^)]+)\)/)?.[1];
	if (!fontUrl) throw new Error(`Could not extract a font URL from Google Fonts for "${family}"`);

	const data = await fetch(fontUrl).then((r) => r.arrayBuffer());
	return [{ data, name: family, style: "normal", weight }];
}
