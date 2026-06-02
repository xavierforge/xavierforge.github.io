// Single source of truth for the bilingual URL split.
//
// Chinese is the default locale and keeps its existing prefix-less URLs
// (`/posts/<slug>/`); English lives under `/en/`. Each rendered page carries
// exactly one language (see the `/en/` mirror routes), so the metadata below
// (`<html lang>`, `og:locale`, `hreflang`) can be unambiguous per URL.

export type Locale = "zh-Hant" | "en";

export const DEFAULT_LOCALE: Locale = "zh-Hant";

export const LOCALES: Record<
	Locale,
	{
		/** value for `<html lang>` */
		htmlLang: string;
		/** value for `og:locale` */
		ogLocale: string;
		/** URL path prefix ("" for the default locale) */
		prefix: string;
		/** glyph shown on the header toggle — the language you switch *to* */
		toggleLabel: string;
		/** accessible label for that toggle */
		toggleAriaLabel: string;
	}
> = {
	"zh-Hant": {
		htmlLang: "zh-Hant",
		ogLocale: "zh_TW",
		prefix: "",
		toggleLabel: "EN",
		toggleAriaLabel: "Switch to English",
	},
	en: {
		htmlLang: "en",
		ogLocale: "en_US",
		prefix: "/en",
		toggleLabel: "中",
		toggleAriaLabel: "切換為中文",
	},
};

/** The canonical slug shared by a zh post and its `.en` companion. */
export const canonicalSlug = (id: string): string => id.replace(/\.en$/, "");

/** Prefix a site-absolute path for a locale: `/about/` + en → `/en/about/`. */
export const localizePath = (path: string, locale: Locale): string => {
	const { prefix } = LOCALES[locale];
	if (!prefix) return path;
	return path === "/" ? `${prefix}/` : `${prefix}${path}`;
};

/** Post URL for a canonical slug in a locale: zh → `/posts/<slug>/`, en → `/en/posts/<slug>/`. */
export const postPath = (slug: string, locale: Locale): string =>
	localizePath(`/posts/${slug}/`, locale);

/** RSS feed path for a locale. */
export const rssPath = (locale: Locale): string => localizePath("/rss.xml", locale);

export interface Alternate {
	hreflang: string;
	/** site-absolute path; BaseHead resolves it against `Astro.site` */
	href: string;
}

/**
 * Build the `hreflang` set for a page given whichever locale paths exist.
 * Emits self + alternates + `x-default` (pointing at Chinese when present,
 * else the only available language). Omitting an absent locale satisfies P3:
 * never advertise a translation that doesn't exist.
 */
export const buildAlternates = (paths: { zh?: string | null; en?: string | null }): Alternate[] => {
	const out: Alternate[] = [];
	if (paths.zh) out.push({ hreflang: "zh-Hant", href: paths.zh });
	if (paths.en) out.push({ hreflang: "en", href: paths.en });
	const xDefault = paths.zh ?? paths.en;
	if (xDefault) out.push({ hreflang: "x-default", href: xDefault });
	return out;
};
