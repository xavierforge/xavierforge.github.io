export interface SiteConfig {
	author: string;
	date: {
		locale: string | string[] | undefined;
		options: Intl.DateTimeFormatOptions;
	};
	description: string;
	lang: string;
	ogLocale: string;
	title: string;
	url: string;
}

/** Giscus comments config (https://giscus.app) */
export interface CommentsConfig {
	category: string;
	categoryId: string;
	inputPosition: "top" | "bottom";
	lang: string;
	mapping: "pathname" | "url" | "title" | "og:title" | "specific" | "number";
	reactionsEnabled: boolean;
	/** "owner/name"; empty string disables comments */
	repo: string;
	repoId: string;
}

/** GoatCounter analytics config (https://www.goatcounter.com) */
export interface AnalyticsConfig {
	/** GoatCounter site code → https://<code>.goatcounter.com; empty disables it */
	code: string;
}

export interface PaginationLink {
	srLabel?: string;
	text?: string;
	url: string;
}

export interface SiteMeta {
	articleDate?: string | undefined;
	description?: string;
	ogImage?: string | undefined;
	title: string;
	/** Page locale; defaults to the site default (zh-Hant). Drives <html lang> + og:locale. */
	locale?: import("@/i18n").Locale | undefined;
	/** hreflang alternates (incl. self + x-default); paths resolved against Astro.site. */
	alternates?: import("@/i18n").Alternate[] | undefined;
	/** RSS feed for this locale, e.g. "/rss.xml" | "/en/rss.xml". */
	rssHref?: string | undefined;
	/** schema.org JSON-LD object (e.g. an Article) emitted in <head>. */
	jsonLd?: Record<string, unknown> | undefined;
}

/** Webmentions */
export interface WebmentionsFeed {
	children: WebmentionsChildren[];
	name: string;
	type: string;
}

export interface WebmentionsCache {
	children: WebmentionsChildren[];
	lastFetched: null | string;
}

export interface WebmentionsChildren {
	author: Author | null;
	content?: Content | null;
	"mention-of": string;
	name?: null | string;
	photo?: null | string[];
	published?: null | string;
	rels?: Rels | null;
	summary?: Summary | null;
	syndication?: null | string[];
	type: string;
	url: string;
	"wm-id": number;
	"wm-private": boolean;
	"wm-property": string;
	"wm-protocol": string;
	"wm-received": string;
	"wm-source": string;
	"wm-target": string;
}

export interface Author {
	name: string;
	photo: string;
	type: string;
	url: string;
}

export interface Content {
	"content-type": string;
	html: string;
	text: string;
	value: string;
}

export interface Rels {
	canonical: string;
}

export interface Summary {
	"content-type": string;
	value: string;
}

export type AdmonitionType = "tip" | "note" | "important" | "caution" | "warning";
