import type { AstroExpressiveCodeOptions } from "astro-expressive-code";
import type { AnalyticsConfig, CommentsConfig, SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	url: "https://xavierforge.dev/",
	title: "Xavier's Data Forge",
	author: "ChihYing Yen",
	description:
		"Rust author and lifelong learner, building and writing at the intersection of systems and ML.",
	// Primary/default locale of the site is Traditional Chinese. Per-page locale
	// (and og:locale) is overridden on the /en/ routes via SiteMeta.locale.
	lang: "zh-Hant",
	ogLocale: "zh_TW",
	date: {
		locale: "en-GB",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
};

// Giscus comments (GitHub Discussions backend). Fill these from https://giscus.app
// after enabling Discussions on the repo and installing the giscus GitHub App.
// See README/setup notes for the one-time steps. Leave `repo` empty to disable.
export const commentsConfig: CommentsConfig = {
	category: "Blog",
	categoryId: "DIC_kwDOJcSeT84C-E8A",
	inputPosition: "bottom",
	lang: "zh-TW",
	// "specific" + a per-post canonical term (the slug, shared by zh & /en/) so both
	// language URLs of a post map to one Giscus/Discussions thread. See Comments.astro.
	mapping: "specific",
	reactionsEnabled: true,
	repo: "xavierforge/xavierforge.github.io",
	repoId: "R_kgDOJcSeTw",
};

// GoatCounter visitor stats. `code` is the subdomain of your GoatCounter site:
// https://<code>.goatcounter.com. Leave empty to disable tracking + counters.
// Requires "Allow adding visitor counts on your website" enabled in GoatCounter
// Settings → Site for the on-page counters to load (CORS).
export const analyticsConfig: AnalyticsConfig = {
	code: "xavierdataforge",
};

export const menuLinks: { path: string; title: string }[] = [
	{ path: "/", title: "Home" },
	{ path: "/posts/", title: "Blog" },
	{ path: "/portfolio/", title: "Portfolio" },
	{ path: "/about/", title: "About" },
];

export const expressiveCodeOptions: AstroExpressiveCodeOptions = {
	styleOverrides: {
		borderRadius: "4px",
		codeFontFamily:
			'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
		codeFontSize: "0.875rem",
		codeLineHeight: "1.7142857rem",
		codePaddingInline: "1rem",
		frames: {
			frameBoxShadowCssValue: "none",
		},
		uiLineHeight: "inherit",
	},
	themeCssSelector(theme, { styleVariants }) {
		if (styleVariants.length >= 2) {
			const baseTheme = styleVariants[0]?.theme;
			const altTheme = styleVariants.find((v) => v.theme.type !== baseTheme?.type)?.theme;
			if (theme === baseTheme || theme === altTheme) return `[data-theme='${theme.type}']`;
		}
		return `[data-theme="${theme.name}"]`;
	},
	themes: ["catppuccin-frappe", "catppuccin-latte"],
	useThemedScrollbars: false,
	shiki: {
		langAlias: {
			Bash: "bash",
			Shell: "bash",
			Sh: "bash",
			Python: "python",
			Py: "python",
			JS: "javascript",
			Js: "javascript",
			TS: "typescript",
			Ts: "typescript",
			Rust: "rust",
			Yaml: "yaml",
			YAML: "yaml",
			Json: "json",
			Md: "markdown",
			Markdown: "markdown",
			Dockerfile: "docker",
		},
	},
};
