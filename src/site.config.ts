import type { AstroExpressiveCodeOptions } from "astro-expressive-code";
import type { SiteConfig } from "@/types";

export const siteConfig: SiteConfig = {
	url: "https://xavierforge.github.io/",
	title: "ChihYing's Blog",
	author: "ChihYing Yen",
	description: "Lifelong learner and MLOps advocate.",
	lang: "en",
	ogLocale: "en",
	date: {
		locale: "en-GB",
		options: {
			day: "numeric",
			month: "short",
			year: "numeric",
		},
	},
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
	themes: ["dracula", "github-light"],
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
			Json: "json",
			Md: "markdown",
			Markdown: "markdown",
			Dockerfile: "docker",
		},
	},
};
