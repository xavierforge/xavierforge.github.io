// Regenerate public/social-card.png — the default Open Graph / Twitter preview
// image used for any page without its own ogImage (see src/components/BaseHead.astro).
// Run: node scripts/make-social-card.mjs
//
// Mirrors the per-post OG style in src/pages/og-image/_ogMarkup.ts (dark bg,
// Roboto Mono, forge-orange footer rule) and adds the anvil logo.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import satori from "satori";
import { html } from "satori-html";
import sharp from "sharp";
import { siteConfig } from "../src/site.config.ts";

const root = fileURLToPath(new URL("..", import.meta.url));

const [robotoReg, robotoBold, logoPng] = await Promise.all([
	readFile(`${root}/src/assets/roboto-mono-regular.ttf`),
	readFile(`${root}/src/assets/roboto-mono-700.ttf`),
	readFile(`${root}/public/logo.png`),
]);

const logo = `data:image/png;base64,${logoPng.toString("base64")}`;

const markup = html`<div tw="flex flex-col w-full h-full bg-[#1d1f21] text-[#c9cacc]">
	<div tw="flex flex-1 items-center w-full p-16">
		<img src="${logo}" tw="w-[220px] h-[220px] mr-14" />
		<div tw="flex flex-col flex-1 min-w-0">
			<h1 tw="text-6xl font-bold text-white leading-tight">${siteConfig.title}</h1>
			<p tw="text-3xl mt-5 leading-snug">${siteConfig.description}</p>
		</div>
	</div>
	<div tw="flex items-center justify-between w-full p-10 border-t-2 border-[#c9803f] text-white">
		<p tw="text-2xl ml-3 font-semibold">${siteConfig.title}</p>
		<p>by ${siteConfig.author}</p>
	</div>
</div>`;

const svg = await satori(markup, {
	width: 1200,
	height: 630,
	fonts: [
		{ data: robotoReg, name: "Roboto Mono", style: "normal", weight: 400 },
		{ data: robotoBold, name: "Roboto Mono", style: "normal", weight: 700 },
	],
});

const out = `${root}/public/social-card.png`;
await sharp(Buffer.from(svg)).png().toFile(out);
console.log(`wrote ${out}`);
