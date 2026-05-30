// Regenerate public/logo.png (the source for all build-time favicons) from the
// inline anvil SVG in src/components/layout/Header.astro, so the favicon always
// matches the header logo. Run: node scripts/make-favicon.mjs
//
// Uses the "forge" (rusted-iron) palette — the same fills the logo shows on
// hover / on touch devices.

import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import sharp from "sharp";

const root = fileURLToPath(new URL("..", import.meta.url));
const header = await readFile(`${root}/src/components/layout/Header.astro`, "utf8");

// Pull the <g class="anvil"> … </g> block (excludes the separate sparks group).
const match = header.match(/<g class="anvil"[\s\S]*?<\/g>/);
if (!match) throw new Error("anvil <g> block not found in Header.astro");

// Rusted-iron palette (matches the :hover / @media (hover:none) fills).
const anvil = match[0]
	.replace(/\s*class="anvil"/, "")
	.replaceAll('class="f-top"', 'fill="#c9803f"')
	.replaceAll('class="f-left"', 'fill="#9c5a2d"')
	.replaceAll('class="f-right"', 'fill="#713d1d"');

// Square viewBox centred on the anvil (source content spans x:4–56, y:4–64).
const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="-4 -1 68 68" width="512" height="512">${anvil}</svg>`;

const out = `${root}/public/logo.png`;
await sharp(Buffer.from(svg), { density: 384 }).resize(512, 512).png().toFile(out);
console.log(`wrote ${out}`);
