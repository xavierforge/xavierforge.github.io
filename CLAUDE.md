# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Personal blog at `https://xavierforge.dev/`, built on Astro 6 with a forked copy of the [Astro Cactus](https://github.com/chrismwilliams/astro-theme-cactus) theme. Posts are written in Obsidian (`<vault>/published/*.md`) and synced into the repo before building. The pre-rebuild Hexo project is parked in `legacy/` for reference; nothing in `legacy/` is part of the live site.

## Commands

- `npm install` — install dependencies. Rebuilds `sharp` via the `postinstall` hook.
- `npm run dev` — local dev server (`astro dev`).
- `npm run sync` — `rsync` the Obsidian vault's `published/` folder into `src/content/post/`. Edit `scripts/sync.sh` (or set `VAULT_PUBLISHED`) before first use.
- `npm run build` — `astro build` → `dist/`, then `pagefind` indexes the site into `dist/pagefind/`.
- `npm run preview` — preview the built site.
- `npm run check` — `astro check` + `biome check`.
- `npm run lint` — `biome check --write` (auto-fix).

## Deployment

`.github/workflows/pages.yml` runs on every push to `main` and on manual dispatch:

1. `build` job: `npm ci` → `npm run build` → uploads `./dist` as a Pages artifact.
2. `deploy` job: `actions/deploy-pages@v4` publishes the artifact to GitHub Pages.

One-time setup: GitHub repo → Settings → Pages → Source must be set to "GitHub Actions" (not the deprecated `gh-pages` branch flow). The site is served from the custom domain `xavierforge.dev` (apex): `public/CNAME` pins the domain on every build, apex `A`/`AAAA` records point at GitHub Pages' IPs, and `www` CNAMEs to `xavierforge.github.io`. Keep `siteConfig.url` (which feeds Astro's `site`) in sync with the domain.

## Architecture notes

### Content layer

Three first-party collections, all defined in `src/content.config.ts` via the `glob` loader on the v5 Content Layer API:

- `post` — blog posts under `src/content/post/`. Schema requires `title`, `description`, `publishDate`; supports `coverImage: { src, alt }` (relative path) for the hero image at the top of the post and the homepage Pinned Posts cards. Other fields: `tags`, `draft`, `pinned`, `updatedDate`, `ogImage`. Posts are normally synced from Obsidian — don't hand-edit them in the repo, edit in Obsidian and re-run `npm run sync`.
- `project` — portfolio entries under `src/content/project/`, one `.md` per project. Schema: `title`, `description`, `link` (URL), optional `coverImage`, `order`. Rendered by `src/pages/portfolio.astro`.
- `tag` — optional per-tag metadata files under `src/content/tag/` (currently empty); used by Cactus's tag detail pages. Add a `<tagname>.md` here if you want a description for a tag page.

Cactus's `notes` collection / routes were removed in the rebuild; if you want a digital-garden-style note stream later, restore from `legacy/` history or re-pull from upstream Cactus.

### Obsidian → site bridge

The Obsidian-specific authoring surface is bridged through two custom remark plugins (registered in `astro.config.ts`):

- `src/plugins/remark-obsidian-images.ts` — rewrites bare relative image paths (`assets/Hello/foo.png`) to Astro-friendly (`./assets/Hello/foo.png`) so the asset pipeline picks them up, and treats numeric alt text (`![400](...)`) as a pixel `width` attribute on the rendered `<img>`. External URLs are passed through untouched.
- `src/plugins/remark-obsidian-callouts.ts` — rewrites Obsidian's `> [!note]` / `> [!warning]` / etc. callout blockquotes into the same `<aside class="admonition">` markup as Cactus's `:::note` directive admonitions (`src/plugins/remark-admonitions.ts`), so both share one set of styles. Runs before `remark-breaks`.
- `remark-breaks` (npm) — renders a single newline as a `<br>` (matching Obsidian), so soft-wrapped lines don't collapse into one paragraph.

Image path convention from Obsidian: `published/<title>.md` + `published/assets/<title>/<file>.png`. Sync mirrors the whole `published/` tree to `src/content/post/`, so relative `assets/<title>/<file>.png` paths just work.

### Layout split

- `src/layouts/Base.astro` — shell (head, header, footer, theme provider).
- `src/layouts/BlogPost.astro` — wraps a post with `Masthead` (renders `coverImage` Notion-style on top), TOC, prose body, back-to-top button.
- `src/pages/index.astro` — homepage. Pinned Posts render with `PinnedPostCard` (thumbnail card grid); the chronological list uses `PostPreview` (text-only row).
- `src/pages/portfolio.astro` — projects collection, card grid.
- `src/pages/about.astro` — static About content.
- `src/site.config.ts` — site title, author, description, URL, date locale, and `menuLinks` for the nav.

### Styling

Tailwind v4 via `@tailwindcss/vite`. Theme tokens (colors, spacing) come from Cactus's CSS in `src/styles/`. `vite` is pinned to `7.3.3` in `package.json` `overrides` because `@tailwindcss/vite` 4.3.0 chokes on the resolver shape in newer vite/rolldown releases — bump both together when upgrading.

## Things to avoid

- Don't edit posts under `src/content/post/` directly — they get clobbered by the next `npm run sync`. Source of truth is the Obsidian vault.
- Don't put anything important in `legacy/` — it exists only as a historical reference of the old Hexo blog and will eventually be deleted.
- Don't reintroduce a `gh-pages` branch deploy — the Pages source is set to GitHub Actions.
