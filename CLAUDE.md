# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Personal blog at `https://xavierforge.dev/`, built on Astro 6 with a forked copy of the [Astro Cactus](https://github.com/chrismwilliams/astro-theme-cactus) theme. Posts are written in Obsidian (`<vault>/published/*.md`) and synced into the repo before building. The pre-rebuild Hexo project is parked in `legacy/` for reference; nothing in `legacy/` is part of the live site.

## Commands

- `npm install` — install dependencies. Rebuilds `sharp` via the `postinstall` hook.
- `npm run dev` — local dev server (`astro dev`). **Caveat:** after changing markdown rendering (remark/rehype plugins in `astro.config.ts`, or what they import like `src/site.config.ts`), `astro dev` often keeps serving the OLD rendered HTML even after clearing `node_modules/.astro` + `node_modules/.vite` and restarting — a wedged `@pagefind/component-ui` dep re-optimization freezes the SSR render. Treat `npm run build` + grepping `dist/**/index.html` as the source of truth for whether a rendering change works. For a clean dev preview: `pkill -9 -f "astro dev"`, `rm -rf node_modules/.astro node_modules/.vite`, restart on a fresh port (`npm run dev -- --port 4399`). (`lsof -ti :4321` may also list the browser's client connection, not a zombie server — check the command name.)
- `npm run sync` — `rsync` the Obsidian vault's `published/` folder into `src/content/post/`. Edit `scripts/sync.sh` (or set `VAULT_PUBLISHED`) before first use. Runs the whitespace check (below) against the vault source first, warn-only.
- `npm run check:whitespace` — scan `src/content/post/` for invisible/odd whitespace (see "Source markdown hygiene"). Pass flags through npm: `-- --verbose` lists hidden info findings, `-- --strict` exits non-zero on any issue.
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

- `post` — blog posts under `src/content/post/`. Schema requires `title` (≤ 60 chars), `description`, `publishDate`; supports `coverImage: { src, alt }` (relative path) for the hero image at the top of the post and the homepage Pinned Posts cards. Other fields: `tags`, `draft`, `pinned`, `updatedDate`, `ogImage`. Posts are normally synced from Obsidian — don't hand-edit them in the repo, edit in Obsidian and re-run `npm run sync`.
  - **Cover image spec:** the hero (`src/components/blog/Masthead.astro`) renders full-width at a fixed **16:9** ratio (`aspect-video … object-cover`), so non-16:9 art gets cropped. Author covers at **16:9, ideally 1920×1080** (min ~1600×900 — the content column is `max-w-5xl` ≈ 960px, so 2× stays sharp on HiDPI). PNG or WebP both fine; Astro re-encodes to WebP and emits a srcset via the `image()` schema, so don't hardcode width/height in Masthead.
- `project` — portfolio entries under `src/content/project/`, one `.md` per project. Schema: `title`, `description`, `link` (URL), optional `coverImage`, `order`. Rendered by `src/pages/portfolio.astro`.
- `tag` — optional per-tag metadata files under `src/content/tag/` (currently empty); used by Cactus's tag detail pages. Add a `<tagname>.md` here if you want a description for a tag page.

Cactus's `notes` collection / routes were removed in the rebuild; if you want a digital-garden-style note stream later, restore from `legacy/` history or re-pull from upstream Cactus.

### Obsidian → site bridge

The Obsidian-specific authoring surface is bridged through two custom remark plugins (registered in `astro.config.ts`):

- `src/plugins/remark-obsidian-images.ts` — rewrites bare relative image paths (`assets/Hello/foo.png`) to Astro-friendly (`./assets/Hello/foo.png`) so the asset pipeline picks them up, and treats numeric alt text (`![400](...)`) as a pixel `width` attribute on the rendered `<img>`. External URLs are passed through untouched.
- `src/plugins/remark-obsidian-callouts.ts` — rewrites Obsidian's `> [!note]` / `> [!warning]` / etc. callout blockquotes into the same `<aside class="admonition">` markup as Cactus's `:::note` directive admonitions (`src/plugins/remark-admonitions.ts`), so both share one set of styles. Runs before `remark-breaks`.
- `remark-breaks` (npm) — renders a single newline as a `<br>` (matching Obsidian), so soft-wrapped lines don't collapse into one paragraph.

Image path convention from Obsidian: `published/<title>.md` + `published/assets/<title>/<file>.png`. Sync mirrors the whole `published/` tree to `src/content/post/`, so relative `assets/<title>/<file>.png` paths just work.

### Source markdown hygiene

Pasting into Obsidian (from LLM output, web pages, Word) drags in invisible/odd whitespace — hair spaces (U+200A), no-break spaces (U+00A0), zero-width chars — that render as stray gaps. `scripts/check-whitespace.mjs` (Node, no deps) scans for these and is wired into `sync.sh` as a warn-only pre-sync step (never blocks a sync). It is tuned for this blog's conventions:

- **A single thin space hugging an em/en dash (`字—字`, U+200A on each side) is an intentional typographic style — do NOT strip it.** The checker allows it; only anomalies are flagged: zero-width chars, runs of 2+ special spaces (the "doubled hair space" bug), and stray thin spaces not adjacent to a dash. A single NBSP (used to keep e.g. `Map 2D` together) is info-only, hidden unless `--verbose`.
- Fenced code blocks are skipped (pasted Python is full of trailing spaces that aren't the point).
- Fix findings in the **Obsidian vault** source, not the repo copy. The `sync` step scans the vault; `npm run check:whitespace` scans the synced `src/content/post/`.

### Bilingual posts (in-page EN/中文 toggle)

Posts are authored in Chinese (the vault is the source of truth). An optional English translation lives **in the repo only** as a sibling `src/content/post/<slug>.en.md` — never in the vault. Key pieces:

- **Survives sync:** `scripts/sync.sh` excludes `*.en.md` from the `rsync --delete`, so the vault mirror never wipes translations. Same-folder placement means image paths (`assets/<slug>/…`) and the `![alt|600](…)` width syntax resolve identically to the Chinese post — no path rewrites.
- **Id convention:** `src/content.config.ts` gives the `post` loader a custom `generateId` that preserves the file path (the default slugifies, collapsing `foo.en.md` → `fooen`). So the translation's id is `<slug>.en`.
- **Hidden from listings/routes:** `getAllPosts()` in `src/data/post.ts` filters out `id.endsWith(".en")`, which covers every surface at once (index, `/posts`, tags, RSS, og-image, and the `[...slug]` page builder). `getPostTranslation(id)` looks up the `<slug>.en` companion.
- **Rendering:** `src/pages/posts/[...slug].astro` renders both bodies into named slots (`body-zh`, `body-en`); `BlogPost.astro` wraps them in `data-lang` panels inside a `data-lang-scope` and shows the reusable `LangToggle.astro` control (default 中文, persisted in `localStorage["post-lang"]`) **only when a translation exists**. Posts without a `.en.md` show no toggle. The post `<h1>` and TOC stay in the original language.
- **AI-translation disclaimer:** each `.en.md` starts (right after the frontmatter, before the body) with a `:::caution[AI-translated]` admonition directive whose second sentence — the "leave a comment" line — is its own paragraph. It lives in the markdown (not the layout) so it renders inline like other admonitions; keep this block identical across all translations.
- **Drift check:** each `.en.md` records a `sourceHash` of the Chinese body it was translated from. `npm run check:translations` (warn-only; `--strict` to gate) flags translations whose source has changed; `node scripts/check-translations.mjs src/content/post --update` re-stamps after (re)translating. Run it before pushing to see which originals drifted.
- **Producing/updating translations:** do it **one subagent per post**, each from a clean context, given only the translation guidelines (keep code/inline-code/commands/links/image paths/math verbatim; translate prose, image alt, link text; `title` ≤ 60 chars; body starts with the shared `:::caution[AI-translated]` block). Then `--update` to stamp, build-verify (toggle + caution render, images resolve), and commit. Per-post fresh context is a deliberate quality choice.

### Layout split

- `src/layouts/Base.astro` — shell (head, header, footer, theme provider).
- `src/layouts/BlogPost.astro` — wraps a post with `Masthead` (renders `coverImage` Notion-style on top), TOC, prose body, back-to-top button.
- `src/pages/index.astro` — homepage. Pinned Posts render with `PinnedPostCard` (thumbnail card grid); the chronological list uses `PostPreview` (text-only row).
- `src/pages/portfolio.astro` — projects collection, card grid.
- `src/pages/about.astro` — static About content. Bilingual: an EN/中文 segmented-control toggle swaps two `prose` panels (`data-lang="en"` / `data-lang="zh"`); a small client `<script>` flips `hidden`, styles the active button, and persists the choice in `localStorage["about-lang"]` (defaults to English, the site lang). The hero photo uses `astro:assets` `<Image>` (don't revert to a raw `<img src={photo.src}>` — that ships the full-size source, ~4.9 MB, unoptimized).
- `src/site.config.ts` — site title, author, description, URL, date locale, and `menuLinks` for the nav.

### Comments & visitor stats

Both are opt-in via `src/site.config.ts` and render nothing until configured, so the
site builds clean with empty values.

- **Comments — Giscus** (GitHub Discussions backend). `src/components/Comments.astro` is
  rendered at the bottom of `BlogPost.astro`; it lazy-loads `giscus.app/client.js` and
  syncs its light/dark theme to the site's `theme-change` event. One-time setup: on the
  repo, Settings → enable **Discussions**, create a category (e.g. *Announcements*),
  install the [giscus GitHub App](https://github.com/apps/giscus) with access to the repo,
  then visit <https://giscus.app>, enter `owner/name`, and copy the generated `repo`,
  `repoId`, `category`, `categoryId` into `commentsConfig`. Leave `repo` empty to disable.
- **Visitor stats — GoatCounter**. The tracking pixel (`gc.zgo.at/count.js`) loads from
  `Base.astro` **only in PROD** (`import.meta.env.PROD`) so dev/preview hits aren't counted.
  `src/components/GoatCounter.astro` is a reusable on-page counter that fetches
  `/counter/<path>.json` client-side: `path="TOTAL" metric="count_unique"` for site-wide UV
  in the footer, and the default (current page path) `metric="count"` for per-post PV in
  `Masthead.astro`. Setup: register a free site at <https://www.goatcounter.com>, put its
  subdomain in `analyticsConfig.code` (`https://<code>.goatcounter.com`), and in GoatCounter
  Settings → *Site* enable **"Allow adding visitor counts on your website"** (needed for the
  on-page counters' CORS). Counters stay hidden until data exists / the endpoint responds.
  Two non-obvious `/counter/` quirks that broke per-post views (both handled in
  `GoatCounter.astro`, but worth knowing when debugging "counter shows nothing"):
  1. **Trailing-slash normalization** — GoatCounter stores paths *without* a trailing slash
     (`/posts/foo`), but the deployed pages + tracking pixel use `/posts/foo/`. Querying the
     raw `location.pathname` 404s, so the component strips the trailing slash (keeping root `/`).
  2. **The public `/counter/` endpoint updates on a periodic cycle (~hourly), not real-time** —
     the *dashboard* is real-time, so a freshly recorded path (labelled "(new)") can 404 from
     `/counter/` for a while even though the data exists. When debugging, compare the dashboard's
     stored path string against what the component queries, and allow for the endpoint's lag
     before concluding the hit wasn't recorded.

### Styling

Tailwind v4 via `@tailwindcss/vite`. Theme tokens (colors, spacing) come from Cactus's CSS in `src/styles/`. `vite` is pinned to `7.3.3` in `package.json` `overrides` because `@tailwindcss/vite` 4.3.0 chokes on the resolver shape in newer vite/rolldown releases — bump both together when upgrading.

## Things to avoid

- Don't edit posts under `src/content/post/` directly — they get clobbered by the next `npm run sync`. Source of truth is the Obsidian vault.
- Don't put anything important in `legacy/` — it exists only as a historical reference of the old Hexo blog and will eventually be deleted.
- Don't reintroduce a `gh-pages` branch deploy — the Pages source is set to GitHub Actions.
