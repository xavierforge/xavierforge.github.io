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

### Bilingual posts + `/en/` URL split (i18n)

Each language is served on its **own URL** for clean SEO (separate indexable pages + `hreflang`), not a same-URL CSS toggle. There are **three kinds of post**, distinguished by filename suffix and a `lang` frontmatter field:

| Kind | File | Authored in | `lang` frontmatter | `:::caution[AI-translated]` | drift-tracked |
|------|------|-------------|--------------------|-----------------------------|---------------|
| Chinese original | `<slug>.md` | Obsidian vault (synced) | omit (default `zh-Hant`) | no | — |
| **English-only original** | `<slug>.md` | Obsidian vault (synced) | **`lang: en`** | no | — |
| English translation of a Chinese post | `<slug>.en.md` | **repo only** (never vault) | omit (locale from suffix) | AI → yes, human → **no** | yes (`sourceHash`) |

So the `.en.md` suffix means "translation of `<slug>.md`"; an English-only **original** is a normal `<slug>.md` carrying `lang: en` (authored in the vault like any post, synced normally, ignored by the drift checker which only scans `*.en.md`). A **human** translation omits the AI-translated disclaimer (it's not machine-made). Key pieces:

- **URL structure:** Chinese is the default locale with **no prefix** (`/posts/<slug>/`, `/`, `/about/`, …); English mirrors live under **`/en/`** (`/en/posts/<slug>/`, `/en/`, `/en/about/`, …). The `/en/` URL uses the **canonical slug** (no `.en`). `astro.config.ts` has a minimal `i18n` block (`defaultLocale: "zh-Hant"`, `prefixDefaultLocale: false`); the split is driven by **static mirror routes** under `src/pages/en/`, not middleware (GitHub Pages is static). **Don't change Chinese slugs/URLs** — external links/SEO/Medium canonicals point at them.
- **Locale config + helpers — `src/i18n.ts`:** single source of truth. `LOCALES` maps each locale → `htmlLang` / `ogLocale` / URL `prefix` / toggle glyph. Helpers: `canonicalSlug(id)` (strip `.en`), `localizePath(path, locale)`, `postPath(slug, locale)`, `rssPath(locale)`, and `buildAlternates({zh, en})` (emits the `hreflang` set incl. `x-default`, **omitting any locale that doesn't exist** → P3: never advertise a missing translation).
- **Data layer — `src/data/post.ts`:** the locale of an entry comes from `localeOf()` — `.en` suffix **or** `lang: en` → English, else Chinese. `getPostsByLocale(locale)` filters on that (+ drafts) and is what every route uses; `getAllPosts()` is just `getPostsByLocale("zh-Hant")`. `getAlternateLocalePost(post)` finds the companion in the other locale — a zh original ↔ its `.en` translation; an English-only original (`lang: en`, no `.en`) has none → no `hreflang` alternate. Drives the conditional `hreflang` + the toggle target.
- **Metadata — `SiteMeta` (`src/types.ts`) carries `locale` + `alternates` + `rssHref`.** `Base.astro` sets `<html lang>` from the locale and passes `locale` to `Header`; `BaseHead.astro` emits per-locale `og:locale`, self `canonical`, the `hreflang` alternates (resolved against `Astro.site`), and the locale's RSS link. Each page (zh routes + `/en/` mirrors) passes `meta.locale` + `meta.alternates` (built with `buildAlternates`). Post routes compute the alternate only when `getAlternateLocalePost` finds a companion.
- **Single-language rendering (no CSS toggle):** there is **no** `data-lang` show/hide rule, `LangProvider`, or dual-body panel anymore — each URL renders one language. `BlogPost.astro` has a single body `<slot>`; `src/pages/posts/[...slug].astro` (zh) and `src/pages/en/posts/[...slug].astro` (en) each render their own body. Listing components (`PostPreview`, `PinnedPostCard`, `Masthead`) take a `locale` prop, render the single title/description of the entry they're given, and link via `postPath(canonicalSlug(post.id), locale)`. The `/en/` mirror routes source posts via `getPostsByLocale("en")`.
- **Header language toggle = navigation.** `LangToggle.astro` is a plain `<a>` to the alternate-locale URL (computed in `Base.astro` from `alternates`, falling back to the other locale's home). The cross-fade is free via the existing `@view-transition { navigation: auto }` rule in `global.css` (MPA transitions) + the speculation-rules prefetch in `Base.astro`. No JS, no `localStorage` — the URL is the language memory.
- **OG images per language:** `src/pages/og-image/[...slug].png.ts` iterates **both** locales, so `/og-image/<slug>.en.png` renders with the English title. Only its `getStaticPaths` input changed — the card markup (`_ogMarkup.ts`) is untouched.
- **Comments — one shared thread across languages:** `commentsConfig.mapping` is `"specific"` and `BlogPost.astro` passes `term={canonicalSlug(post.id)}` to `Comments.astro` (emitted as `data-term`), so `/posts/<slug>/` and `/en/posts/<slug>/` map to the **same** Giscus/Discussions thread.
- **Visitor stats split per URL (intentional):** GoatCounter counts `/posts/<slug>/` and `/en/posts/<slug>/` separately (no change needed); the footer site-wide `TOTAL` is unaffected.
- **Survives sync:** `scripts/sync.sh` excludes `*.en.md` from the `rsync --delete`, so the vault mirror never wipes translations. Same-folder placement means image paths (`assets/<slug>/…`) and the `![alt|600](…)` width syntax resolve identically to the Chinese post — no path rewrites.
- **Id convention:** `src/content.config.ts` gives the `post` loader a custom `generateId` that preserves the file path (the default slugifies, collapsing `foo.en.md` → `fooen`). So the translation's id is `<slug>.en`.
- **AI-translation disclaimer:** each `.en.md` starts (right after the frontmatter, before the body) with a `:::caution[AI-translated]` admonition directive whose second sentence — the "leave a comment" line — is its own paragraph. It lives in the markdown (not the layout) so it renders inline like other admonitions; keep this block identical across all translations.
- **Drift check:** each `.en.md` records a `sourceHash` of the Chinese body it was translated from. `npm run check:translations` (warn-only; `--strict` to gate) flags translations whose source has changed; `node scripts/check-translations.mjs src/content/post --update` re-stamps after (re)translating. Run it before pushing to see which originals drifted.
- **Producing/updating translations:** do it **one subagent per post**, each from a clean context, given only the translation guidelines (keep code/inline-code/commands/links/image paths/math verbatim; translate prose, image alt, link text; `title` ≤ 60 chars; body starts with the shared `:::caution[AI-translated]` block). Then `--update` to stamp, build-verify (the `/en/` page renders, caution + images resolve), and commit. Per-post fresh context is a deliberate quality choice.

- **hreflang is auto-maintained — never hand-write it.** Every signal (per-page `<head>` hreflang/canonical, the language toggle target, and the sitemap alternates) is **derived at build time from which files exist**. To make a post bilingual you only create the file (a `.en.md` translation, or a `lang: en` original); the next build wires up the reciprocal hreflang on both pages, the toggle, and the sitemap automatically — and emits nothing for a language that doesn't exist (P3). Sitemap `hreflang` comes from `@astrojs/sitemap`'s `i18n` option in `astro.config.ts` (groups pages by their locale-stripped path).

### SEO plan status

All of `~/Downloads/xavierforge-seo-plan.md` is shipped: **P0** i18n routing, **P1** metadata consistency, **P2** sitemap hreflang, **P3** conditional hreflang, **P4** JSON-LD `Article`, **P5** robots/404/links/alt.

- **P4 — JSON-LD:** `BlogPost.astro` builds a schema.org `Article` (headline/description/image/dates/author/publisher/`inLanguage`) and passes it via `SiteMeta.jsonLd`; `BaseHead.astro` emits it as `<script type="application/ld+json">` (escaping `<`). Post pages only — listings/home don't carry it.
- **P5 — robots.txt** is already correct (the `astro-robots-txt` integration emits `Sitemap:` automatically); **404** is a single bilingual page with nav links (GitHub Pages can't serve a per-locale 404).
- **Remaining content fixes (do in the Obsidian vault, not the repo):** a few Chinese posts link to siblings with absolute `https://xavierforge.dev/posts/...` URLs (prefer root-relative `/posts/...` so a domain change doesn't break them), and `numba-cuda-puzzles-1` has one image with an empty `![]()` alt (line ~377) that wants a description. The English `.en.md` cross-links were already fixed to root-relative `/en/posts/...`.

### Layout split

- `src/layouts/Base.astro` — shell (head, header, footer, theme provider).
- `src/layouts/BlogPost.astro` — wraps a post with `Masthead` (renders `coverImage` Notion-style on top), TOC, prose body, back-to-top button.
- `src/pages/index.astro` — homepage. Pinned Posts render with `PinnedPostCard` (thumbnail card grid); the chronological list uses `PostPreview` (text-only row).
- `src/pages/portfolio.astro` — projects collection, card grid.
- `src/pages/about.astro` — static About content, **Chinese** prose. The English version is its mirror `src/pages/en/about.astro` (English prose, `locale: "en"`) — see "Bilingual posts + `/en/` URL split". The hero photo uses `astro:assets` `<Image>` (don't revert to a raw `<img src={photo.src}>` — that ships the full-size source, ~4.9 MB, unoptimized).
- `src/pages/en/` — English mirror routes (`index`, `posts/[...page]`, `posts/[...slug]`, `tags/index`, `tags/[tag]/[...page]`, `about`, `portfolio`, `rss.xml`). Thin wrappers over the same components/layouts with `locale: "en"`, sourcing posts via `getPostsByLocale("en")`.
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
  **`mapping` is `"specific"`**: `BlogPost.astro` passes `term={canonicalSlug(post.id)}` so a
  post's Chinese and `/en/` URLs share one discussion thread (see the bilingual section).
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
- Don't commit posts with `draft: true` — the repo is **public**, so committing the markdown leaks WIP source even though the built site hides drafts (`src/data/post.ts` filters them in PROD). When staging synced content, add finished posts individually rather than `git add`-ing the whole `src/content/post/` tree, and confirm `git status` still shows the draft(s) untracked.
