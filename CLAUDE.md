# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

[Hexo](https://hexo.io) static blog published at `https://xavierforge.github.io/`. Content lives in `source/_posts/` (Markdown with front-matter); the site is rendered with the `miccall` theme under `themes/miccall/`.

## Commands

- `npm install` — install Hexo and plugin dependencies (run once after clone).
- `npm run server` — local dev server (hexo server), default `http://localhost:4000`. Use this to preview changes.
- `npm run build` — generate the static site into `public/`.
- `npm run clean` — wipe `public/` and `db.json`; run if rendered output looks stale.
- `npm run deploy` — push `public/` via `hexo-deployer-git` to the `gh-pages` branch of the configured repo (see `deploy:` in `_config.yml`). Note: deployment normally happens automatically through CI — only run this manually if bypassing CI.
- New post: `npx hexo new "Post Title"` creates `source/_posts/Post-Title.md` from `scaffolds/post.md`.

## Deployment

`.github/workflows/pages.yml` runs on every push to `main`: it installs deps, runs `npm run build`, and publishes `./public` to the `gh-pages` branch with `peaceiris/actions-gh-pages`. GitHub Pages serves the site from that branch. Editing content and pushing to `main` is the normal release path — there is no separate deploy step.

## Architecture notes

Two config layers, both matter:

- `_config.yml` (site root) — Hexo-level config: site metadata, permalink scheme (`:year/:month/:day/:title/`), `theme: miccall`, `prismjs` highlighting (the built-in `highlight` is disabled in favor of Prism), and the `deploy:` target (used by `hexo deploy`, but CI bypasses this and pushes directly).
- `themes/miccall/_config.yml` — theme-level config: nav, intro/profile copy, Prism color scheme (`prism: night_owl`), comment system (`disqus_click`, shortname `xavierforge`), search, MathJax, and visitor counters (`busuanzi`).

The `miccall` theme is vendored in-tree (not an npm dep). Layouts in `themes/miccall/layout/` (`index.ejs`, `post.ejs`, `layout.ejs`, plus `_partial/` and `_widget/`) are EJS; styles are under `themes/miccall/source/css/`. Edit these directly to change look-and-feel — there is no upstream sync.

Special source directories beyond `_posts/`:

- `source/about/index.md`, `source/portfolio/index.md` — custom pages referenced from the theme's `Nav.pages` config. The portfolio page uses `layout: gallery` (front-matter) and pulls project entries from `source/_data/gallery.yml`.
- `source/_data/` — Hexo data files exposed to templates as `site.data.*`.

Post front-matter convention (see `README.md` and `scaffolds/post.md`): `title`, `date`, `tags`, `categories`, `thumbnail`. The `thumbnail` URL is used by the theme's index card view.
