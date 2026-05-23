---
title: "Hello World"
description: "First post on the new Astro-powered blog."
publishDate: "23 May 2026"
pinned: true
tags: ["meta"]
---

This blog has moved off Hexo onto Astro (Cactus theme). Posts now flow
straight from my Obsidian vault into `src/content/post/` via a `npm run sync`
script — no more reformatting between editor and publisher.

## What changed

- Static site generator: Hexo → Astro
- Theme: `miccall` (unmaintained) → Cactus (forked, customised)
- Authoring: ad-hoc → Obsidian vault's `published/` subfolder
- Deploy: `gh-pages` branch via `peaceiris/actions-gh-pages` → native
  `actions/deploy-pages`

## Why

Medium and the old Hexo flow forced reformatting on every publish. With
Obsidian as the source of truth and a thin sync step, the friction is gone.
