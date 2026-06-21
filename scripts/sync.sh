#!/usr/bin/env bash
#
# Sync the Obsidian vault's `published/` subfolder into this repo's
# `src/content/post/`. Run via `npm run sync` before `npm run dev` or
# committing.
#
# Adjust VAULT_PUBLISHED below to point at your vault's published folder.
# The path can contain ~ and spaces.

set -euo pipefail

VAULT_PUBLISHED="${VAULT_PUBLISHED:-$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/my_obsidian/Published}"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/content/post"

if [[ ! -d "$VAULT_PUBLISHED" ]]; then
	echo "error: vault published folder not found at: $VAULT_PUBLISHED" >&2
	echo "       set VAULT_PUBLISHED env var or edit scripts/sync.sh" >&2
	exit 1
fi

mkdir -p "$DEST"

# Pre-sync lint: warn (but don't block) about invisible / odd whitespace in the
# source markdown before it propagates into the repo — hair spaces, no-break
# spaces, zero-width chars that sneak in via copy-paste. Warn-only by design so
# it never aborts a sync; run `npm run check:whitespace` on demand, or add
# --strict there for a hard gate.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
node "$SCRIPT_DIR/check-whitespace.mjs" "$VAULT_PUBLISHED" || true

# --delete keeps the repo a mirror of vault/published — files removed from
# the vault disappear from the repo on the next sync. Add --dry-run while
# testing if that scares you.
#
# `*.en.md` are repo-only English translations (authored in the repo, not the
# vault). Exclude them so --delete never wipes them — see
# scripts/check-translations.mjs and the bilingual-posts notes in CLAUDE.md.
rsync -av --delete \
	--exclude '.obsidian/' \
	--exclude '.trash/' \
	--exclude '.DS_Store' \
	--exclude '*.en.md' \
	"$VAULT_PUBLISHED/" "$DEST/"

# Second pass: pull vault-authored `*.en.md` through WITHOUT --delete.
#
# `.en.md` come in two flavours: AI translations that live only in the repo
# (never the vault), and human-written English whose Chinese companion you also
# author yourself — that English IS written in the vault. The --delete mirror
# above excludes `*.en.md` so the repo-only AI translations are never wiped;
# this no-delete pass copies any `.en.md` that DOES exist in the vault into the
# repo. Repo-only `.en.md` (absent from the vault) are left untouched because
# this pass carries no --delete. `-m` prunes empty dirs the filter would create.
rsync -avm \
	--include '*/' \
	--include '*.en.md' \
	--exclude '*' \
	"$VAULT_PUBLISHED/" "$DEST/"

echo
echo "synced: $VAULT_PUBLISHED → $DEST"
