# Xavier's Data Forge
用 Astro 6 + 客製化 [Astro Cactus](https://github.com/chrismwilliams/astro-theme-cactus) 主題，文章從 Obsidian vault 同步進 repo 後 commit & push 自動部署。

架構細節寫在 [`CLAUDE.md`](./CLAUDE.md)，這份 README 只講「日常怎麼用」。

## 發新文章

### 1. 在 Obsidian 寫

- 文章放在 vault 的 `Published/` 子資料夾才會發出去（其他資料夾不會被同步）
- 圖片用 Obsidian 預設的 attachment 設定，會自動放到 `Published/assets/<文章名>/`
- 檔名建議用 kebab-case（例：`openclaw-on-jetson.md`），URL 會更乾淨；用中文/空白也能跑，slug 會被自動轉小寫連字號

### 2. 在文章最上面補 frontmatter

最少：

```yaml
---
title: 文章標題                # 60 字元上限
description: 一句話描述         # 顯示在 meta tag 跟首頁卡片下方
publishDate: 23 May 2026
---
```

完整：

```yaml
---
title: 文章標題
description: 一句話描述
publishDate: 23 May 2026
updatedDate: 24 May 2026        # 之後改文時加上，內文會顯示「Updated:」
tags: ["tag1", "tag2"]
pinned: true                    # true 會出現在首頁置頂卡片（最多 3 篇）
draft: false                    # true 在 build 時隱藏
coverImage:                     # 文章頂端 Notion 風格封面 + 首頁卡片 thumbnail
  src: ./assets/文章名/cover.png
  alt: 圖片說明
---
```

### 3. 同步 + 本地預覽

```bash
npm run sync       # vault/Published/ → src/content/post/  (rsync --delete)
npm run dev        # http://localhost:4321/
```

> ⚠️ `sync` 是鏡像同步：vault 的 `Published/` 刪掉某篇，repo 端下次 sync 也會跟著刪。

### 4. 發佈

```bash
git add -A
git commit -m "post: <文章標題>"
git push
```

GitHub Actions 大約 1-2 分鐘跑完，自動部署到 https://xavierforge.dev/。

## Markdown 寫作小抄

### 圖片（已自動處理 Obsidian 語法）

```markdown
![alt|400](assets/文章名/foo.png)   <!-- alt 是 alt 文字，400 = 寬 400px -->
![](assets/文章名/foo.png)          <!-- 不指定尺寸 -->
![alt](https://example.com/x.png)   <!-- 外連直接通過 -->
```

### Callout（Obsidian 原生語法）

```markdown
> [!note] 標題
> 內容文字

> [!warning]
> 沒寫標題就用 type 當標題
```

支援的 type：`note` / `info` / `tip` / `warning` / `caution` / `important` / `success` / `question` / `bug` / `example` / `quote`。

### 程式碼 Highlight

Fence 語言名稱大小寫都認得（已 alias）：`bash` / `shell` / `python` / `py` / `js` / `ts` / `rust` / `yaml` / `json` / `md` / `dockerfile`。

要加新語言去 `src/site.config.ts` 的 `shiki.langAlias` 加。

## 其他常見任務

### 改 About 頁

編輯 `src/pages/about.astro`。頭像在 `src/assets/about.png`，換圖片只要覆蓋這個檔案。裁切位置改 class 裡的 `object-[center_25%]` 數字（0% = 對齊頂端，25% ≈ 露出臉）。

### 新增 Portfolio 條目

在 `src/content/project/` 開新資料夾，放 `index.md` + `cover.png`：

```yaml
---
title: 專案名
description: 一句話描述
link: https://github.com/...
order: 4                # 數字越小越靠前
coverImage:
  src: ./cover.png
  alt: 截圖說明
---
```

### 改社群連結 / icon

編輯 `src/data/social.ts`。首頁的 "Find me on" 跟 footer 都從這支讀。新 icon 用 [iconify](https://icon-sets.iconify.design/) 找名字，品牌類用 `simple-icons:*`，通用 UI 用 `mdi:*`。

### 改網站標題 / 描述 / 導航列

編輯 `src/site.config.ts`。

## 指令速查

```bash
npm install       # 安裝依賴（第一次或拉到別台機器）
npm run dev       # 開發伺服器
npm run sync      # 把 vault Published/ rsync 進 repo
npm run build     # 產生 dist/
npm run preview   # 預覽 build 結果（含 pagefind 搜尋）
npm run check     # astro check + biome check
```

## 部署設定（一次性，不用再動）

- GitHub repo → Settings → Pages → Source 已設成 **GitHub Actions**
- Workflow 在 `.github/workflows/pages.yml`，每次 push 到 `main` 觸發
- 自訂網域 `xavierforge.dev`（apex）：`public/CNAME` 每次 build 釘住網域，apex `A`/`AAAA` 指向 GitHub Pages，`www` 則 CNAME 到 `xavierforge.github.io`
