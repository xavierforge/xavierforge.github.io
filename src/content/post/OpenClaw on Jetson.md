---
title: 讓老舊開發板重獲新生：在 Jetson Nano (Ubuntu 18.04) 部署 OpenClaw
description: 在 Jetson Nano 部署 OpenClaw
publishDate: 22 Feb 2026
tags:
  - jetson nano
  - openclaw
pinned: true
draft: false
coverImage:
  src: ./assets/OpenClaw on Jetson/cover.png
  alt: Yeah~ AI 生成的縮圖
---
今年過年返鄉大掃除的時候，在百寶箱裡找到了閒置已久的 Jetson Nano Developer Kit，才赫然想起自己有這個好東西。  
但在 2026 年的今天，我的 Jetson Nano P3450 (B01 開發者套件) 的運算能力早已不具優勢。  
好前一陣子才讀到 [PicoClaw](https://github.com/sipeed/picoclaw/tree/main) 可以跑在嵌入式開發板上，我就想：Jetson Nano 即使已經被淘汰了，但運算能力畢竟還算不差，而且加上極低的功耗與完整的 GPIO 介面，它依然是作為本地端 Agent Gateway 的絕佳載體。  
而且坦白說，看了那麼多 AI 代理因為權限過大而引發的安全性災情，即使大概知道怎麼做防護，我還是不敢把平常有在工作的主力硬體馬上交給這類 AI Agent。  
所以不如就趁著年假，試試看能不能把這台老公公打造成一個安全且隔離的環境，當作我的小小龍蝦缸，在裡面把 [Clawdbot-Moltbot-OpenClaw](https://github.com/openclaw/openclaw) 養起來吧～
# 系統規格
| 項目      | 規格                                         |
| ------- | ------------------------------------------ |
| 開發板     | NVIDIA Jetson Nano (4GB)                   |
| GPU     | NVIDIA Maxwell with 128 NVIDIA CUDA® cores |
| CPU     | Quad-core ARM Cortex-A57                   |
| RAM     | 4GB 64-bit LPDDR4                          |
| Storage | 64GB MicroSD                               |
| OS      | Ubuntu 18.04.6 LTS (JetPack R32.7.6)       |
| Kernel  | 4.9.337-tegra                              |

首先要說的是，因為我已經完全失去上次使用這孩子的記憶了，打開後只在桌面上看到滿滿一堆 YOLOv4 相關的東西，也不知道自己當時是想建造什麼奇怪的跟蹤狂系統。
所以這次我選擇完全重刷，給自己一個乾淨的新環境，也把 JetPack 更新到停止支援前最後的穩定狀態。
這部分可以直接參考官方的 [Get Started With Jetson Nano Developer Kit](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#intro)，有相當完整的說明，這裡就不贅述了。
連線的部分，我插了一支 USB Wi-Fi 網卡，主要透過區域網路 (Local network) SSH 進去操作，但偶而也會接上 HDMI 看看瀏覽器是否有正確被開啟等實際運作情況。
# Jetson Nano 的先天限制與破局之法
前面提到，這個版本的 Jetson Nano 已經被澈底放生，官方支援停留在 JetPack 4.6.1，這意味著作業系統被永遠鎖死在 Ubuntu 18.04。
這個限制帶來了致命的連鎖反應：Ubuntu 18.04 內建的 C 標準函式庫 (`glibc`) 版本過舊。
而現代的 AI 工具 (包含 OpenClaw 所依賴的 Node.js 22+) 都需要較新的 `glibc` 才能編譯與執行。
如果直接照著官方文件安裝 Node.js，只會得到冷酷的 `Kernel too old` 錯誤。
既然硬體與底層 OS 無法改變，我們就必須從執行環境與系統安全性著手，一步步打造這座堅固的龍蝦缸。
## 第一步：清理水質與加固蝦缸 (系統限制與安全防護)
讓一個擁有 Shell 執行權限的 Agent 在舊系統上在裸奔實在太過刺激。
所以我們在開始養龍蝦之前，必須先建立好防線並榨出極限效能。
### 1. 啟用 ESM 獲取安全性更新
Ubuntu 18.04 雖然在 2023 年就停止常規支援，但我們可以透過免費的延伸安全性維護 (ESM) 續命到 2028 年。
請至 ubuntu.com/pro 獲取免費的 Token (有五台免費的額度)，並在終端機執行：
```Bash
sudo pro attach YOUR_FREE_TOKEN 
sudo apt update && sudo apt upgrade -y
```
這裡我實測刷完機啟用 ESM 後，直接補上了三百多個潛在的安全性漏洞。
### 2. 建立 Swap 與關閉無用服務 (榨取記憶體)
這台機器只有可憐的 4GB RAM。為了防止 OpenClaw 記憶體耗盡當機，必須強制劃分 4GB 的硬碟空間作為虛擬記憶體：
```Bash
sudo fallocate -l 4G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```
接著，把 `rpcbind` (Port 111) 這種單機環境毫無用處、又常被當成內網攻擊目標的老舊服務徹底封印：
```Bash
sudo systemctl stop rpcbind.socket rpcbind.service
sudo systemctl disable rpcbind.socket rpcbind.service
sudo systemctl mask rpcbind.socket rpcbind.service
```
### 3. 防火牆與 SSH 金鑰鎖定
為了確保這缸水只有我們能碰。
先把 SSH 改為純金鑰登入，接著啟動 UFW 防火牆，阻擋所有不請自來的連線 (請務必先放行 SSH，不然自己會被關在門外)：
```Bash
sudo apt install ufw -y
sudo ufw allow ssh
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw enable
```
## 第二步：解決老系統的消化不良，吃我特製包子 (Bun) 啦！
延續龍蝦缸的比喻，OpenClaw 這隻現代化的 AI 龍蝦，原本的主食是 Node.js 22+。
但 Ubuntu 18.04 這缸老水裡面的基礎生態 (C 標準函式庫 `glibc`) 實在太過古老，如果硬塞新版的 Node.js 給它，系統就會嚴重消化不良，直接吐出無情的 `Kernel too old` 報錯。
面對 `glibc` 報錯，前陣子被 Anthropic 收購的 Bun 是我們的完美救星。
作為一個現代化的 JavaScript Runtime，它採用靜態編譯，直接無視了 Ubuntu 18.04 底層老舊的依賴限制。
加上極快的啟動速度和較低的記憶體開銷，完全就是為了這種 [經典老爺車](https://x.com/heroaca_anime/status/2016164303900954705/photo/1) 量身打造的：
```Bash
sudo apt install -y curl unzip
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```
但要注意，雖然 Bun 讓我們可以安裝 OpenClaw，但官方並不建議在「生產環境」的 Gateway 使用 Bun 作為 Runtime，因為它在銜接 WhatsApp 或 Telegram 等對外通訊軟體時，可能會遇到一些水土不服的 Bug。
但面對我們這缸 2019 年的老水 (Ubuntu 18.04)，這已經是最輕鬆的方法了。
既然是趁著年假搞的「廢物利用」極限沙盒實驗，這點潛在的腸胃副作用，也是我們為了讓這隻 AI 龍蝦活下去，所必須接受的妥協啦！
BTW，筆者本人在除夕的時候真的因為急性腸胃炎去急診室吊了一桶 🤦，所以這裡才硬塞了一個消化不良的比喻。
## 第三步：放蝦囉與 Shebang 修正
有了 Bun 這款特製包子，我們終於可以安心地把 OpenClaw 抓進缸裡：
```Bash
bun install -g openclaw@latest
```
但這裡有個小小的坑： OpenClaw 執行檔開頭的 shebang 預設是 `#!/usr/bin/env node`。
如果不改，系統還是會傻傻地去找那個不存在的 Node.js，所以這安裝好要記得修改一下，強制把它轉向 Bun：
```Bash
sed -i '1s|#!/usr/bin/env node|#!/usr/bin/env bun|' ~/.bun/install/global/node_modules/openclaw/openclaw.mjs
```
## 第四步：讓牠乖乖待在缸底 (Daemon 背景服務的坑)
至此，OpenClaw 總算是順利入缸了！
但剛才修改 shebang 的動作，僅僅是讓 `openclaw` 這個指令本身可以順利運作，不再因為找不到 Node 而跳出報錯。
也就是說，我們現在可以「手動」逗弄這隻龍蝦了。
但如果想讓牠在我們關閉終端機後，還能安分地待在缸底常駐存活 (建立 systemd 背景服務)，就會立刻踩到下一個地雷。
當我們滿懷希望地敲下官方的背景服務安裝指令：
```
openclaw onboard --install-daemon
```
會發現安裝腳本非常固執，依然死死盯著系統環境，強制尋找 Node 22+ 的蹤影。
理所當然地，腳本會因為找不到符合標準的 Node 而直接崩潰罷工。
所以這裡為了避免腳本去系統環境裡亂找，我們得利用 `bunx`，強制把安裝腳本關在 Bun 的虛擬環境中執行，這樣就能成功騙過系統的檢查機制，順利產出服務設定檔：
```Bash
bunx openclaw onboard --install-daemon
```
跟著提示一路設定，應該就能成功建立好服務，進入到可以設定龍蝦人格的 TUI 或 WebUI 了。
但！先別急著慶祝！
打開 `~/.config/systemd/user/openclaw-gateway.service` 看一下，應該會發現 `ExecStart` 居然被寫死成類似 `/tmp/bun-node-xxx/node` 的暫時路徑。 
這可是個致命傷，只要 Nano 一重開機，`/tmp` 被清空，龍蝦就直接暴斃了。
所以這裡必須手動把它換成真實的 Bun 路徑 (請把 `YOUR_USER_NAME` 換成 Ubuntu 的登入帳號)：
```Bash
sed -i 's|/tmp/bun-node-[a-zA-Z0-9]*/node|/home/YOUR_USER_NAME/.bun/bin/bun|g' ~/.config/systemd/user/openclaw-gateway.service
systemctl --user daemon-reload
systemctl --user restart openclaw-gateway.service
```
最後，打上這行指令啟用 Linger (使用者駐留)，確保服務常駐背景：
```Bash
sudo loginctl enable-linger YOUR_USER_NAME
```
至此，這座隔離且安全的龍蝦缸已經打造完畢，我們成功繞過了 2019 年硬體的 OS 限制，把這台退役的 Jetson Nano 變成了一台安靜、省電的 AI 秘書主機！
未來不論是想寫一些 Rust 擴充技能給 Agent 呼叫，還是讓它在背景跑量化交易的監控腳本，都可以放心地在這個沙盒裡盡情折騰，再也不用擔心它會把工作用的主力機搞到翻車了。
這裡就讓我的小龍蝦跟大家拜個晚年吧：
![祝大家馬上舞吉、馬西搜呦|500](assets/OpenClaw%20on%20Jetson/file-20260523230132215.gif)
# 結語
回過頭來看，我們這幾步其實都在想辦法規避老舊系統帶來的麻煩。
Jetson Nano 的宿命被官方釘死在 JetPack 4.6.x，這不單單是作業系統停留在 Ubuntu 18.04 的問題，而是它的 Kernel、GPU 驅動和 CUDA 生態全都和底層的 L4T (Linux for Tegra) 深度綁定。
我們不能隨便下指令升級系統，縱使使用第三方魔改的 OS 又可能會失去最寶貴的 GPU 加速支援，這樣就沒必要堅持使用 Jetson 了。
除了作業系統以外，這個硬體的生態系也已經非常脆弱，例如：
- **Node 22 絕對裝不起來**：因為它需要 `glibc 2.28+`，但 18.04 只能給到 `2.27`。
  自己編譯原始碼要花上好幾個小時，用 Docker 又會遇到舊版 Runtime 的限制。
- **Python 3.6 已經是化石**：想裝現代的 AI 套件，很多 wheels 根本沒有預先編譯好的 aarch64 版本。
因此要在 2026 年讓它還能搾出一點價值，唯一生存法則就是：**永遠不要依賴系統底層的函式庫。**
這也是為什麼我們這次選擇了自帶執行環境的 Bun。
未來如果要幫這隻 AI 龍蝦擴充能力，也是得遵循一模一樣的邏輯，尋找那些能把依賴打包帶走的工具。
剛好我自己平常就在寫 Rust，它的特性完美契合這個生存法則：交叉編譯出獨立、靜態且效能極佳的二進位檔 (Binaries)，完全不需要看老系統的臉色。
接下來，我打算直接用 Rust 寫一些本地端的執行檔丟進這個沙盒裡，當作 Agent 的自訂技能 (Custom Skills)，雖然目前只先測試了交叉編譯與調用 Rust 預設的 `Hello, world!`，但既然已經摸透了這套不依賴底層環境的生存法則，想必未來在擴充這隻龍蝦的技能樹時，應該不會再踩到那麼多莫名其妙的坑了 (大概吧)。
大家再見！