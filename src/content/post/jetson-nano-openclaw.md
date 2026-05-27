---
title: "讓老舊開發板重獲新生：在 Jetson Nano 部署 OpenClaw AI Gateway"
description: "把退役的 Jetson Nano (Ubuntu 18.04) 打造成安全隔離的本地 AI Agent 沙盒：啟用 ESM 續命、用 Bun 繞過 glibc 限制，一步步部署 OpenClaw AI Gateway 並設定 systemd 常駐服務。"
publishDate: 2026-02-22
updatedDate: 2026-03-06
coverImage:
  src: ./assets/jetson-nano-openclaw/cover.png
  alt: "AI 生成的 Jetson Nano 部署 OpenClaw 縮圖"
tags: [Jetson Nano, OpenClaw, AI Agent, Bun, Ubuntu]
pinned: true
draft: false
---
今年過年返鄉大掃除的時候，在百寶箱裡找到了閒置已久的 Jetson Nano Developer Kit，才赫然想起自己有這個好東西。  

但在 2026 年的今天，我的 Jetson Nano P3450 (B01 開發者套件) 的運算能力早已不具優勢。  

剛好前一陣子才讀到 [PicoClaw](https://github.com/sipeed/picoclaw/tree/main) 可以跑在嵌入式開發板上，我就想：Jetson Nano 即使已經被淘汰了，但運算能力畢竟還算不差，而且加上極低的功耗與完整的 GPIO 介面，它依然是作為本地端 Agent Gateway 的絕佳載體。

而且坦白說，看了那麼多 Agent 因為權限過大而引發的安全性災情，即使大概知道怎麼做防護，我還是不敢把平常有在工作的主力硬體馬上交給這類 AI Agent。  

所以乾脆就趁著年假，試試看能不能把這台老公公打造成一個安全且隔離的環境，當作我的小小龍蝦缸，在裡面把 [Clawdbot-Moltbot-OpenClaw](https://github.com/openclaw/openclaw) 養起來吧～
## 系統規格
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
## Jetson Nano 的先天限制與破局之法
前面提到，這個版本的 Jetson Nano 已經被澈底放生，官方支援停留在 JetPack 4.6.1，這意味著作業系統被永遠鎖死在 Ubuntu 18.04。
這個限制帶來了致命的連鎖反應：Ubuntu 18.04 內建的 C 標準函式庫 (`glibc`) 版本過舊。

而現代的 AI 工具 (包含 OpenClaw 所依賴的 Node.js 22+) 都需要較新的 `glibc` 才能編譯與執行。

如果直接照著官方文件安裝 Node.js，只會得到冷酷的 `Kernel too old` 錯誤。
既然硬體與底層 OS 無法改變，我們就必須從執行環境與系統安全性著手，一步步打造這座堅固的龍蝦缸。
### 第一步：清理水質與加固蝦缸 (系統限制與安全防護)
讓一個擁有 Shell 執行權限的 Agent 在舊系統上在裸奔實在太過刺激。

所以在開始養龍蝦之前，必須先建立好防線並榨出極限效能。
#### 1. 啟用 ESM 獲取安全性更新
Ubuntu 18.04 雖然在 2023 年就停止常規支援，但我們可以透過免費的延伸安全性維護 (ESM) 續命到 2028 年。

請至 ubuntu.com/pro 獲取免費的 Token (有五台免費的額度)，並在終端機執行：
```Bash
sudo pro attach YOUR_FREE_TOKEN 
sudo apt update && sudo apt upgrade -y
```
這裡筆者實測刷完機啟用 ESM 後，直接補上了三百多個潛在的安全性漏洞。
#### 2. 建立 Swap 與關閉無用服務 (榨取記憶體)
這台機器只有可憐的 4GB RAM，為了防止 OpenClaw 記憶體耗盡當機，必須強制劃分 4GB 的硬碟空間作為虛擬記憶體：
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
#### 3. 防火牆與 SSH 金鑰鎖定
為了確保這缸水只有我們能碰。
##### 第一階段：在操作主機 (例如筆電) 上產生並傳送金鑰
打開平時操作用電腦的終端機，產生一把安全性較高的 `ed25519` 金鑰 (過程詢問設定時，一路按 Enter 預設到底即可)：
```bash
ssh-keygen -t ed25519 -C "nano-key"
```
接著，將這把產生的公鑰傳送到 Jetson Nano 裡 (請將 IP 替換成實際位址，假設在同一個 WiFi 的 Local network 的話，可以透過 `ifconfig` 來查看，應該是 `wlan0` 的 `inet`後面那個 IP)：
```bash
ssh-copy-id YOUR_USER@<Nano的IP位址>
```
系統會要求我們最後一次輸入 Nano 的登入密碼，輸入完成後，公鑰就會自動寫入 Nano 的授權名單中。
##### 第二階段：關閉密碼驗證與啟動防火牆 (在 Nano 端操作)
請立刻在操作主機上測試 `ssh YOUR_USER@<Nano的IP位址>`，確認現在不需要輸入密碼就能直接登入。確認無誤後，我們就可以放心地把密碼登入的後門鎖上了。
在 Nano 的終端機輸入：
```bash
# 編輯 SSH 設定檔  
sudo vim /etc/ssh/sshd_config
```
尋找 `PasswordAuthentication yes` 這一行，將它改為 `no` (如果該行最前方有 `#` 註解符號，請一併刪除)。

存檔離開 (`:wq`) 後，重啟 SSH 服務讓設定生效：
```bash
sudo systemctl restart ssh
```
最後，啟動 UFW 防火牆，預設阻擋所有不請自來的連線 (請務必先執行放行 `ssh` 的指令，不然防火牆一開，連我們自己也會被關在門外)：
```Bash
sudo apt install ufw -y
sudo ufw allow ssh
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw enable
```
### 第二步：解決老系統的消化不良，吃我特製包子 (Bun) 啦！
延續龍蝦缸的比喻，OpenClaw 這隻現代化的 AI 龍蝦，原本的主食是 Node.js 22+。

但 Ubuntu 18.04 這缸老水裡面的基礎生態 (C 標準函式庫 `glibc`) 實在太過古老，如果硬塞新版的 Node.js 給它，系統就會嚴重消化不良，直接吐出無情的 `Kernel too old` 報錯。

面對 `glibc` 報錯，前陣子被 Anthropic 收購的 Bun 是我們的完美救星。
作為一個現代化的 JavaScript Runtime，它採用靜態編譯，直接無視了 Ubuntu 18.04 底層老舊的依賴限制。

加上極快的啟動速度和較低的記憶體開銷，完全就是為了這種 [經典老爺車](https://x.com/heroaca_anime/status/2016164303900954705/photo/1) 量身打造的，所以這裡就直接把它安裝起來：
```Bash
sudo apt install -y curl unzip
curl -fsSL https://bun.sh/install | bash
source ~/.bashrc
```
但要注意，雖然 Bun 讓我們可以安裝 OpenClaw，但 [官方並不建議在「生產環境」的 Gateway 使用 Bun 作為 Runtime](https://docs.openclaw.ai/install/bun)，因為它在銜接 WhatsApp 或 Telegram 等對外通訊軟體時，可能會遇到一些水土不服的 Bug。

但面對我們這缸 2019 年的老水 (Ubuntu 18.04)，這已經是最輕鬆的方法了。

既然是趁著年假搞的「廢物利用」極限沙盒實驗，這點潛在的腸胃副作用，也是我們為了讓這隻 AI 龍蝦活下去，所必須接受的妥協啦！
BTW，筆者本人在除夕的時候真的因為急性腸胃炎去急診室吊了一桶 🤦，所以這裡才硬塞了一個消化不良的比喻。
### 第三步：放蝦囉與 Shebang 修正
有了 Bun 這款特製包子，我們終於可以安心地把 OpenClaw 抓進缸裡：
```Bash
bun install -g openclaw@latest
```

> [!NOTE] 2026/03/06 更新
> 在純 Bun 環境下執行 OpenClaw 時，常會遇到 `Executable not found in $PATH: "node"` 的報錯。  
> 一開始我會像下面的文章所提的一樣，手動去改腳本的 shebang，但這招在每次 OpenClaw 更新後都會被覆蓋重置，加上許多底層指令預設還是會去尋找 `node`，反覆修改實在太麻煩。  
> 既然我們的 Jetson Nano 本來就沒有安裝、也不需要原生的 Node.js，最一勞永逸的解法就是直接建立軟連結，把 `node` 指令導向 `bun`，讓高度相容的 Bun 上場代打。  
> 至於為什麼可以直接把 `node` 指令導向 `bun`？這是因為 Bun 的設計初衷就是作為 Node.js 的無縫替代品，底層完整實作了 Node.js 的核心 API 與模組解析機制。所以當系統順著捷徑把任務交接過去時，Bun 完全能直接看懂並正確執行那些腳本，程式根本不會察覺背後其實是 Bun 在代打。  
> 只要在終端機執行 `ln -s $(which bun) ~/.bun/bin/node` 建立捷徑，接著輸入 `node -v` 確認有跳出 Bun 的版本號就搞定了，以後無論怎麼更新都不用再煩惱！

但這裡有個小小的坑： OpenClaw 執行檔開頭的 shebang 預設是 `#!/usr/bin/env node`。

如果不改，系統還是會傻傻地去找那個不存在的 Node.js，所以這安裝好要記得修改一下，強制把它轉向 Bun：
```Bash
sed -i '1s|#!/usr/bin/env node|#!/usr/bin/env bun|' ~/.bun/install/global/node_modules/openclaw/openclaw.mjs
```
### 第四步：讓牠乖乖待在缸底 (Daemon 背景服務的坑)
至此，OpenClaw 總算是順利入缸了！

但剛才修改 shebang 的動作，僅僅是讓 `openclaw` 這個指令本身可以順利運作，不再因為找不到 Node 而跳出報錯。

也就是說，我們現在可以「手動」逗弄這隻龍蝦了。
但如果想讓牠在我們關閉終端機後，還能安分地待在缸底常駐存活 (建立 systemd 背景服務)，就會立刻踩到下一個地雷。

當我們滿懷希望地敲下官方的背景服務安裝指令：
```
openclaw onboard --install-daemon
```
會發現安裝腳本非常固執，依然死死盯著系統環境，強制尋找 Node 22+ 的蹤影，理所當然地，腳本會因為找不到符合標準的 Node 而直接崩潰罷工。

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
好啦，這座隔離且安全的龍蝦缸已經打造完畢，我們成功繞過了 2019 年硬體的 OS 限制，把這台退役的 Jetson Nano 變成了一台安靜、省電的 AI 秘書主機！

未來不論是想寫一些 Rust 擴充技能給 Agent 呼叫，還是讓它在背景跑量化交易的監控腳本，都可以放心地在這個沙盒裡盡情折騰，再也不用擔心它會把工作用的主力機搞到翻車了。

對啦，要記得時常執行以下兩個指令，確保我們的使用都是安全的哦：
```bash
openclaw security audit --deep  
openclaw security audit --fix
```
這裡就讓我的小龍蝦跟大家拜個晚年吧：
![祝大家馬上舞吉、馬西搜呦](assets/jetson-nano-openclaw/file-20260527152859738.gif)
## 結語
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

例如筆者較為熟悉的 Rust，它的特性完美契合這個生存法則：交叉編譯出獨立、靜態且效能極佳的二進位檔 (Binaries)，完全不需要看老系統的臉色。

接下來，我打算直接用 Rust 寫一些本地端的執行檔丟進這個沙盒裡，當作 Agent 的自訂技能 (Custom Skills)，雖然目前只先測試了交叉編譯與調用 Rust 預設的 `Hello, world!`，但既然已經摸透了這套不依賴底層環境的生存法則，想必未來在擴充這隻龍蝦的技能樹時，應該不會再踩到那麼多莫名其妙的坑了 (大概吧)。

另外如果 OpenClaw 真的太肥的話，可能還會試試看用 Rust 重寫的超輕量版本 [ZeroClaw](https://github.com/openagen/zeroclaw)。
  
今天的分享到這裡結束了，大家再見，新年快樂！