---
title: 2023 鐵人賽完賽｜Rust 加 MLOps，你說有沒有搞頭？
description: 2023 iThome 鐵人賽完賽心得與 30 天文章總索引：從一個 Python 使用者的視角，探索 Rust 在 MLOps 中有沒有搞頭，本系列後來延伸成書《從 Pythonista 到 Rustacean》。
publishDate: 2023-10-15
coverImage:
  src: ./assets/Ironman 2023/cover-ferris-finish.png
  alt: 完賽衝線圖，中間 C 位是 Ferris
tags:
  - 鐵人賽
  - Rust
  - MLOps
  - Python
draft: false
---
> 這裡整理我參加 2023 iThome 鐵人賽的 30 天完賽總目錄與心得，各篇原文發表於 iThome，點擊可跳轉至原文。
## 這個系列後來變成一本書了！
這次的鐵人賽系列，後來經過大量增修、擴寫成了我的第一本書 **《從 Pythonista 到 Rustacean：資料從業者的第一本 Rust 指南》**（深智數位出版）。

如果各位讀者想要更完整、更新、更系統化的版本，歡迎支持：
[![從 Pythonista 到 Rustacean 書封](assets/Ironman%202023/file-20260525141742071.jpg)](https://www.books.com.tw/products/0011044858)
- 📕 [博客來](https://www.books.com.tw/products/0011044858)
- 📗 [天瓏網路書店](https://www.tenlong.com.tw/products/9786267757789)
> 書中相較於本系列，補上了 PyO3 擴充套件實戰、Candle 與 Burn 框架的 AI 落地應用，以及更完整的工程化實踐。

---
## 今日份 Ferris
今天最後一天啦，恭喜自己完賽，這 30 天就跟用 Stable diffusion 畫圖一樣，有時候很崩潰，有時候很順遂 (單押)，但無論如何，我們都在嘗試中成長了，附上大量衝線圖，中間 C 位是 [Ferris boy](https://www.youtube.com/watch?v=dQw4w9WgXcQ)（如上方封面）。
## 完賽心得
存活下來啦～ 不得不說，裸跑真的是滿精實的，先寫在這裡提醒自己下次記得囤點文章，不然...
![裸跑心得|300](assets/Ironman%202023/file-20260525142157510.png)
＊提醒未來的自己這是時事梗，[「你就像茶碗蒸」 壽司店對到眼 引爆衝突](https://www.youtube.com/watch?v=xr8PzQ7qntk&t=5s)

這次挑戰依然拖到了跨越兩個連假 (雖然我是第五個報名的哈哈哈)，還遇到了貓貓需要看醫生 (目前已經是健康寶寶了，感謝天，感謝地，[感謝命運讓我們相遇](https://www.youtube.com/watch?v=Mh5r3aD2iwg&t=23s))，真的很難把文章完善到自己完全滿意的程度，尤其是專案的部分，實在沒辦法花太多時間去 Debug 與優化，只能先想辦法讓現有的程式運作起來就好，這大概是這次挑戰最遺憾的地方：
![專案遺憾|300](assets/Ironman%202023/file-20260525142341219.png)

我想很多挑戰者都能有感觸的一點就是，文章沒有完成的盡頭，但鐵人賽 12 點前要交 (那個最下面的系統標準時間 UTC+0800 會變成紅色的，嚇死人)，無論如何，作品在讀者看到的時候就完成了，所以很感謝願意收看的孩子們，尤其感謝放心點開位置很奇怪的超連結然後不意外地發現奇怪的東西也可以會心一笑的孩子，而且沒想到這次比上次多了五個人訂閱，真是萬分感謝讓我為您右上角的小鈴鐺貢獻！

最後，這次挑戰的目標在於從一個 Python 使用者的角度來探索 Rust 加 MLOps 有沒有搞頭，以下是我個人的結論 (歡迎在下方留言區討論！)：

> 我認為 Rust 絕對是未來的一部分，它為我們帶來了更好的表現與更簡潔的解決方案。 但這對社群來說是加分，而非取代，就像 Scala 與 R 都還在、小象 Hadoop 會陪伴我們到老一樣。 在可預見的未來 Python 依然會穩居 ML 之王的寶座，但 Rust 絕對是工具包裡一把強而有力的瑞士刀。 總的來說，Rust 加 MLOps 的確有搞頭，它在我們需要效能時，是非常完美的！
> ![結論配圖|300](assets/Ironman%202023/file-20260525142523632.jpg)
## 文章索引
在過去的 30 天中，我們首先將 Rust 引入日常的工作流程、完成 LLM chatbot 專案，到談論從資料、模型、產品的角度可以如何應用 Rust，真是一段驚險刺激的旅程！
![旅程回顧|400](assets/Ironman%202023/file-20260525142701570.jpg)
以下是按照順序各個主題的發文整理，請大家笑納：
### 1. From Python to Rust
> 以所有軟體專案都會經歷的工作流程來說明如何優雅地從 Python 轉換到 Rust 基本上就是說服大家為什麼要學 Rust 🤣
- [[Day 02] 從 Python 🐍 到 Rust 🦀｜超級概覽，那 Mojo🔥呢？](https://ithelp.ithome.com.tw/articles/10319293)
- [[Day 03] 從 Python 🐍 到 Rust 🦀｜上工啦！安裝、環境與函式庫相依管理](https://ithelp.ithome.com.tw/articles/10321315)
- [[Day 04] 走到哪用到哪 🏃，Rust MLOps GitHub 模板與在容器中開發](https://ithelp.ithome.com.tw/articles/10322608)
- [[Day 05] Rust 原廠要你命 3000 🧨，Makefile 跑起來啦寶貝](https://ithelp.ithome.com.tw/articles/10323209)
- [[Day 06] 無敵風火輪 🌪️ GitHub Action 讓 Rust CI 轉起來](https://ithelp.ithome.com.tw/articles/10324013)
- [[Day 07] Rust x 單元測試 x MLOps (上)](https://ithelp.ithome.com.tw/articles/10324999)
- [[Day 08] Rust x 單元測試 x MLOps (下)](https://ithelp.ithome.com.tw/articles/10325651)
- [[Day 09] 從 Python 🐍 到 Rust 🦀｜MLOps 最終比較 ⚔️ 與環境永續 🍀](https://ithelp.ithome.com.tw/articles/10326453)
### 2. Project!
> 完成一個利用 Huggingface LLM 模型的 chatbot 專案
> ![Demo|700](assets/Ironman%202023/file-20260525142831951.gif)
- [[Day 10] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (1/10)｜專案簡介](https://ithelp.ithome.com.tw/articles/10327495)
- [[Day 11] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (2/10)｜行前準備](https://ithelp.ithome.com.tw/articles/10328736)
- [[Day 12] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (3/10)｜Leptos 小教室](https://ithelp.ithome.com.tw/articles/10329279)
- [[Day 13] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (4/10)｜對話の資料結構](https://ithelp.ithome.com.tw/articles/10329850)
- [[Day 14] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (5/10)｜Signal & Action](https://ithelp.ithome.com.tw/articles/10330642)
- [[Day 15] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (6/10)｜GGML 量化 LLaMa](https://ithelp.ithome.com.tw/articles/10331431)
- [[Day 16] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (7/10)｜後端 LLM API](https://ithelp.ithome.com.tw/articles/10332054)
- [[Day 17] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (8/10)｜Rust 中載入 GGML 模型](https://ithelp.ithome.com.tw/articles/10332674)
- [[Day 18] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (9/10)｜前端美化與最終成果](https://ithelp.ithome.com.tw/articles/10332816)
- [[Day 19] 鋼鐵草泥馬 🦙 LLM chatbot 🤖 (10/10)｜結論及展望](https://ithelp.ithome.com.tw/articles/10334027)
### 3. Keep on Rusting
> 喘口氣，聊聊 Rust 的學習資源與如何與 Rust 最勸退人的 Borrow checker 相處
- [[Day 20] 中場休息 🏖️ 如何度過 Rust 新手村? 我要打十個 Borrow checker！](https://ithelp.ithome.com.tw/articles/10334331)
### 4. Rust + MLOps
> 回到 MLOps 的主題，以 ML 系統設計的角度談談 Rust 如何在這裡發揮。這部分參考 [Chip Huyen](https://huyenchip.com/) 的著作 [Designing Machine Learning Systems](https://learning.oreilly.com/library/view/designing-machine-learning/9781098107956/)，以兩至三天為一組，從資料、模型、產品三個面向看待 ML 系統，以及 Rust 可以如何被應用在 MLOps 中。
- [[Day 21] 機器學習系統設計 🏭 x Rust 🦀](https://ithelp.ithome.com.tw/articles/10335213)
- [[Day 22] 資料處理和特徵工程 🔢 (上) | ML 系統設計 🏭](https://ithelp.ithome.com.tw/articles/10335893)
- [[Day 23] 資料處理和特徵工程 🔢 (中) | ML 系統設計 🏭](https://ithelp.ithome.com.tw/articles/10336339)
- [[Day 24] 資料處理和特徵工程 🔢 (下) | Rust x Jupyter 資料工程 🦀](https://ithelp.ithome.com.tw/articles/10336919)
- [[Day 25] 模型開發 🧠 (上) | ML 系統設計 🏭](https://ithelp.ithome.com.tw/articles/10337842)
- [[Day 26] 模型開發 🧠 (下) | Rust x PyTorch 模型訓練與輸出 🦀](https://ithelp.ithome.com.tw/articles/10337882)
- [[Day 27] 預測服務 🚀 (上) | ML 系統設計 🏭](https://ithelp.ithome.com.tw/articles/10338524)
- [[Day 28] 預測服務 🚀 (下) | Rust x Docker 部署鋼鐵草泥馬 🦙🦀](https://ithelp.ithome.com.tw/articles/10338993)
- [[Day 29] 期末專欄 🎞️ | Rust 是資料分析的未來嗎？](https://ithelp.ithome.com.tw/articles/10339414)

好啦，下次見囉～