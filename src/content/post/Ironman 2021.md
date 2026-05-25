---
title: 2021 鐵人賽佳作｜然後模型就死在 Jupyter Notebook 了 (ಥ﹏ಥ)
description: 2021 iThome 鐵人賽佳作作品。從機器學習產品的生命週期出發，盤點一個 ML 產品從概念到落地會經過的部署、建模、資料、計畫各階段，並實作一個簡單的 Web App。
publishDate: 2021-10-15
coverImage:
  src: ./assets/Ironman 2021/cover.png
  alt: 佳作版封面
tags:
  - 鐵人賽
  - 機器學習
  - MLOps
  - Python
draft: false
---

> 這是我參加 2021 iThome 鐵人賽的 30 天完賽總目錄與心得，各篇原文發表於 iThome，點擊可跳轉至原文。
## 🏆 本系列榮獲 2021 iThome 鐵人賽・佳作

---
## 完賽心得
轉眼間就過了 30 天啦，第一次參賽有夠菜沒想到還能迎來這一天。
要堅持每天發文真的很考驗意志力 (也大大考驗了我的 PPT 作圖技術哈哈哈)，期間還經歷了兩次連假，差點就放飛自我了。
關於這次挑戰，一開始的動機是因為在機器學習這個領域摸爬滾打也大概兩年的時間，從入門到稍有概念剛好搭上了大家都在談論 AI 落地的風潮，所以想趁此機會盤點一下一個機器學習產品從概念到實際產出會經過哪些步驟以及需要考慮什麼事，最後再利用這些概念實作一個簡單的 Web App：
![Web App demo|600](assets/Ironman%202021/file-20260525145337256.gif)
可以看得出來這個 App 離完美還有好大一段距離，但至少可以做個敲門磚，朝著規模較大的應用前進。

這次挑戰的文章基本上算是 Coursera 專項課程 [Machine Learning Engineering for Production (MLOps) Specialization](https://www.coursera.org/specializations/machine-learning-engineering-for-production-mlops) 第一部分 Introduction to Machine Learning in Production 的心得筆記。

為了保留整體性以及讓每天的文章簡單易讀一點，這裡沒有把課程後面更深入的實作 (基於 [TensorFlow Extended (TFX)](https://www.tensorflow.org/tfx?hl=zh-tw)) 放進來，但真的很推薦大家去聽聽看這門課程，裡面有談到真正商業場域的大規模應用該怎麼做。

另外同樣推薦的還有 [Full Stack Deep Learning](https://fullstackdeeplearning.com/) 的相關內容，這次挑戰結束後我也會把這個拖了很久的坑補上，到時有機會再與大家分享！

如果是比較忙碌的朋友，我大力推薦威利斯大大今年的系列文 [從 AI 落地談 MLOps](https://ithelp.ithome.com.tw/users/20121130/ironman/4015)，都幫你整理好了還手把手教你怎麼做，再不看真的對不起自己，趕快手刀分頁開起來 [Day 30：綜合整理 MLOps Level 0 ~ 2](https://ithelp.ithome.com.tw/articles/10274317)。

最後，感謝訂閱我的三個孩子，真的沒想到有人願意賞臉訂閱哈哈哈
## 文章索引
本系列的文章都圍繞下面這張「機器學習產品生命週期」，這裡就按照發文順序整理起來：
![機器學習產品生命週期示意圖|600](assets/Ironman%202021/file-20260525145456995.png)
＊圖片修改自 [Introduction to Machine Learning in Production](https://www.coursera.org/learn/introduction-to-machine-learning-in-production/home/welcome)
### 1. 概覽
- [[Day 01] 前言 — 是誰殺了模型?](https://ithelp.ithome.com.tw/articles/10265358)
- [[Day 02] Why MLOps — 從「地平說」走向宇宙](https://ithelp.ithome.com.tw/articles/10265468)
- [[Day 03] 機器學習產品生命週期 — 救救我啊我救我](https://ithelp.ithome.com.tw/articles/10265896)
### 2. 部署
- [[Day 04] 部署模型的挑戰 — 資料也懂超級變變變!?](https://ithelp.ithome.com.tw/articles/10267729)
- [[Day 05] 部署模式 — 我的模型叫崔弟](https://ithelp.ithome.com.tw/articles/10268119)
- [[Day 06] 監控、維護 — 自己開一家徵信社吧!](https://ithelp.ithome.com.tw/articles/10268867)
- [[Day 07] 使用 fastAPI 部署 YOLOv4 (1/2) — 以內建 Client 進行互動](https://ithelp.ithome.com.tw/articles/10269911)
- [[Day 08] 使用 fastAPI 部署 YOLOv4 (2/2) — 自行撰寫 Client 進行互動](https://ithelp.ithome.com.tw/articles/10270393)
### 3. 建立模型
- [[Day 09] 建立機器學習模型 — Andrew Ng 大神說要這樣做](https://ithelp.ithome.com.tw/articles/10270985)
- [[Day 10] 模型達到商業指標的挑戰 — Test set performance 的殞落](https://ithelp.ithome.com.tw/articles/10271734)
- [[Day 11] 建立 Baseline — 開啟機器學習專案的第一步](https://ithelp.ithome.com.tw/articles/10272420)
![建立 baseline 重要性的比喻](assets/Ironman%202021/file-20260525145749582.png)
> 幕後花絮：真的很想把這個比喻用圖表達清楚，但苦於沒有作圖能力，所以花了超多時間才找到這張背景圖，差點就在這裡翻船哈哈哈
- [[Day 12] Error analysis — 錯誤中學會成長 (咩噗)](https://ithelp.ithome.com.tw/articles/10273175)
- [[Day 13] 資料增強 — 我全都要.jpg](https://ithelp.ithome.com.tw/articles/10273884)
- [[Day 14] Audit perfomance — 模型也要期末稽核༼ಢ_ಢ༽](https://ithelp.ithome.com.tw/articles/10274410)
- [[Day 15] ML 實驗管理 — 翻開覆蓋的陷阱卡~ 記帳小本本!](https://ithelp.ithome.com.tw/articles/10275212)
### 4. 資料
- [[Day 16] Data! — 資料就是我的超能力](https://ithelp.ithome.com.tw/articles/10275652)
- [[Day 17] 定義資料 — 講清楚很難嗎?](https://ithelp.ithome.com.tw/articles/10276457)
- [[Day 18] 再訪 HLP — 人(?)的表現是己欲立而立人](https://ithelp.ithome.com.tw/articles/10276854)
- [[Day 19] 收集資料 — 你要對人家負責啊!](https://ithelp.ithome.com.tw/articles/10277122)
- [[Day 20] 資料標註 (1/2) — Forget about the price tag ♫](https://ithelp.ithome.com.tw/articles/10277720)
- [[Day 21] 資料標註 (2/2) — 各種標註方法](https://ithelp.ithome.com.tw/articles/10278202)
- [[Day 22] 驗證資料 — 不可以色色! 加裝資料界的色情守門員](https://ithelp.ithome.com.tw/articles/10278667)
- [[Day 23] 資料旅程 — 好想出去玩 V1.0 ٩(●ᴗ●)۶](https://ithelp.ithome.com.tw/articles/10278825)
### 5. 計畫
- [[Day 24] Scoping — 計画通り](https://ithelp.ithome.com.tw/articles/10279726)
### 6. Final Project
- [[Day 25] Final Project (1/5) — 目標、計畫說明](https://ithelp.ithome.com.tw/articles/10279962)
- [[Day 26] Final Project (2/5) — 準備開始](https://ithelp.ithome.com.tw/articles/10280136)
- [[Day 27] Final Project (3/5) — 讓 App 在本機端運行](https://ithelp.ithome.com.tw/articles/10280584)
- [[Day 28] Final Project (4/5) — 部署模型到 Google AI Platform](https://ithelp.ithome.com.tw/articles/10281097)
- [[Day 29] Final Project (5/5) — 部署 App 到 Google App Engine](https://ithelp.ithome.com.tw/articles/10281139)

阿對了，我會盡快把 [GitHub](https://github.com/eatPizza311/iThome-2021ironman) 的 README 都補完的，那我們明年見啦～