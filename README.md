# 人體姿態即時分析 — Mobile / Tablet V2

此版本專門調整為 GitHub Pages 上的手機／平板瀏覽器使用。

## V2 修正

- 不在頁面載入時自動初始化模型，避免手機停在「載入模型中」
- 使用 MediaPipe Tasks Vision 1.0.1
- 使用 Pose Landmarker Lite
- CPU 優先，提升 Android / iOS / 平板相容性
- 按下「啟動分析」後才載入模型與要求相機權限
- 支援前／後鏡頭切換
- 相機權限後可列出裝置相機
- 手機分析頻率限制，降低發熱與瀏覽器卡頓
- 顯示更完整的模型、相機與錯誤狀態
- 支援安全區域（iPhone / iPad）
- 手機與平板響應式版面

## GitHub 更新方式

將下列檔案覆蓋 repository 根目錄的舊版本：

- `index.html`
- `app.js`
- `style.css`
- `.nojekyll`

README.md 可一起更新，也可以保留原本 README。

GitHub Pages 不需要重新設定。Commit 後等待 Pages 自動重新部署。

## 使用方式

1. 用 HTTPS GitHub Pages 網址開啟。
2. 按「啟動分析」。
3. 等待模型載入。
4. 瀏覽器詢問 Camera 時按「允許」。
5. 人體進入畫面即可顯示骨架與分析數據。
6. 可按「切換前／後鏡頭」。

## 建議瀏覽器

- Android：Chrome / Samsung Internet / Edge
- iPhone / iPad：Safari 或最新版 Chrome
- 桌面：Chrome / Edge / Safari

## 注意

所有數據為 2D 影像推估，不可作為醫療、臨床或正式生物力學診斷。
