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


## V2.1 修正

- 修正手機上黑色相機提示層沒有隱藏、覆蓋即時影像的問題。
- 加入 `.placeholder[hidden] { display: none !important; }`。
- JavaScript 開啟相機後也會直接隱藏提示層，雙重保護。
- 肩膀與髖部傾斜角正規化為 -90° 到 +90°，避免出現 -175° 這類不直覺顯示。
- 更新 CSS / JS cache-busting 版本，降低手機瀏覽器讀到舊檔的機率。


## V3：本體／近體／遠體空間

畫面增加三層空間：

- 本體空間：依人體姿態偵測後的身體 envelope 顯示。
- 近體空間（PPS）：以身體為中心，向外擴張的動態區域。
- 遠體空間：近體區域以外的畫面。

V3 使用相對人體高度的方式呈現近體空間，預設為 `0.42 × 偵測身高`。
這是單眼相機試用版的視覺化 heuristic，不等同實際公分量測。

若要正式轉為公分／公尺，需要額外做：
- 相機標定
- 已知尺寸參考物
- 深度相機 / AR 深度資訊
- 或球場/地面平面 Homography

學術概念依據：
- body schema 與身體本體界限相關
- PPS 是身體周圍可達／可被外界接觸的空間
- extrapersonal space 位於較遠、不可直接觸及的位置
- PPS 具有動態、body-part-centered 與 graded-field 特性


## V4：持物展延區域

V4 加入「展延區域（tool-use extension）」：

- 球拍：預設展延 `0.42 × 偵測身高`
- 筆：預設展延 `0.12 × 偵測身高`
- 一般工具：預設展延 `0.28 × 偵測身高`
- 自訂：可手動調整

持物手可選：
- 自動
- 左手
- 右手

### 推估方式

本版仍維持手機／平板可即時執行，因此沒有另外載入大型物件偵測模型。
展延方向由 MediaPipe Pose 的「肘部 → 手腕」向量推估，
並由手腕向外畫出一個 capsule-shaped 功能區域。

這代表它是「姿態導向的展延估計」，不是實際辨識球拍、筆或工具輪廓。

若下一版要精準辨識持物，可再加入：
- TensorFlow.js / MediaPipe Object Detector
- 自訂 racket / pen / tool object detection model
- 手部 landmarks + object bounding box 關聯


## V5：自動持物判斷

V5 改成三種展延模式：

- `自動判斷`：只有物件偵測確認「手腕附近存在可展延物件」時才顯示展延區。
- `手動開啟`：不做物件確認，直接依肘部→手腕方向顯示展延。
- `強制關閉`：永遠不顯示展延。

### 空手判斷

自動模式下，如果沒有偵測到持物：

`本體 + 近體 PPS + 遠體` 會正常顯示，但 `展延區` 不會出現。

### 防止閃爍

- 連續 2 次物件偵測週期確認後才啟動展延。
- 已確認持物消失後，連續 3 次偵測不到才關閉展延。

### 自動物件偵測

V5 使用 MediaPipe Object Detector + EfficientDet Lite0 (COCO)。
主要支援：

- tennis racket
- baseball bat
- scissors
- knife
- fork
- spoon
- toothbrush

其中球拍 (`tennis racket`) 是本版本最主要的自動辨識目標。

### 關於筆

COCO 的 EfficientDet Lite0 沒有 `pen` 類別，因此「筆」目前無法可靠自動辨識。
若要分析持筆，請使用：

`展延判斷 → 手動開啟`
`持物種類 → 筆`

若要真正自動辨識筆，需要下一階段加入自訂 pen detection 模型。

### 手腕關聯

只有物件偵測框的中心位置接近左／右手腕，才會被視為「正在持用」，
避免背景中出現球拍但人其實沒有持拍時誤開展延區。
