# GitHub Pages 人體姿態即時分析

這個版本是純 HTML / CSS / JavaScript，可直接部署到 GitHub Pages。

## 為什麼不用 Python？

GitHub Pages 是靜態網站託管，不能直接執行 Python 伺服器程式。
因此這版改成 MediaPipe Tasks Vision，在使用者自己的瀏覽器中執行人體姿態分析。

## 功能

- 瀏覽器直接使用 Webcam / 手機相機
- MediaPipe Pose Landmarker
- 人體骨架
- 頭部粗略方向
- 肩膀傾斜
- 髖部傾斜
- 左右膝角度
- 粗略支撐腳
- 支援最多 2 人姿態偵測

## GitHub Pages 部署

1. 在 GitHub 建立新 repository，例如：

   `human-pose-web`

2. 將下列檔案上傳到 repository 根目錄：

   - `index.html`
   - `style.css`
   - `app.js`
   - `.nojekyll`

3. GitHub repository → `Settings` → `Pages`

4. `Build and deployment`：
   - Source：`Deploy from a branch`
   - Branch：`main`
   - Folder：`/ (root)`

5. 儲存後即可用：

   `https://你的GitHub帳號.github.io/human-pose-web/`

## 使用

開啟網頁後：

1. 等待「模型已就緒」
2. 按「啟動相機」
3. 允許瀏覽器使用相機
4. 人體進入畫面後即可即時分析

## 網路需求

第一次載入需要網路，因為網頁會從 CDN 載入 MediaPipe JavaScript / WASM，
並下載 Pose Landmarker 模型。

## 相機權限

瀏覽器的 `getUserMedia()` 通常要求安全來源（HTTPS）。
GitHub Pages 提供 HTTPS，因此適合直接使用相機。

## 注意

這是試用版。所有人體角度與支撐腳判定都是 2D 影像估計，
不能當作醫療、臨床或正式生物力學診斷結果。
