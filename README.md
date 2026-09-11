# V27 reliability patch

This revision prioritizes single-person consistency, safer camera switching, a more conservative held-object detector threshold, and a smoothed 2D estimated weight-bearing side. The weight-bearing value remains a 2D image estimate and is not a clinical force/pressure measurement.

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


## V6：基於白色骨架線向外展延

V6 不再用人體 bounding box 的矩形或橢圓當作近體空間基準。

新的空間計算順序：

1. MediaPipe Pose 產生白色人體骨架線。
2. 本體空間：沿每一條白線做小幅度向外擴張。
3. 近體空間 PPS：沿同一組白線再做更大的向外擴張。
4. 遠體空間：近體空間以外的畫面。

因此手臂伸出去時，近體空間會跟著手臂向外；腿部姿勢改變時，
空間也會跟著腿部骨架改變，而不是固定的人體外框。

`近體範圍` slider 控制的是「白線向外展延半徑 / 偵測身高」。


## V8：修正近體過大 + 改善持物判斷

### 近體空間
- 不再直接把骨架線大幅加粗。
- 先由肩、髖、四肢重建本體輪廓。
- 近體空間由本體輪廓小幅向外展延。
- 比例改用肩寬/身寬，不用偵測身高。
- 預設 `0.20 × 身寬`，避免半身畫面出現巨大黃色區。

### 持物判斷
V8 新增/改善：
- cell phone
- tennis racket
- baseball bat
- book
- bottle
- cup
- scissors / knife / fork / spoon / toothbrush

判斷方式改為「手腕是否碰到或靠近物件 bounding box」，
不再使用手腕到物件中心點的距離，因此對手機、球拍等長形物件更可靠。

物件偵測 threshold 改為 0.25。


## V9：木偶式本體區塊

V9 將本體顯示改成關節木偶式結構：

- 頭：獨立橢圓區塊
- 胸廓／軀幹：獨立多邊形
- 骨盆：獨立區塊
- 左右上臂
- 左右前臂
- 左右大腿
- 左右小腿
- 肩、肘、腕、髖、膝、踝：顯示關節節點

白色 Pose 骨架仍畫在本體區塊上方。

近體空間仍不是單純骨架加粗，而是由整體本體 envelope 向外展延，
因此視覺上會像「木偶本體 + 外圍近體場」。

預設近體範圍調整為 `0.18 × 身寬`。


## V10：手持物 Bounding Box

V10 會直接把物件偵測框畫在即時畫面上：

- 藍色框：Object Detector 已偵測到可展延物件。
- 黃色虛線框：物件已靠近手腕，正在確認是否為持物。
- 綠色框：已確認為手持物。
- 候選／已確認狀態會畫「手腕 → 物件框」關聯線。

支援的自動框選類別沿用 V8：
`tennis racket`, `cell phone`, `baseball bat`, `book`, `bottle`, `cup`,
`scissors`, `knife`, `fork`, `spoon`, `toothbrush`。

注意：框線來自 EfficientDet Lite0 的 bounding box，因此球拍可能是較大的矩形框，
不會精準貼合球拍輪廓；若要物件輪廓，需要 instance segmentation 模型。


## V11：修正球拍展延方向與「已持物但沒有框」

從實測畫面發現兩個核心問題：

1. 球拍已在手中，但綠色展延仍沿「肘→手腕」方向投射，與球拍實際方向不同。
2. 持物 hysteresis 仍維持「已確認」時，最新一次 detector 可能暫時漏偵測，
   因此 UI 會顯示展延但沒有 bounding box。

V11 修正：

- 自動模式：展延幾何改用已偵測物件 bounding box。
- 起點使用手腕與物件框的接觸點。
- 終點使用物件框遠端方向，不再使用前臂向量猜球拍方向。
- detector 暫時漏一兩次時，保留最後已確認物件框，避免「有展延、沒框」。
- bounding-box matching 改成容許框在不同影格有小幅位移。
- 物件偵測間隔由 450 ms 改為 250 ms。
- 手動模式才使用肘→手腕姿態推估，並明確標示「手動推估」。


## V12：統整完成版

V12 將前面各版的修正整合成一個完整 GitHub Pages 程式：

### 人體本體
- MediaPipe Pose 白色骨架。
- 木偶式本體區塊：頭、軀幹、骨盆、上臂、前臂、大腿、小腿與關節節點。
- 本體輪廓作為 PPS 基準，不直接把白色骨架粗暴放大。

### 近體 / 遠體
- 近體空間由本體 envelope 向外展延。
- 比例以身寬/肩寬為主，避免半身入鏡造成巨大 PPS。
- 遠體為 PPS 之外區域。

### 持物偵測
自動模式使用 EfficientDet Lite0，可處理 COCO 已有類別，例如：
- tennis racket
- cell phone
- baseball bat
- bottle
- cup
- book
- scissors / knife / fork / spoon / toothbrush

### 桌球拍與未知工具
COCO 不包含 table-tennis paddle / ping-pong paddle，所以 V12 提供
「手動框選持物」fallback：

1. 啟動分析。
2. 持工具進入畫面。
3. 按「手動框選持物」。
4. 在畫面拖曳框住工具。
5. 程式會判斷框是否靠近左/右手腕。
6. 只有靠近手腕才視為持物並啟用綠色工具展延。

這避免把「模型不知道這個類別」錯誤解讀為「使用者沒有持物」。

### 展延幾何
- 自動偵測成功：依物件 bounding box + 手腕接觸位置計算工具方向。
- 手動框選：依手動 ROI + 手腕接觸位置計算工具方向。
- 不再用肘→手腕方向硬猜球拍方向。
- 手動模式才使用姿態向量推估。

### 框線顏色
- 藍色：偵測到物件。
- 黃色：持物候選。
- 綠色：已確認持物。
- 橘色：手動框選但尚未靠近手腕。
- 手動框選靠近手腕後會轉為綠色。


## V13：Contact-based Holding Detection

V13 參考 100 Days of Hands / Hand Object Detector 的核心概念：
「持物」不應只由 wrist-to-object distance 決定，而應判斷 hand-object contact。

瀏覽器版沒有直接載入原研究的 Faster-RCNN/PyTorch 權重，因為該模型不是
GitHub Pages 可直接執行的 MediaPipe Tasks 模型。因此 V13 採用可在瀏覽器
即時執行的近似架構：

- MediaPipe Pose Landmarker：人體骨架、本體、PPS。
- MediaPipe Hand Landmarker：每手 21 個 landmarks。
- EfficientDet Lite0：COCO 物件 bounding box。
- Contact score：手指端點、掌心、手腕與物件框的空間接觸。
- Temporal hysteresis：連續影格確認/釋放，降低閃爍與瞬間誤判。
- Manual ROI fallback：桌球拍、筆或未知工具不在 COCO 類別時可手動框選，
  但仍由 Hand Landmarker 的接觸分數決定是否真的「持物」。

### Contact score

V13 使用以下瀏覽器端 heuristic：

- 30%：最近手指端點到物件的距離
- 25%：五個手指端點落在物件/容差框內的比例
- 20%：掌心到物件距離
- 15%：手腕到物件距離
- 10%：拇指 + 其他手指的抓握接觸 cue

分數 >= 0.50 才成為持物候選，之後仍需通過連續影格 hysteresis。

這不是 100DOH 原模型的 contact-state classifier，而是依其「contact 而非純距離」
概念改寫成可部署於純 GitHub Pages 的版本。

### 已知限制

EfficientDet Lite0 / COCO 沒有 table-tennis paddle 類別，因此無法靠降低
confidence 自動得到桌球拍名稱。V13 對這類物件保留手動 ROI fallback。
若需要「未知物件也全自動找框」，下一階段需要加入 open-vocabulary detector
或自行訓練桌球拍/工具 detector。


## V14：遠體 + 持拍判斷完整修正

### 遠體空間
V13 的遠體只是整個畫面 2.5% alpha 的藍色背景，視覺上幾乎等於沒有，
而且不是「PPS 外部」的真正幾何分類。

V14 改為：
- 本體：木偶 body envelope。
- 近體 PPS：expanded body mask - body mask，成為真正的黃色環帶。
- 遠體：canvas - expanded PPS mask，成為 PPS 外部的藍色區域。
- 三區互斥顯示，不再讓黃色 PPS 蓋住本體。

### 持拍判斷
V13 最大問題是先用 EXTENDABLE_OBJECT_LABELS 白名單過濾。
桌球拍不在 COCO；即使 EfficientDet 把它誤判為 frisbee 或其他類別，
也會在 contact 判斷前直接被丟掉。

V14 改為：
- 除 person 外，所有合理大小的 detector bbox 都可進入 hand-object contact 判斷。
- 已知工具：contact threshold 0.46。
- 未知/誤分類物件：較嚴格 threshold 0.56。
- 未知物件若通過手指/掌心接觸判斷，顯示「未知持物 / detector label」。
- 未知物件只在成為候選/確認持物時才畫框，避免畫面塞滿一般物件框。
- ObjectDetector threshold 0.18、maxResults 12，增加桌球拍被其他 COCO 類別捕捉的機會。
- 若未知物件確認持有，展延直接用其 bbox 幾何；工具種類採 UI 選擇，
  因此桌球拍請在工具類型選「桌球拍」。

### 仍保留 fallback
若 EfficientDet 對桌球拍完全沒有任何 bbox，仍可使用「手動框選持物」。
手動框選不會直接算持物，仍需 Hand Landmarker 接觸分數通過。


## V15：參考程式後的混合式修正

本版不是再單純調 threshold，而是依照下列參考架構重新整理：

- Google MediaPipe Hands：21 個手部 landmarks，適合即時瀏覽器 hand geometry。
- Google MediaPipe Pose Landmarker：支援 segmentation masks，可直接取得人體遮罩。
- 100 Days of Hands / hand_object_detector：重點是 hand-contact state 與 hand-object matching，而不是只看 wrist distance。
- QPIC / HOTR：Human-Object Interaction 的核心是 human/object/interaction 關係，而不是單獨 object label。

### 空間修正
- Pose Landmarker 開啟 `outputSegmentationMasks: true`。
- 有 segmentation 時，人體本體遮罩直接取真實人形。
- PPS 由真實人體 mask 做形態式 dilation。
- 遠體 = 畫面 - PPS。
- segmentation 不可用時才 fallback 到木偶式 body reconstruction。

### 持拍修正
優先順序：
1. Object Detector bbox + Hand Landmarker contact score。
2. Object bbox 被誤分類也可以成為未知持物。
3. Object Detector 完全漏框時，若使用者已明確選擇「桌球拍 / 網球拍 / 手機 / 工具」，
   才允許使用 Hand Landmarker 的持續握持姿態作 fallback。
4. 握持 fallback 明確標示為「握持姿態推定」，不宣稱模型真的看見了物件。

### 握持姿態分數
- 指尖相對掌心的收攏程度。
- 拇指與食指的 opposition / pinch 程度。
- 手掌整體 compactness。
- 需連續 3 個偵測週期通過門檻才啟動。
- 工具種類為「未知工具（自動安全模式）」時不啟用 grip-only fallback，
  避免單純握拳被誤判成持拍。

### 研究限制
握持姿態 fallback 是「probable holding」，不是 object recognition。
若需要桌球拍全自動精準 bbox / mask，仍需自訂桌球拍 detector 或
open-vocabulary / segmentation 模型。


## V16：手機／平板全螢幕分析

新增沉浸式分析模式：

- 啟動分析後可按「全螢幕分析」。
- 支援標準 Fullscreen API；不支援時自動使用 CSS 沉浸式滿版。
- 全螢幕隱藏頁面設定區，只保留相機、Canvas、HUD 與必要控制。
- HUD 顯示持物狀態、展延狀態與 PPS 比例。
- 全螢幕內可切換鏡頭與退出。
- 使用 `100dvh` 適應手機瀏覽器動態工具列。
- video / canvas 均維持 `object-fit: contain`，避免相機畫面被裁切後骨架、
  PPS 與物件框座標錯位。

注意：手機瀏覽器是否能完全隱藏網址列由瀏覽器政策決定；
即使 Fullscreen API 被拒絕，CSS 沉浸模式仍會把分析區填滿目前可視畫面。


## V17 整合策略

GitHub Pages 正式版繼續以 MediaPipe Tasks Vision 為核心：
- Pose Landmarker：33 點、世界座標、人體 segmentation。
- Hand Landmarker：21 點手部 landmarks，用於 contact / grip。
- EfficientDet Lite0：物件 bbox；一般顯示採較高門檻，低信心框只有在強手部接觸時才可進入持物候選。
- PPS：優先由 segmentation 人體遮罩向外膨脹；無遮罩時才使用 Pose 幾何 fallback。
- HUD / 設定面板直接顯示 `SEGMENTATION` 或 `POSE FALLBACK`，方便實機驗證。
- 效能模式：自動 / 省電 / 高精度，調整 Pose、Hand、Object 的推論節奏。
- MoveNet：介面保留備援插槽，但 V17 不同時載入第二套姿態模型，避免手機端記憶體與熱負載增加。
- OpenPose / MMPose / Apple Vision 不直接塞入靜態 GitHub Pages；適合作為離線研究比較或原生 App 路線。

重要限制：EfficientDet COCO 沒有桌球拍類別。V17 可用手部接觸與握持姿態降低漏判，
但要真正自動辨識「桌球拍」類別，仍需要自訂 detector / browser-compatible model。


## V21 Stable：從可用 V17 小幅增量修正

此版刻意不從 V18/V19/V20 繼續疊加，而是直接以實機可使用的 V17 為基礎，只加入已確認需要的功能：

- 優先顯示臉、左手、右手、左腳、右腳的中心座標。
- 座標格式為百分比，例如 `左手：x 42.8% / y 50.2%`。
- 臉／手／腳可選擇性加強 PPS，預設開啟。
- 手部局部加強權重最高；臉與腳次之。
- 本體／近體 PPS／遠體三個標籤集中顯示在頭部附近，提高手機畫面可讀性。
- 保留 V17 原本的 segmentation、持物偵測、手部 contact、全螢幕與效能模式。

設計原則：先維持 V17 可運作架構，再做最小增量；沒有額外載入 MoveNet、OpenPose 或 MMPose。


## V22：遠體空間改為獨立距離場

V22 移除舊版「遠體 = 全畫面 − PPS」的排除法。

新的遠體空間由以下資訊主動建立：
- 人體軀幹中心：優先使用左右肩 + 左右髖平均中心。
- 人體尺度：使用目前程式的身寬參考值 `refW`。
- 遠體起始半徑：預設 `0.85 × 身寬`。
- 遠體終止半徑：預設 `2.50 × 身寬`。
- 垂直方向半徑約為水平方向的 `1.22×`，形成橢圓距離場。

因此：
- 本體、PPS、遠體分別有自己的計算規則。
- 遠體不是「剩下來的畫面」。
- 超過遠體終止半徑的區域保留為未分類背景。
- 畫面會顯示遠體內／外兩條藍色虛線邊界，方便觀察距離場。

這仍是單眼 2D 相機的相對尺度估計，不等同公分／公尺。
若需要真實距離，仍需相機標定、已知尺度、深度資訊或地面 Homography。


## V23：遠體空間可拖曳 + 局部藍色顯示

V23 直接以 V22 為基礎修正遠體空間顯示：

- 遠體仍是獨立距離場，不使用 `畫面 − PPS`。
- 藍色只填在「遠體內半徑到外半徑」之間的距離帶，其他畫面完全不塗藍。
- 預設遠體起始縮為 `0.75 × 身寬`。
- 預設遠體終止縮為 `1.65 × 身寬`，避免遠體區域佔滿整個相機畫面。
- 外距離最大限制改為 `2.60 × 身寬`。
- 新增「拖曳遠體位置」：按下後可在即時影像上直接拖動遠體中心。
- 拖曳完成後，遠體中心固定在自訂位置，不再跟著人體移動。
- 按「跟隨人體」可清除自訂位置，恢復跟隨肩＋髖的軀幹中心。
- 拖曳模式與「手動框選持物」互斥，避免手機觸控操作衝突。

這個遠體位置仍是 2D 影像座標中的研究視覺化，不代表真實三維公尺位置。


## V24：遠體從近體 PPS 外輪廓繼續延伸

依實機畫面修正 V23 的橢圓／圓圈問題。

V24 的空間順序固定為：

`本體 → 近體 PPS → 遠體空間 → 未分類背景`

遠體計算方式：
1. 先完成目前的 PPS mask（包含 Segmentation、本體 envelope、臉／手／腳局部加強）。
2. 直接將這個 PPS mask 向外膨脹 `farMargin`。
3. 只保留 PPS 外緣到膨脹後外緣之間的新區域作為藍色遠體帶。
4. 藍色帶之外保持原始相機畫面，不填滿整張畫面。

因此遠體會繼承 PPS 的實際形狀：手臂伸出時，遠體跟著手臂附近的 PPS 外緣延伸；
頭部、軀幹、腿部也同理，不再形成以軀幹中心為圓心的橢圓。

「遠體向外延伸」滑桿預設為 `0.45 × 身寬`，範圍 `0.10–1.20 × 身寬`。


## V25：遠體改為深藍色

V25 不改變 V24 的遠體幾何計算，只修改遠體視覺辨識：
- 遠體仍直接從黃色 PPS 外輪廓向外延伸。
- 遠體填色改為深藍 `rgba(8, 38, 115, 0.58)`。
- 外緣提示也改為同系深藍。
- 本體紅色、PPS 黃色與持物綠色維持不變。


## V26：深藍遠體提高不透明度

- 遠體色維持深藍 RGB `(8, 38, 115)`。
- 填色 alpha 由 `0.58` 提高至 `0.82`。
- 遠體外緣提示同步加深。
- 遠體幾何不變：仍由黃色 PPS 外輪廓向外延伸。
- 本體、PPS、持物與全螢幕功能不變。


## V27.1：前／後鏡頭切換修正

- 切換鏡頭時不再使用 `video: true` 靜默 fallback，避免切換失敗卻又回到前鏡頭。
- 優先依 `enumerateDevices()` 的 camera label 尋找 `front/back/rear/environment` 對應鏡頭並用 `deviceId` 精準切換。
- 若裝置沒有可辨識 label，改用 `facingMode: { exact: "user/environment" }`。
- Samsung / Android 切換時先釋放目前相機，避免部分手機無法同時開啟前後鏡頭。
- 切換完成後會讀取 `track.getSettings()` 與相機 label 驗證實際鏡頭方向；只有符合要求才顯示「相機已切換 ✓」。
- 切換失敗會嘗試恢復原鏡頭，並明確顯示失敗原因，不再誤報成功。
