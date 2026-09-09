import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { alpha: true });

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const flipBtn = document.getElementById("flipBtn");
const cameraSelect = document.getElementById("cameraSelect");
const placeholder = document.getElementById("placeholder");

const statusEl = document.getElementById("status");
const detailEl = document.getElementById("detail");
const headEl = document.getElementById("head");
const shoulderEl = document.getElementById("shoulder");
const hipEl = document.getElementById("hip");
const leftKneeEl = document.getElementById("leftKnee");
const rightKneeEl = document.getElementById("rightKnee");
const supportEl = document.getElementById("support");

let poseLandmarker = null;
let stream = null;
let running = false;
let starting = false;
let facingMode = "user";
let lastVideoTime = -1;
let lastInferenceAt = 0;
let animationId = null;

// Mobile/tablet: limiting inference rate reduces heat and browser stalls.
const MIN_INFERENCE_INTERVAL_MS = 85;

function setStatus(text, detail = "") {
  statusEl.textContent = text;
  detailEl.textContent = detail;
}

function resetMetrics() {
  headEl.textContent = "—";
  shoulderEl.textContent = "—";
  hipEl.textContent = "—";
  leftKneeEl.textContent = "—";
  rightKneeEl.textContent = "—";
  supportEl.textContent = "—";
}

function friendlyError(err) {
  const name = err?.name || "";
  if (name === "NotAllowedError") return "相機權限被拒絕，請到瀏覽器網站權限允許 Camera。";
  if (name === "NotFoundError") return "找不到可用相機。";
  if (name === "NotReadableError") return "相機正被其他 App 使用，請關閉其他相機程式再試。";
  if (name === "OverconstrainedError") return "相機不支援目前設定，請換另一支相機。";
  if (name === "SecurityError") return "瀏覽器安全設定阻止相機，請使用 HTTPS 網址。";
  return err?.message || String(err);
}

function deg(v) {
  return Number.isFinite(v) ? `${v.toFixed(1)}°` : "—";
}

function angleLine(a, b) {
  let angle = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;

  // A shoulder/hip line has no meaningful 180-degree direction.
  // Normalize to a readable tilt: -90° .. +90°.
  while (angle > 90) angle -= 180;
  while (angle < -90) angle += 180;

  return angle;
}

function jointAngle(a, b, c) {
  const bax = a.x - b.x;
  const bay = a.y - b.y;
  const bcx = c.x - b.x;
  const bcy = c.y - b.y;
  const dot = bax * bcx + bay * bcy;
  const n1 = Math.hypot(bax, bay);
  const n2 = Math.hypot(bcx, bcy);
  if (n1 < 1e-6 || n2 < 1e-6) return null;
  const cosine = Math.max(-1, Math.min(1, dot / (n1 * n2)));
  return Math.acos(cosine) * 180 / Math.PI;
}

function visible(lm, i, min = 0.35) {
  const p = lm[i];
  return p && (p.visibility ?? 1) >= min && (p.presence ?? 1) >= min;
}

function headDirection(lm) {
  if (![0, 2, 5].every(i => visible(lm, i))) return "UNKNOWN";

  const nose = lm[0];
  const leftEye = lm[2];
  const rightEye = lm[5];
  const eyeMidX = (leftEye.x + rightEye.x) / 2;
  const eyeSpan = Math.abs(leftEye.x - rightEye.x);

  if (eyeSpan < 0.002) return "UNKNOWN";

  const dx = (nose.x - eyeMidX) / eyeSpan;
  if (dx < -0.18) return "LEFT";
  if (dx > 0.18) return "RIGHT";
  return "FORWARD";
}

function supportLeg(lm) {
  if (![23, 24, 27, 28].every(i => visible(lm, i))) return "UNKNOWN";

  const hipMidX = (lm[23].x + lm[24].x) / 2;
  const leftD = Math.abs(hipMidX - lm[27].x);
  const rightD = Math.abs(hipMidX - lm[28].x);
  const margin = 0.018;

  if (leftD + margin < rightD) return "LEFT";
  if (rightD + margin < leftD) return "RIGHT";
  return "BOTH";
}

function showAnalysis(lm) {
  headEl.textContent = headDirection(lm);
  supportEl.textContent = supportLeg(lm);

  shoulderEl.textContent =
    [11, 12].every(i => visible(lm, i))
      ? deg(angleLine(lm[11], lm[12]))
      : "—";

  hipEl.textContent =
    [23, 24].every(i => visible(lm, i))
      ? deg(angleLine(lm[23], lm[24]))
      : "—";

  leftKneeEl.textContent =
    [23, 25, 27].every(i => visible(lm, i))
      ? deg(jointAngle(lm[23], lm[25], lm[27]))
      : "—";

  rightKneeEl.textContent =
    [24, 26, 28].every(i => visible(lm, i))
      ? deg(jointAngle(lm[24], lm[26], lm[28]))
      : "—";
}

async function initPoseModel() {
  if (poseLandmarker) return;

  setStatus("正在下載分析元件…", "第一次開啟需要網路，之後瀏覽器通常會快取。");

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);

  setStatus("正在載入人體模型…", "手機／平板相容模式：CPU + Lite Pose 模型");

  // CPU is deliberately used first for wider Android/iOS/tablet compatibility.
  poseLandmarker = await PoseLandmarker.createFromModelPath(
    vision,
    MODEL_URL
  );

  await poseLandmarker.setOptions({
    runningMode: "VIDEO",
    numPoses: 2,
    minPoseDetectionConfidence: 0.45,
    minPosePresenceConfidence: 0.45,
    minTrackingConfidence: 0.45
  });

  setStatus("模型已就緒 ✓", "請允許瀏覽器使用相機。");
}

async function listCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === "videoinput");

    const currentId = stream?.getVideoTracks?.()[0]?.getSettings?.().deviceId || "";
    cameraSelect.innerHTML = "";

    cams.forEach((cam, index) => {
      const option = document.createElement("option");
      option.value = cam.deviceId;
      option.textContent = cam.label || `相機 ${index + 1}`;
      if (cam.deviceId === currentId) option.selected = true;
      cameraSelect.appendChild(option);
    });

    cameraSelect.disabled = cams.length < 2;
    flipBtn.disabled = cams.length < 2;
  } catch (err) {
    console.warn("enumerateDevices failed", err);
  }
}

function stopTracks() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }
  video.srcObject = null;
}

async function openCamera({ deviceId = "", facing = facingMode } = {}) {
  stopTracks();

  setStatus("正在開啟相機…", "若瀏覽器詢問權限，請選「允許」。");

  const baseVideo = {
    width: { ideal: 960 },
    height: { ideal: 720 },
    frameRate: { ideal: 24, max: 30 }
  };

  if (deviceId) {
    baseVideo.deviceId = { exact: deviceId };
  } else {
    baseVideo.facingMode = { ideal: facing };
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: baseVideo
    });
  } catch (firstError) {
    console.warn("Preferred camera constraints failed, retrying basic video.", firstError);
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: true
    });
  }

  video.srcObject = stream;
  video.muted = true;
  video.setAttribute("playsinline", "");
  await video.play();

  await new Promise(resolve => {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      resolve();
      return;
    }
    video.addEventListener("loadedmetadata", resolve, { once: true });
  });

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;

  const settings = stream.getVideoTracks()[0]?.getSettings?.() || {};
  if (settings.facingMode) facingMode = settings.facingMode;

  // Front camera is mirrored for natural selfie behavior; rear camera is not.
  const mirror = facingMode !== "environment";
  video.style.transform = mirror ? "scaleX(-1)" : "none";
  canvas.style.transform = mirror ? "scaleX(-1)" : "none";

  placeholder.hidden = true;
  placeholder.style.display = "none";
  await listCameras();
}

async function startAnalysis() {
  if (starting || running) return;

  if (!window.isSecureContext) {
    setStatus("無法啟動", "相機需要 HTTPS。請使用 GitHub Pages 的 https:// 網址。");
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("此瀏覽器不支援相機", "請使用最新版 Chrome、Safari、Samsung Internet 或 Edge。");
    return;
  }

  starting = true;
  startBtn.disabled = true;

  try {
    await initPoseModel();
    await openCamera({ facing: facingMode });

    running = true;
    stopBtn.disabled = false;
    setStatus("即時分析中 ✓", "請讓全身盡量進入畫面。");
    lastVideoTime = -1;
    lastInferenceAt = 0;
    animationId = requestAnimationFrame(predict);
  } catch (err) {
    console.error(err);
    stopTracks();
    setStatus("啟動失敗", friendlyError(err));
    startBtn.disabled = false;
  } finally {
    starting = false;
  }
}

function stopAnalysis() {
  running = false;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  stopTracks();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resetMetrics();

  placeholder.hidden = false;
  placeholder.style.display = "";
  startBtn.disabled = false;
  stopBtn.disabled = true;
  flipBtn.disabled = true;
  cameraSelect.disabled = true;
  setStatus("已停止", "按「啟動分析」可再次開始。");
}

async function switchCameraByFacing() {
  if (!running) return;

  facingMode = facingMode === "user" ? "environment" : "user";
  running = false;

  try {
    await openCamera({ facing: facingMode });
    running = true;
    lastVideoTime = -1;
    lastInferenceAt = 0;
    setStatus("已切換相機 ✓", facingMode === "environment" ? "目前偏好後鏡頭" : "目前偏好前鏡頭");
    animationId = requestAnimationFrame(predict);
  } catch (err) {
    console.error(err);
    setStatus("切換相機失敗", friendlyError(err));
    running = true;
    animationId = requestAnimationFrame(predict);
  }
}

async function switchCameraByDevice() {
  if (!running || !cameraSelect.value) return;

  running = false;

  try {
    await openCamera({ deviceId: cameraSelect.value });
    running = true;
    lastVideoTime = -1;
    lastInferenceAt = 0;
    setStatus("相機已切換 ✓");
    animationId = requestAnimationFrame(predict);
  } catch (err) {
    console.error(err);
    setStatus("切換相機失敗", friendlyError(err));
    running = true;
    animationId = requestAnimationFrame(predict);
  }
}

function drawResults(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result?.landmarks?.length) {
    resetMetrics();
    return;
  }

  const drawingUtils = new DrawingUtils(ctx);

  for (const lm of result.landmarks) {
    drawingUtils.drawConnectors(
      lm,
      PoseLandmarker.POSE_CONNECTIONS,
      { lineWidth: 3 }
    );
    drawingUtils.drawLandmarks(
      lm,
      { radius: 4, lineWidth: 2 }
    );
  }

  showAnalysis(result.landmarks[0]);
}

function predict(now) {
  if (!running) return;

  animationId = requestAnimationFrame(predict);

  if (
    video.readyState < 2 ||
    video.currentTime === lastVideoTime ||
    now - lastInferenceAt < MIN_INFERENCE_INTERVAL_MS
  ) {
    return;
  }

  lastVideoTime = video.currentTime;
  lastInferenceAt = now;

  try {
    const result = poseLandmarker.detectForVideo(video, performance.now());
    drawResults(result);
  } catch (err) {
    console.error("Inference error", err);
    setStatus("分析暫停", friendlyError(err));
  }
}

startBtn.addEventListener("click", startAnalysis);
stopBtn.addEventListener("click", stopAnalysis);
flipBtn.addEventListener("click", switchCameraByFacing);
cameraSelect.addEventListener("change", switchCameraByDevice);

document.addEventListener("visibilitychange", () => {
  if (document.hidden && running) {
    // Mobile browsers may suspend camera in background; stop cleanly.
    stopAnalysis();
    setStatus("已暫停", "切回頁面後請重新按「啟動分析」。");
  }
});

window.addEventListener("pagehide", () => {
  if (running || stream) stopTracks();
});

setStatus("等待啟動", "支援 Android 手機／平板、iPhone／iPad 與桌面瀏覽器。");
