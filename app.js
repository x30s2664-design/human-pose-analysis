import {
  FilesetResolver,
  PoseLandmarker,
  HandLandmarker,
  ObjectDetector,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const OBJECT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite";

const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

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

const ppsScaleEl = document.getElementById("ppsScale");
const ppsScaleValueEl = document.getElementById("ppsScaleValue");

const extensionModeEl = document.getElementById("extensionMode");
const toolTypeEl = document.getElementById("toolType");
const toolHandEl = document.getElementById("toolHand");
const extensionScaleEl = document.getElementById("extensionScale");
const extensionScaleValueEl = document.getElementById("extensionScaleValue");
const extensionStatusEl = document.getElementById("extensionStatus");
const objectDetectionStatusEl = document.getElementById("objectDetectionStatus");
const manualBoxBtn = document.getElementById("manualBoxBtn");
const clearManualBoxBtn = document.getElementById("clearManualBoxBtn");
const manualBoxStatusEl = document.getElementById("manualBoxStatus");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const exitFullscreenBtn = document.getElementById("exitFullscreenBtn");
const stageFlipBtn = document.getElementById("stageFlipBtn");
const analysisHud = document.getElementById("analysisHud");
const hudObject = document.getElementById("hudObject");
const hudExtension = document.getElementById("hudExtension");
const hudPps = document.getElementById("hudPps");
const stageEl = document.querySelector(".stage");




let poseLandmarker = null;
let handLandmarker = null;
let handLandmarkerLoading = false;
let objectDetector = null;
let objectDetectorLoading = false;
let lastObjectDetectionAt = 0;
let latestObjects = [];
let latestHands = [];
let lastHandDetectionAt = 0;
let currentHoldCandidate = null;
let holdConfirmFrames = 0;
let holdMissFrames = 0;
let confirmedHeldObject = null;
let stream = null;
let running = false;
let starting = false;
let facingMode = "user";
let lastVideoTime = -1;
let lastInferenceAt = 0;
let animationId = null;

// V12 manual held-object fallback for classes not present in COCO
// (e.g. table-tennis paddle / ping-pong paddle).
let manualObjectBox = null;
let manualBoxDrawing = false;
let manualBoxStart = null;
let manualBoxCurrent = null;
let latestPoseLandmarks = null;


// Mobile/tablet: limiting inference rate reduces heat and browser stalls.
const MIN_INFERENCE_INTERVAL_MS = 85;
const OBJECT_INFERENCE_INTERVAL_MS = 250;
const HAND_INFERENCE_INTERVAL_MS = 100;

// Contact-based holding threshold. Temporal hysteresis below still applies.
const HOLDING_SCORE_THRESHOLD = 0.50;

// Hysteresis: avoid extension flicker.
// Detection must be confirmed across multiple object-detection cycles.
const HOLD_CONFIRM_CYCLES = 2;
const HOLD_RELEASE_CYCLES = 4;
const GRIP_PROXY_CONFIRM_CYCLES = 3;
const GRIP_PROXY_THRESHOLD = 0.64;


// COCO / EfficientDet Lite0 supported categories that can plausibly extend action space.
// Tennis racket is the primary target. Pen is not a COCO class, so pen remains manual/custom.
const EXTENDABLE_OBJECT_LABELS = new Set([
  "tennis racket",
  "baseball bat",
  "cell phone",
  "book",
  "bottle",
  "cup",
  "scissors",
  "knife",
  "fork",
  "spoon",
  "toothbrush"
]);

// V14: do not equate "not in our tool whitelist" with "not held".
// COCO may classify a table-tennis paddle as another category (often a round
// sports object). Strong hand contact is therefore allowed to confirm a
// generic/unknown held object.
const NON_HOLDABLE_LABELS = new Set([
  "person"
]);

const KNOWN_HOLD_THRESHOLD = 0.46;
const UNKNOWN_HOLD_THRESHOLD = 0.56;



function isFullscreenActive() {
  return !!(
    document.fullscreenElement ||
    document.webkitFullscreenElement ||
    document.body.classList.contains("analysis-fullscreen")
  );
}

function syncFullscreenUi() {
  const active = isFullscreenActive();
  document.body.classList.toggle("analysis-fullscreen", active);

  if (analysisHud) analysisHud.hidden = !active;
  if (exitFullscreenBtn) exitFullscreenBtn.hidden = !active;
  if (stageFlipBtn) stageFlipBtn.hidden = !active || !running;
  if (fullscreenBtn) {
    fullscreenBtn.textContent = active ? "退出全螢幕" : "全螢幕分析";
    fullscreenBtn.disabled = !running;
  }
}

async function enterImmersiveAnalysis() {
  if (!running || !stageEl) return;

  // CSS immersive mode is the reliable fallback on mobile browsers.
  document.body.classList.add("analysis-fullscreen");
  syncFullscreenUi();

  try {
    if (!document.fullscreenElement && !document.webkitFullscreenElement) {
      if (stageEl.requestFullscreen) {
        await stageEl.requestFullscreen({ navigationUI: "hide" });
      } else if (stageEl.webkitRequestFullscreen) {
        stageEl.webkitRequestFullscreen();
      }
    }
  } catch (err) {
    // Browsers such as iOS Safari may reject Fullscreen API. CSS mode remains.
    console.warn("Fullscreen API unavailable; using immersive CSS mode.", err);
  }

  syncFullscreenUi();
}

async function exitImmersiveAnalysis() {
  document.body.classList.remove("analysis-fullscreen");

  try {
    if (document.fullscreenElement && document.exitFullscreen) {
      await document.exitFullscreen();
    } else if (document.webkitFullscreenElement && document.webkitExitFullscreen) {
      document.webkitExitFullscreen();
    }
  } catch (err) {
    console.warn("Exit fullscreen failed", err);
  }

  syncFullscreenUi();
}

async function toggleImmersiveAnalysis() {
  if (isFullscreenActive()) {
    await exitImmersiveAnalysis();
  } else {
    await enterImmersiveAnalysis();
  }
}

function updateAnalysisHud() {
  if (hudPps && ppsScaleEl) {
    hudPps.textContent = `PPS：${Number(ppsScaleEl.value).toFixed(2)} × 身寬`;
  }

  if (hudObject) {
    if (confirmedHeldObject?.gripProxy) {
      const pct = Math.round((confirmedHeldObject.contactScore || 0) * 100);
      hudObject.textContent = `持物：握持姿態推定 ${pct}%`;
    } else if (confirmedHeldObject) {
      const shown = confirmedHeldObject.genericObject
        ? `未知持物 / ${confirmedHeldObject.label}`
        : confirmedHeldObject.label;
      hudObject.textContent = `持物：${shown}`;
    } else {
      hudObject.textContent = "持物：未確認";
    }
  }

  if (hudExtension && extensionStatusEl) {
    hudExtension.textContent = `展延：${extensionStatusEl.textContent || "—"}`;
  }
}

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
  if (extensionStatusEl) extensionStatusEl.textContent = extensionModeEl?.value === "off" ? "關閉" : "等待人體";
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


function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function landmarkPixels(lm) {
  return lm.map(p => ({
    x: p.x * canvas.width,
    y: p.y * canvas.height,
    visibility: p.visibility ?? 1
  }));
}

function bodyBounds(lm) {
  const pts = landmarkPixels(lm)
    .filter((p, i) => p.visibility >= 0.35 && i <= 32);

  if (!pts.length) return null;

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  const w = Math.max(1, maxX - minX);
  const h = Math.max(1, maxY - minY);

  return {
    minX, minY, maxX, maxY, w, h,
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2
  };
}


const TOOL_PRESETS = {
  racket: { scale: 0.42, width: 0.11, label: "網球拍" },
  paddle: { scale: 0.24, width: 0.12, label: "桌球拍" },
  pen:    { scale: 0.12, width: 0.025, label: "筆" },
  phone:  { scale: 0.16, width: 0.055, label: "手機" },
  tool:   { scale: 0.28, width: 0.055, label: "工具" },
  unknown:{ scale: 0.26, width: 0.08, label: "未知持物" },
  custom: { scale: 0.30, width: 0.055, label: "自訂" }
};

function chooseToolHand(lm, b) {
  const selected = toolHandEl?.value || "auto";
  if (selected === "left" || selected === "right") return selected;

  // MediaPipe Pose:
  // left elbow 13 / wrist 15, right elbow 14 / wrist 16.
  // AUTO chooses the visible wrist farther from the trunk center,
  // which often corresponds to the hand actively extending an object.
  const candidates = [];

  if (visible(lm, 13) && visible(lm, 15)) {
    const dx = lm[15].x * canvas.width - b.cx;
    const dy = lm[15].y * canvas.height - b.cy;
    candidates.push({ hand: "left", score: Math.hypot(dx, dy) });
  }

  if (visible(lm, 14) && visible(lm, 16)) {
    const dx = lm[16].x * canvas.width - b.cx;
    const dy = lm[16].y * canvas.height - b.cy;
    candidates.push({ hand: "right", score: Math.hypot(dx, dy) });
  }

  candidates.sort((a, b2) => b2.score - a.score);
  return candidates[0]?.hand || null;
}

function toolVectorFromPose(lm, hand) {
  const elbowIndex = hand === "left" ? 13 : 14;
  const wristIndex = hand === "left" ? 15 : 16;

  if (!visible(lm, elbowIndex) || !visible(lm, wristIndex)) return null;

  const elbow = {
    x: lm[elbowIndex].x * canvas.width,
    y: lm[elbowIndex].y * canvas.height
  };
  const wrist = {
    x: lm[wristIndex].x * canvas.width,
    y: lm[wristIndex].y * canvas.height
  };

  let dx = wrist.x - elbow.x;
  let dy = wrist.y - elbow.y;
  const length = Math.hypot(dx, dy);

  if (length < 8) return null;

  dx /= length;
  dy /= length;

  return { elbow, wrist, dx, dy };
}

function drawRoundedCapsule(start, end, radius) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const len = Math.hypot(dx, dy);
  if (len < 1) return;

  const nx = -dy / len;
  const ny = dx / len;

  const p1 = { x: start.x + nx * radius, y: start.y + ny * radius };
  const p2 = { x: end.x + nx * radius, y: end.y + ny * radius };
  const p3 = { x: end.x - nx * radius, y: end.y - ny * radius };
  const p4 = { x: start.x - nx * radius, y: start.y - ny * radius };

  ctx.beginPath();
  ctx.moveTo(p1.x, p1.y);
  ctx.lineTo(p2.x, p2.y);
  ctx.arc(end.x, end.y, radius, Math.atan2(ny, nx), Math.atan2(-ny, -nx), false);
  ctx.lineTo(p4.x, p4.y);
  ctx.arc(start.x, start.y, radius, Math.atan2(-ny, -nx), Math.atan2(ny, nx), false);
  ctx.closePath();
}


function extensionGeometryFromDetectedObject(lm, hand, held) {
  const box = held?.detection?.boundingBox;
  const wrist = wristPixels(lm, hand);
  if (!box || !wrist) return null;

  const left = box.originX;
  const top = box.originY;
  const right = box.originX + box.width;
  const bottom = box.originY + box.height;

  // Start from the hand/object contact point.
  const contact = closestPointOnBox(wrist, box);

  // Use the bbox corner farthest from the wrist as the distal object endpoint.
  // This is substantially better for rackets than extending the forearm vector.
  const corners = [
    { x: left, y: top },
    { x: right, y: top },
    { x: right, y: bottom },
    { x: left, y: bottom }
  ];

  let far = corners[0];
  let farD = -1;
  for (const p of corners) {
    const d = Math.hypot(p.x - wrist.x, p.y - wrist.y);
    if (d > farD) {
      farD = d;
      far = p;
    }
  }

  const cx = left + box.width / 2;
  const cy = top + box.height / 2;

  // Blend the far corner with bbox center to avoid a diagonal pointing to
  // an irrelevant extreme corner on large boxes.
  const distal = {
    x: far.x * 0.72 + cx * 0.28,
    y: far.y * 0.72 + cy * 0.28
  };

  return {
    wrist,
    contact,
    distal,
    box,
    center: { x: cx, y: cy }
  };
}

function drawExtensionZone(lm, b) {
  if (!extensionShouldBeActive()) {
    if (extensionStatusEl) {
      extensionStatusEl.textContent =
        extensionModeEl?.value === "off" ? "關閉" : "未偵測持物";
    }
    return;
  }

  const mode = extensionModeEl?.value || "auto";
  const hand = resolvedExtensionHand(lm, b);
  if (!hand) {
    if (extensionStatusEl) extensionStatusEl.textContent = "手部未偵測";
    return;
  }

  const toolType = resolvedToolType();
  const preset = TOOL_PRESETS[toolType] || TOOL_PRESETS.custom;

  // AUTO MODE: use actual object geometry.
  // Priority: confirmed detector result -> manual ROI fallback.
  let autoHeld = null;

  if (mode === "auto" && confirmedHeldObject?.detection) {
    autoHeld = confirmedHeldObject;
  } else if (mode === "auto" && manualObjectBox) {
    const manualHeld = manualHeldObject(lm, b);
    if (manualHeld?.validGrip) autoHeld = manualHeld;
  }

  if (mode === "auto" && confirmedHeldObject?.gripProxy) {
    const g = gripProxyGeometry(lm, hand, confirmedHeldObject, b);

    if (g) {
      const { start, end, wrist } = g;
      const radius = Math.max(10, bodyReferenceWidth(lm, b) * preset.width * 0.72);

      ctx.save();
      drawRoundedCapsule(start, end, radius);
      ctx.fillStyle = "rgba(65, 235, 145, 0.08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(75, 240, 150, 0.92)";
      ctx.lineWidth = Math.max(2, canvas.width / 480);
      ctx.setLineDash([9, 7]);
      ctx.stroke();

      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(wrist.x, wrist.y);
      ctx.lineTo(start.x, start.y);
      ctx.strokeStyle = "rgba(120, 255, 180, 0.88)";
      ctx.lineWidth = Math.max(2, canvas.width / 650);
      ctx.stroke();

      const fontSize = Math.max(14, Math.round(canvas.width / 62));
      ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
      ctx.textBaseline = "bottom";
      ctx.fillStyle = "rgba(120, 255, 180, 0.98)";
      ctx.fillText(
        `握持姿態推定 ${preset.label}`,
        clamp(end.x + 8, 8, canvas.width - fontSize * 9),
        clamp(end.y - 8, fontSize + 8, canvas.height - 8)
      );
      ctx.restore();

      if (extensionStatusEl) {
        const gripPct = Math.round((confirmedHeldObject.contactScore || 0) * 100);
        extensionStatusEl.textContent =
          `${hand === "left" ? "左手" : "右手"}・${preset.label}・握持推定 ${gripPct}%`;
      }
      return;
    }
  }

  if (mode === "auto" && autoHeld?.detection) {
    const g = extensionGeometryFromDetectedObject(lm, hand, autoHeld);

    if (g) {
      const { wrist, contact, distal, box } = g;
      const objectThickness = Math.max(12, Math.min(box.width, box.height) * 0.22);
      const margin = Math.max(10, objectThickness * 0.55);

      ctx.save();

      // Functional extension follows the detected object itself.
      drawRoundedCapsule(contact, distal, objectThickness + margin);
      ctx.fillStyle = "rgba(65, 235, 145, 0.10)";
      ctx.fill();
      ctx.lineWidth = Math.max(2, canvas.width / 480);
      ctx.strokeStyle = "rgba(75, 240, 150, 0.98)";
      ctx.setLineDash([11, 8]);
      ctx.stroke();

      // Hand → object contact.
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(wrist.x, wrist.y);
      ctx.lineTo(contact.x, contact.y);
      ctx.strokeStyle = "rgba(120, 255, 180, 0.95)";
      ctx.lineWidth = Math.max(2, canvas.width / 650);
      ctx.stroke();

      // Object functional axis.
      ctx.beginPath();
      ctx.moveTo(contact.x, contact.y);
      ctx.lineTo(distal.x, distal.y);
      ctx.strokeStyle = "rgba(120, 255, 180, 0.92)";
      ctx.lineWidth = Math.max(2, canvas.width / 650);
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(distal.x, distal.y, Math.max(5, objectThickness * 0.20), 0, Math.PI * 2);
      ctx.fillStyle = "rgba(120, 255, 180, 1)";
      ctx.fill();

      const fontSize = Math.max(15, Math.round(canvas.width / 58));
      ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
      ctx.textBaseline = "bottom";
      ctx.fillStyle = "rgba(120, 255, 180, 0.98)";
      ctx.fillText(
        `持物展延 ${preset.label}`,
        clamp(distal.x + 10, 8, canvas.width - fontSize * 8),
        clamp(distal.y - 8, fontSize + 8, canvas.height - 8)
      );

      ctx.restore();

      if (extensionStatusEl) {
        const pct = Math.round((autoHeld.confidence || 0) * 100);
        extensionStatusEl.textContent =
          autoHeld.manual
            ? `${hand === "left" ? "左手" : "右手"}・${preset.label}・手動框選`
            : `${hand === "left" ? "左手" : "右手"}・${preset.label}・物件框 ${pct}%`;
      }
      return;
    }
  }

  // MANUAL MODE ONLY: pose-guided estimate.
  const vector = toolVectorFromPose(lm, hand);
  if (!vector) {
    if (extensionStatusEl) extensionStatusEl.textContent = "等待手臂";
    return;
  }

  const scale = parseFloat(extensionScaleEl?.value || String(preset.scale));
  const toolLengthPx = Math.max(20, b.h * scale);
  const radius = Math.max(8, b.h * preset.width);

  const start = {
    x: vector.wrist.x - vector.dx * radius * 0.25,
    y: vector.wrist.y - vector.dy * radius * 0.25
  };
  const end = {
    x: vector.wrist.x + vector.dx * toolLengthPx,
    y: vector.wrist.y + vector.dy * toolLengthPx
  };

  ctx.save();
  drawRoundedCapsule(start, end, radius);
  ctx.fillStyle = "rgba(65, 235, 145, 0.10)";
  ctx.fill();
  ctx.lineWidth = Math.max(2, canvas.width / 480);
  ctx.strokeStyle = "rgba(75, 240, 150, 0.90)";
  ctx.setLineDash([11, 8]);
  ctx.stroke();

  const fontSize = Math.max(15, Math.round(canvas.width / 58));
  ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "rgba(120, 255, 180, 0.95)";
  ctx.fillText(
    `手動推估 ${preset.label}`,
    clamp(end.x + 10, 8, canvas.width - fontSize * 8),
    clamp(end.y - 8, fontSize + 8, canvas.height - 8)
  );
  ctx.restore();

  if (extensionStatusEl) {
    extensionStatusEl.textContent =
      `${hand === "left" ? "左手" : "右手"}・${preset.label}・手動推估`;
  }
}
function skeletonSegments(lm) {
  const segments = [];
  for (const connection of PoseLandmarker.POSE_CONNECTIONS) {
    const aIndex = connection.start;
    const bIndex = connection.end;
    if (aIndex <= 10 && bIndex <= 10) continue;
    if (!visible(lm, aIndex) || !visible(lm, bIndex)) continue;

    segments.push({
      aIndex, bIndex,
      ax: lm[aIndex].x * canvas.width,
      ay: lm[aIndex].y * canvas.height,
      bx: lm[bIndex].x * canvas.width,
      by: lm[bIndex].y * canvas.height
    });
  }
  return segments;
}

function pxPoint(lm, i) {
  if (!visible(lm, i)) return null;
  return { x: lm[i].x * canvas.width, y: lm[i].y * canvas.height };
}

function bodyReferenceWidth(lm, b) {
  if (visible(lm, 11) && visible(lm, 12)) {
    const s = Math.hypot(
      (lm[11].x - lm[12].x) * canvas.width,
      (lm[11].y - lm[12].y) * canvas.height
    );
    if (s > 20) return s;
  }

  if (visible(lm, 23) && visible(lm, 24)) {
    const h = Math.hypot(
      (lm[23].x - lm[24].x) * canvas.width,
      (lm[23].y - lm[24].y) * canvas.height
    );
    if (h > 20) return h * 1.15;
  }

  return Math.max(60, b.w * 0.42);
}

function newMaskCanvas() {
  const c = document.createElement("canvas");
  c.width = canvas.width;
  c.height = canvas.height;
  return c;
}

function maskCapsule(mctx, a, b, width) {
  if (!a || !b) return;
  mctx.save();
  mctx.strokeStyle = "#fff";
  mctx.lineCap = "round";
  mctx.lineJoin = "round";
  mctx.lineWidth = width;
  mctx.beginPath();
  mctx.moveTo(a.x, a.y);
  mctx.lineTo(b.x, b.y);
  mctx.stroke();
  mctx.restore();
}

function maskPolygon(mctx, pts) {
  const p = pts.filter(Boolean);
  if (p.length < 3) return;
  mctx.save();
  mctx.fillStyle = "#fff";
  mctx.beginPath();
  mctx.moveTo(p[0].x, p[0].y);
  for (let i = 1; i < p.length; i++) mctx.lineTo(p[i].x, p[i].y);
  mctx.closePath();
  mctx.fill();
  mctx.restore();
}

function buildBodyMask(lm, b, margin = 0) {
  const mask = newMaskCanvas();
  const mctx = mask.getContext("2d");
  const refW = bodyReferenceWidth(lm, b);

  const p11 = pxPoint(lm, 11), p12 = pxPoint(lm, 12);
  const p13 = pxPoint(lm, 13), p14 = pxPoint(lm, 14);
  const p15 = pxPoint(lm, 15), p16 = pxPoint(lm, 16);
  const p23 = pxPoint(lm, 23), p24 = pxPoint(lm, 24);
  const p25 = pxPoint(lm, 25), p26 = pxPoint(lm, 26);
  const p27 = pxPoint(lm, 27), p28 = pxPoint(lm, 28);

  // Keep the reconstructed body visually close to actual anatomy.
  const torsoEdge = refW * 0.07 + margin * 2;
  const upperArmW = refW * 0.18 + margin * 2;
  const forearmW  = refW * 0.15 + margin * 2;
  const thighW    = refW * 0.22 + margin * 2;
  const shinW     = refW * 0.16 + margin * 2;

  if (p11 && p12 && p23 && p24) {
    maskPolygon(mctx, [p11, p12, p24, p23]);
    maskCapsule(mctx, p11, p12, torsoEdge);
    maskCapsule(mctx, p12, p24, torsoEdge);
    maskCapsule(mctx, p24, p23, torsoEdge);
    maskCapsule(mctx, p23, p11, torsoEdge);
  }

  maskCapsule(mctx, p11, p13, upperArmW);
  maskCapsule(mctx, p13, p15, forearmW);
  maskCapsule(mctx, p12, p14, upperArmW);
  maskCapsule(mctx, p14, p16, forearmW);

  maskCapsule(mctx, p23, p25, thighW);
  maskCapsule(mctx, p25, p27, shinW);
  maskCapsule(mctx, p24, p26, thighW);
  maskCapsule(mctx, p26, p28, shinW);

  // Hands
  mctx.fillStyle = "#fff";
  for (const p of [p15, p16]) {
    if (!p) continue;
    mctx.beginPath();
    mctx.arc(p.x, p.y, refW * 0.07 + margin, 0, Math.PI * 2);
    mctx.fill();
  }

  // Head - compact oval, no giant circular PPS bubble.
  if (visible(lm, 0)) {
    const nose = pxPoint(lm, 0);
    if (nose) {
      mctx.beginPath();
      mctx.ellipse(
        nose.x,
        nose.y,
        refW * 0.19 + margin,
        refW * 0.24 + margin,
        0, 0, Math.PI * 2
      );
      mctx.fill();
    }
  }

  return mask;
}

function paintMaskColor(mask, color) {
  const tmp = newMaskCanvas();
  const tctx = tmp.getContext("2d");
  tctx.drawImage(mask, 0, 0);
  tctx.globalCompositeOperation = "source-in";
  tctx.fillStyle = color;
  tctx.fillRect(0, 0, tmp.width, tmp.height);
  ctx.drawImage(tmp, 0, 0);
}


function drawRotatedCapsule(a, b, width, fillStyle, strokeStyle = null) {
  if (!a || !b) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = width;
  ctx.strokeStyle = fillStyle;

  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();

  if (strokeStyle) {
    ctx.lineWidth = Math.max(1.5, width * 0.06);
    ctx.strokeStyle = strokeStyle;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  ctx.restore();
}

function drawJointDisc(p, radius, fillStyle, strokeStyle = null) {
  if (!p) return;

  ctx.save();
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fillStyle;
  ctx.fill();

  if (strokeStyle) {
    ctx.lineWidth = Math.max(1.5, radius * 0.14);
    ctx.strokeStyle = strokeStyle;
    ctx.stroke();
  }
  ctx.restore();
}

function drawMannequinBody(lm, b) {
  const refW = bodyReferenceWidth(lm, b);

  const p0  = pxPoint(lm, 0);
  const p11 = pxPoint(lm, 11), p12 = pxPoint(lm, 12);
  const p13 = pxPoint(lm, 13), p14 = pxPoint(lm, 14);
  const p15 = pxPoint(lm, 15), p16 = pxPoint(lm, 16);
  const p23 = pxPoint(lm, 23), p24 = pxPoint(lm, 24);
  const p25 = pxPoint(lm, 25), p26 = pxPoint(lm, 26);
  const p27 = pxPoint(lm, 27), p28 = pxPoint(lm, 28);

  const red = "rgba(238, 92, 74, 0.42)";
  const redEdge = "rgba(255, 126, 108, 0.78)";
  const joint = "rgba(232, 82, 66, 0.58)";

  // Head block
  if (p0) {
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(
      p0.x, p0.y,
      Math.max(16, refW * 0.18),
      Math.max(20, refW * 0.23),
      0, 0, Math.PI * 2
    );
    ctx.fillStyle = red;
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, refW * 0.015);
    ctx.strokeStyle = redEdge;
    ctx.stroke();
    ctx.restore();
  }

  // Torso block = shoulder trapezoid to hips
  if (p11 && p12 && p23 && p24) {
    const shoulderMid = { x: (p11.x + p12.x)/2, y: (p11.y + p12.y)/2 };
    const hipMid = { x: (p23.x + p24.x)/2, y: (p23.y + p24.y)/2 };

    const insetShoulder = 0.05;
    const insetHip = 0.02;

    const t11 = {
      x: p11.x + (shoulderMid.x - p11.x) * insetShoulder,
      y: p11.y + (shoulderMid.y - p11.y) * insetShoulder
    };
    const t12 = {
      x: p12.x + (shoulderMid.x - p12.x) * insetShoulder,
      y: p12.y + (shoulderMid.y - p12.y) * insetShoulder
    };
    const t23 = {
      x: p23.x + (hipMid.x - p23.x) * insetHip,
      y: p23.y + (hipMid.y - p23.y) * insetHip
    };
    const t24 = {
      x: p24.x + (hipMid.x - p24.x) * insetHip,
      y: p24.y + (hipMid.y - p24.y) * insetHip
    };

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(t11.x, t11.y);
    ctx.lineTo(t12.x, t12.y);
    ctx.lineTo(t24.x, t24.y);
    ctx.lineTo(t23.x, t23.y);
    ctx.closePath();
    ctx.fillStyle = red;
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, refW * 0.015);
    ctx.strokeStyle = redEdge;
    ctx.stroke();
    ctx.restore();

    // Separate pelvis block, giving a mannequin-like two-piece torso.
    const pelvisTopY = hipMid.y - refW * 0.05;
    const pelvisBottomY = hipMid.y + refW * 0.13;
    const pelvisHalfW = Math.max(refW * 0.22, Math.abs(p24.x - p23.x) * 0.55);

    ctx.save();
    ctx.beginPath();
    ctx.roundRect(
      hipMid.x - pelvisHalfW,
      pelvisTopY,
      pelvisHalfW * 2,
      Math.max(12, pelvisBottomY - pelvisTopY),
      Math.max(6, refW * 0.06)
    );
    ctx.fillStyle = red;
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, refW * 0.015);
    ctx.strokeStyle = redEdge;
    ctx.stroke();
    ctx.restore();
  }

  // Limb blocks
  const upperArmW = Math.max(12, refW * 0.17);
  const forearmW  = Math.max(10, refW * 0.14);
  const thighW    = Math.max(14, refW * 0.21);
  const shinW     = Math.max(11, refW * 0.15);

  drawRotatedCapsule(p11, p13, upperArmW, red, redEdge);
  drawRotatedCapsule(p13, p15, forearmW,  red, redEdge);
  drawRotatedCapsule(p12, p14, upperArmW, red, redEdge);
  drawRotatedCapsule(p14, p16, forearmW,  red, redEdge);

  drawRotatedCapsule(p23, p25, thighW, red, redEdge);
  drawRotatedCapsule(p25, p27, shinW,  red, redEdge);
  drawRotatedCapsule(p24, p26, thighW, red, redEdge);
  drawRotatedCapsule(p26, p28, shinW,  red, redEdge);

  // Joint discs emphasize articulation like a wooden mannequin.
  const shoulderR = Math.max(6, refW * 0.055);
  const elbowR = Math.max(5, refW * 0.045);
  const wristR = Math.max(5, refW * 0.04);
  const hipR = Math.max(6, refW * 0.055);
  const kneeR = Math.max(5, refW * 0.05);
  const ankleR = Math.max(5, refW * 0.04);

  for (const p of [p11, p12]) drawJointDisc(p, shoulderR, joint, redEdge);
  for (const p of [p13, p14]) drawJointDisc(p, elbowR, joint, redEdge);
  for (const p of [p15, p16]) drawJointDisc(p, wristR, joint, redEdge);
  for (const p of [p23, p24]) drawJointDisc(p, hipR, joint, redEdge);
  for (const p of [p25, p26]) drawJointDisc(p, kneeR, joint, redEdge);
  for (const p of [p27, p28]) drawJointDisc(p, ankleR, joint, redEdge);
}


function segmentationBodyMask(result) {
  const mpMask = result?.segmentationMasks?.[0];
  if (!mpMask || typeof mpMask.getAsFloat32Array !== "function") return null;

  let values;
  try {
    values = mpMask.getAsFloat32Array();
  } catch (err) {
    console.warn("Segmentation mask unavailable", err);
    return null;
  }

  if (!values?.length) return null;

  const sourceW = Number(mpMask.width || canvas.width);
  const sourceH = Number(mpMask.height || canvas.height);
  if (!sourceW || !sourceH) return null;

  // Build at half resolution for mobile speed, then scale back up.
  const targetW = Math.max(1, Math.round(sourceW / 2));
  const targetH = Math.max(1, Math.round(sourceH / 2));
  const small = document.createElement("canvas");
  small.width = targetW;
  small.height = targetH;
  const sctx = small.getContext("2d");
  const img = sctx.createImageData(targetW, targetH);

  for (let y = 0; y < targetH; y++) {
    const sy = Math.min(sourceH - 1, Math.floor(y * sourceH / targetH));
    for (let x = 0; x < targetW; x++) {
      const sx = Math.min(sourceW - 1, Math.floor(x * sourceW / targetW));
      const confidence = values[sy * sourceW + sx] || 0;
      const alpha = confidence >= 0.48 ? 255 : confidence >= 0.30 ? 150 : 0;
      const i = (y * targetW + x) * 4;
      img.data[i] = 255;
      img.data[i + 1] = 255;
      img.data[i + 2] = 255;
      img.data[i + 3] = alpha;
    }
  }

  sctx.putImageData(img, 0, 0);

  const full = newMaskCanvas();
  const fctx = full.getContext("2d");
  fctx.imageSmoothingEnabled = true;
  fctx.drawImage(small, 0, 0, full.width, full.height);
  return full;
}

function dilateMask(mask, radius) {
  const out = newMaskCanvas();
  const octx = out.getContext("2d");
  const r = Math.max(0, radius);

  if (!mask || r <= 0.5) {
    if (mask) octx.drawImage(mask, 0, 0);
    return out;
  }

  // Morphological-style dilation using repeated circular offsets.
  // This follows the actual silhouette far better than widening pose bones.
  const samples = 24;
  octx.globalAlpha = 1;
  octx.drawImage(mask, 0, 0);

  for (let i = 0; i < samples; i++) {
    const a = i / samples * Math.PI * 2;
    const dx = Math.cos(a) * r;
    const dy = Math.sin(a) * r;
    octx.drawImage(mask, dx, dy);
  }

  // Add a middle ring to close small holes.
  const r2 = r * 0.55;
  for (let i = 0; i < 12; i++) {
    const a = i / 12 * Math.PI * 2;
    octx.drawImage(mask, Math.cos(a) * r2, Math.sin(a) * r2);
  }

  return out;
}

function handGripFeatures(handPts) {
  if (!handPts?.length) return null;

  const wrist = handPts[0];
  const mcpIds = [5, 9, 13, 17];
  const tipIds = [4, 8, 12, 16, 20];
  const mcps = mcpIds.map(i => handPts[i]).filter(Boolean);
  const tips = tipIds.map(i => handPts[i]).filter(Boolean);
  if (!wrist || mcps.length < 3 || tips.length < 4) return null;

  const palmCenter = {
    x: mcps.reduce((s, p) => s + p.x, 0) / mcps.length,
    y: mcps.reduce((s, p) => s + p.y, 0) / mcps.length
  };

  const palmSize = Math.max(
    8,
    mcps.reduce((s, p) => s + Math.hypot(p.x - wrist.x, p.y - wrist.y), 0) / mcps.length
  );

  const tipPalm = tips.map(p => Math.hypot(p.x - palmCenter.x, p.y - palmCenter.y) / palmSize);
  const curlScore = tipPalm.reduce((s, d) => s + clamp(1.35 - d, 0, 1), 0) / tipPalm.length;

  const thumb = handPts[4];
  const index = handPts[8];
  const middle = handPts[12];
  const pinchD = thumb && index ? Math.hypot(thumb.x - index.x, thumb.y - index.y) / palmSize : 2;
  const thumbOpposition = clamp(1.20 - pinchD, 0, 1);

  const compactness = middle
    ? clamp(1.55 - Math.hypot(middle.x - wrist.x, middle.y - wrist.y) / palmSize, 0, 1)
    : 0;

  const score =
    0.58 * curlScore +
    0.24 * thumbOpposition +
    0.18 * compactness;

  const axisDx = palmCenter.x - wrist.x;
  const axisDy = palmCenter.y - wrist.y;
  const axisLen = Math.hypot(axisDx, axisDy);

  return {
    score,
    curlScore,
    thumbOpposition,
    compactness,
    wrist,
    palmCenter,
    axis: axisLen > 1
      ? { dx: axisDx / axisLen, dy: axisDy / axisLen }
      : null
  };
}

function findGripProxyCandidate(lm, b) {
  if (!lm || !b || !latestHands?.length) return null;

  // Grip-only fallback must never pretend it recognized an object.
  // It is enabled only when the user explicitly selected a physical tool type.
  const selectedTool = toolTypeEl?.value || "unknown";
  if (selectedTool === "unknown" || selectedTool === "custom") return null;

  const mapped = assignDetectedHandsToPose(lm);
  let best = null;

  for (const side of ["left", "right"]) {
    const pts = mapped[side];
    if (!pts) continue;

    const f = handGripFeatures(pts);
    if (!f || f.score < GRIP_PROXY_THRESHOLD || !f.axis) continue;

    // Prefer the hand extending farther from the body center.
    const wrist = wristPixels(lm, side);
    if (!wrist) continue;
    const extension = Math.hypot(wrist.x - b.cx, wrist.y - b.cy) /
      Math.max(40, bodyReferenceWidth(lm, b));

    const rank = f.score * 2 + clamp(extension / 2.2, 0, 1);

    if (!best || rank > best.rank) {
      best = {
        label: "grip proxy",
        displayLabel: "握持姿態推定",
        confidence: 0,
        hand: side,
        contactScore: f.score,
        fingertipHits: 0,
        detection: null,
        rank,
        contactBased: false,
        genericObject: true,
        gripProxy: true,
        gripFeatures: f
      };
    }
  }

  return best;
}

function gripProxyGeometry(lm, hand, held, b) {
  const f = held?.gripFeatures;
  if (!f?.axis) return null;

  const preset = TOOL_PRESETS[resolvedToolType()] || TOOL_PRESETS.unknown;
  const lengthPx = Math.max(40, b.h * preset.scale);
  const start = {
    x: f.palmCenter.x,
    y: f.palmCenter.y
  };
  const end = {
    x: start.x + f.axis.dx * lengthPx,
    y: start.y + f.axis.dy * lengthPx
  };

  return { start, end, wrist: f.wrist };
}

function drawSpaceZones(lm, result = null) {
  const b = bodyBounds(lm);
  if (!b) return;

  const refW = bodyReferenceWidth(lm, b);
  const scale = parseFloat(ppsScaleEl?.value || "0.18");
  const ppsMargin = Math.max(10, refW * scale);

  // Prefer the actual Pose Landmarker human segmentation mask.
  // Fall back to the articulated reconstruction when segmentation is absent.
  const segmentedBody = segmentationBodyMask(result);
  const bodyMask = segmentedBody || buildBodyMask(lm, b, 0);
  const ppsMask = segmentedBody
    ? dilateMask(segmentedBody, ppsMargin)
    : buildBodyMask(lm, b, ppsMargin);

  // FAR SPACE = everything outside the PPS envelope.
  const farLayer = newMaskCanvas();
  const fctx = farLayer.getContext("2d");
  fctx.fillStyle = "rgba(70, 125, 235, 0.17)";
  fctx.fillRect(0, 0, farLayer.width, farLayer.height);
  fctx.globalCompositeOperation = "destination-out";
  fctx.drawImage(ppsMask, 0, 0);
  fctx.globalCompositeOperation = "source-over";
  ctx.drawImage(farLayer, 0, 0);

  // Draw a subtle PPS outer boundary so the far/peripersonal transition
  // remains visible even on bright backgrounds.
  ctx.save();
  ctx.globalAlpha = 0.50;
  ctx.drawImage(ppsMask, 0, 0);
  ctx.globalCompositeOperation = "source-in";
  ctx.strokeStyle = "rgba(255, 225, 100, 0.95)";
  ctx.lineWidth = Math.max(1.5, canvas.width / 700);
  ctx.restore();

  // PPS RING = expanded body envelope minus actual body silhouette.
  const ppsRing = newMaskCanvas();
  const prctx = ppsRing.getContext("2d");
  prctx.drawImage(ppsMask, 0, 0);
  prctx.globalCompositeOperation = "destination-out";
  prctx.drawImage(bodyMask, 0, 0);
  prctx.globalCompositeOperation = "source-over";
  paintMaskColor(ppsRing, "rgba(255, 214, 70, 0.25)");

  // Keep the articulated body visualization on top of the true silhouette.
  drawMannequinBody(lm, b);

  const fontSize = Math.max(13, Math.round(canvas.width / 70));
  ctx.save();
  ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  ctx.fillStyle = "rgba(255, 110, 110, 0.98)";
  ctx.fillText("本體", clamp(b.maxX - 48, 8, canvas.width - 60),
               clamp(b.minY + b.h * 0.52, 8, canvas.height - 26));

  ctx.fillStyle = "rgba(255, 230, 110, 0.98)";
  ctx.fillText("近體 PPS", clamp(b.maxX - 85, 8, canvas.width - 105),
               clamp(b.minY - ppsMargin * 0.45, 8, canvas.height - 26));

  ctx.fillStyle = "rgba(170, 205, 255, 1)";
  ctx.fillText("遠體", 12, canvas.height - fontSize - 12);
  ctx.restore();

  drawExtensionZone(lm, b);
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



async function initHandLandmarker() {
  if (handLandmarker || handLandmarkerLoading) return;

  handLandmarkerLoading = true;

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: HAND_MODEL_URL
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence: 0.35,
      minTrackingConfidence: 0.35
    });
  } catch (err) {
    console.error("Hand landmarker init failed", err);
  } finally {
    handLandmarkerLoading = false;
  }
}

function handPixels(handLm) {
  if (!handLm?.length) return null;
  return handLm.map(p => ({
    x: p.x * canvas.width,
    y: p.y * canvas.height
  }));
}

function boxExpanded(box, pad) {
  return {
    originX: box.originX - pad,
    originY: box.originY - pad,
    width: box.width + pad * 2,
    height: box.height + pad * 2
  };
}

function pointInsideBox(p, box) {
  return !!p && !!box &&
    p.x >= box.originX &&
    p.x <= box.originX + box.width &&
    p.y >= box.originY &&
    p.y <= box.originY + box.height;
}

function poseWristPoint(lm, hand) {
  return wristPixels(lm, hand);
}

function assignDetectedHandsToPose(lm) {
  const result = { left: null, right: null };
  if (!lm || !latestHands?.length) return result;

  const poseWrists = {
    left: poseWristPoint(lm, "left"),
    right: poseWristPoint(lm, "right")
  };

  const candidates = latestHands
    .map((handLm, idx) => {
      const pts = handPixels(handLm);
      return pts ? { idx, pts, wrist: pts[0] } : null;
    })
    .filter(Boolean);

  // Geometry-based matching is robust to front-camera mirroring and
  // handedness-label ambiguity.
  const pairs = [];
  for (const h of candidates) {
    for (const side of ["left", "right"]) {
      const pw = poseWrists[side];
      if (!pw) continue;
      pairs.push({
        handIdx: h.idx,
        side,
        distance: Math.hypot(h.wrist.x - pw.x, h.wrist.y - pw.y),
        pts: h.pts
      });
    }
  }
  pairs.sort((a, b) => a.distance - b.distance);

  const usedHands = new Set();
  const usedSides = new Set();
  for (const pair of pairs) {
    if (usedHands.has(pair.handIdx) || usedSides.has(pair.side)) continue;
    result[pair.side] = pair.pts;
    usedHands.add(pair.handIdx);
    usedSides.add(pair.side);
  }

  return result;
}

function handObjectContactFeatures(handPts, box, bodyWidth) {
  if (!handPts || !box) return null;

  const pad = Math.max(10, bodyWidth * 0.045);
  const expanded = boxExpanded(box, pad);

  // MediaPipe Hands: thumb tip=4, index=8, middle=12, ring=16, pinky=20.
  const fingertips = [4, 8, 12, 16, 20]
    .map(i => handPts[i])
    .filter(Boolean);

  const palmIds = [0, 5, 9, 13, 17];
  const palmPts = palmIds.map(i => handPts[i]).filter(Boolean);
  const palmCenter = palmPts.length
    ? {
        x: palmPts.reduce((s, p) => s + p.x, 0) / palmPts.length,
        y: palmPts.reduce((s, p) => s + p.y, 0) / palmPts.length
      }
    : handPts[0];

  const wrist = handPts[0];

  const fingertipHits = fingertips.filter(p => pointInsideBox(p, expanded)).length;
  const fingertipRatio = fingertips.length ? fingertipHits / fingertips.length : 0;

  const minTipDistance = fingertips.length
    ? Math.min(...fingertips.map(p => pointToBoxDistance(p, box, pad)))
    : 9999;

  const wristDistance = pointToBoxDistance(wrist, box, pad);
  const palmDistance = pointToBoxDistance(palmCenter, box, pad);

  const scale = Math.max(24, bodyWidth * 0.16);
  const proximityScore = Math.max(0, 1 - minTipDistance / scale);
  const wristScore = Math.max(0, 1 - wristDistance / (scale * 1.25));
  const palmScore = Math.max(0, 1 - palmDistance / (scale * 1.15));

  // Pinch/contact cue: thumb + at least one finger touching the object region.
  const thumbHit = pointInsideBox(handPts[4], expanded) ? 1 : 0;
  const opposingHit = [8, 12, 16, 20].some(i => pointInsideBox(handPts[i], expanded)) ? 1 : 0;
  const gripCue = thumbHit && opposingHit ? 1 : Math.max(thumbHit, opposingHit) * 0.45;

  // Inspired by hand-object contact pipelines: use hand/object spatial
  // contact, not wrist distance alone. Hysteresis supplies temporal stability.
  const score =
    0.30 * proximityScore +
    0.25 * fingertipRatio +
    0.20 * palmScore +
    0.15 * wristScore +
    0.10 * gripCue;

  return {
    score,
    fingertipHits,
    fingertipRatio,
    minTipDistance,
    wristDistance,
    palmDistance,
    gripCue
  };
}

function holdingCandidateForDetection(lm, b, det) {
  const cat = categoryOfDetection(det);
  if (!cat) return null;

  const box = det?.boundingBox;
  if (!box) return null;

  if (NON_HOLDABLE_LABELS.has(cat.name)) return null;

  // Reject extremely large scene/background boxes. Held objects can be large
  // near the camera, but should not occupy nearly the whole frame.
  const boxArea = Math.max(0, box.width * box.height);
  const frameArea = Math.max(1, canvas.width * canvas.height);
  if (boxArea / frameArea > 0.58) return null;

  const bodyWidth =
    visible(lm, 11) && visible(lm, 12)
      ? Math.hypot(
          (lm[11].x - lm[12].x) * canvas.width,
          (lm[11].y - lm[12].y) * canvas.height
        )
      : Math.max(60, b.w * 0.4);

  const mappedHands = assignDetectedHandsToPose(lm);
  const isKnownTool = EXTENDABLE_OBJECT_LABELS.has(cat.name);
  const threshold = isKnownTool ? KNOWN_HOLD_THRESHOLD : UNKNOWN_HOLD_THRESHOLD;
  let best = null;

  for (const side of ["left", "right"]) {
    const handPts = mappedHands[side];

    if (handPts) {
      const f = handObjectContactFeatures(handPts, box, bodyWidth);
      if (!f || f.score < threshold) continue;

      // Unknown labels require stronger physical evidence, but are still
      // accepted when the hand is actually wrapping/touching the object.
      const rank =
        f.score * 3 +
        cat.score +
        (isKnownTool ? 0.30 : 0);

      if (!best || rank > best.rank) {
        best = {
          label: cat.name,
          displayLabel: isKnownTool ? cat.name : `unknown (${cat.name})`,
          confidence: cat.score,
          hand: side,
          distance: f.minTipDistance,
          contactScore: f.score,
          fingertipHits: f.fingertipHits,
          detection: det,
          rank,
          contactBased: true,
          genericObject: !isKnownTool
        };
      }
      continue;
    }

    // Fallback only if Hand Landmarker temporarily loses the hand.
    // For unknown labels this fallback is intentionally stricter.
    const wrist = wristPixels(lm, side);
    const relation = wristTouchesObject(wrist, det, bodyWidth);
    if (!relation?.touches) continue;

    const fallbackScore = Math.max(0, 0.45 - relation.distance / Math.max(1, bodyWidth));
    const minFallback = isKnownTool ? 0.25 : 0.34;
    if (fallbackScore < minFallback) continue;

    const rank = cat.score + fallbackScore + (isKnownTool ? 0.2 : 0);
    if (!best || rank > best.rank) {
      best = {
        label: cat.name,
        displayLabel: isKnownTool ? cat.name : `unknown (${cat.name})`,
        confidence: cat.score,
        hand: side,
        distance: relation.distance,
        contactScore: fallbackScore,
        fingertipHits: 0,
        detection: det,
        rank,
        contactBased: false,
        genericObject: !isKnownTool
      };
    }
  }

  return best;
}
async function initObjectDetector() {
  if (objectDetector || objectDetectorLoading) return;

  objectDetectorLoading = true;
  if (objectDetectionStatusEl) {
    objectDetectionStatusEl.textContent = "載入物件模型中…";
  }

  try {
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);

    objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: OBJECT_MODEL_URL
      },
      runningMode: "VIDEO",
      scoreThreshold: 0.18,
      maxResults: 12
    });

    if (objectDetectionStatusEl) {
      objectDetectionStatusEl.textContent = "物件模型已就緒";
    }
  } catch (err) {
    console.error("Object detector init failed", err);
    if (objectDetectionStatusEl) {
      objectDetectionStatusEl.textContent = "物件模型載入失敗";
    }
  } finally {
    objectDetectorLoading = false;
  }
}

function normalizeCategoryName(name) {
  return String(name || "").trim().toLowerCase();
}

function detectionCenter(det) {
  const box = det?.boundingBox;
  if (!box) return null;
  return {
    x: box.originX + box.width / 2,
    y: box.originY + box.height / 2
  };
}

function pointToBoxDistance(p, box, padding = 0) {
  if (!p || !box) return Infinity;

  const left = box.originX - padding;
  const top = box.originY - padding;
  const right = box.originX + box.width + padding;
  const bottom = box.originY + box.height + padding;

  const dx =
    p.x < left ? left - p.x :
    p.x > right ? p.x - right : 0;

  const dy =
    p.y < top ? top - p.y :
    p.y > bottom ? p.y - bottom : 0;

  return Math.hypot(dx, dy);
}

function wristTouchesObject(wrist, det, bodyWidth) {
  const box = det?.boundingBox;
  if (!wrist || !box) return null;

  // Use a small hand-sized padding around the object bbox.
  // This works much better for long rackets and phones than comparing
  // wrist distance to the bounding-box CENTER.
  const padding = Math.max(18, bodyWidth * 0.10);
  const edgeDistance = pointToBoxDistance(wrist, box, padding);

  return {
    distance: edgeDistance,
    touches: edgeDistance <= Math.max(22, bodyWidth * 0.12)
  };
}

function wristPixels(lm, hand) {
  const idx = hand === "left" ? 15 : 16;
  if (!visible(lm, idx)) return null;
  return {
    x: lm[idx].x * canvas.width,
    y: lm[idx].y * canvas.height
  };
}

function categoryOfDetection(det) {
  if (det?.__manual) {
    return { name: "manual object", score: 1 };
  }

  const cat = det?.categories?.[0];
  if (!cat) return null;
  return {
    name: normalizeCategoryName(cat.categoryName || cat.displayName),
    score: Number(cat.score || 0)
  };
}

function findHeldObjectNearHands(lm, b, detections) {
  if (!lm || !b || !detections?.length) return null;

  let best = null;
  for (const det of detections) {
    const candidate = holdingCandidateForDetection(lm, b, det);
    if (!candidate) continue;
    if (!best || candidate.rank > best.rank) best = candidate;
  }
  return best;
}
function updateHoldState(candidate) {
  const sameAsPrevious =
    candidate &&
    currentHoldCandidate &&
    candidate.label === currentHoldCandidate.label &&
    candidate.hand === currentHoldCandidate.hand;

  if (candidate) {
    holdMissFrames = 0;

    if (sameAsPrevious) {
      holdConfirmFrames += 1;
    } else {
      currentHoldCandidate = candidate;
      holdConfirmFrames = 1;
    }

    const requiredCycles = candidate.gripProxy
      ? GRIP_PROXY_CONFIRM_CYCLES
      : HOLD_CONFIRM_CYCLES;

    if (holdConfirmFrames >= requiredCycles) {
      confirmedHeldObject = candidate;
    }
  } else {
    holdConfirmFrames = 0;
    currentHoldCandidate = null;

    if (confirmedHeldObject) {
      holdMissFrames += 1;
      if (holdMissFrames >= HOLD_RELEASE_CYCLES) {
        confirmedHeldObject = null;
        holdMissFrames = 0;
      }
    }
  }

  if (objectDetectionStatusEl) {
    if (confirmedHeldObject) {
      const pct = Math.round(confirmedHeldObject.confidence * 100);
      const contactPct = Math.round((confirmedHeldObject.contactScore || 0) * 100);
      if (confirmedHeldObject.gripProxy) {
        objectDetectionStatusEl.textContent =
          `握持姿態推定：${confirmedHeldObject.hand === "left" ? "左手" : "右手"}・${contactPct}%・無物件框`;
      } else {
        const shownLabel = confirmedHeldObject.genericObject
          ? `未知持物 / ${confirmedHeldObject.label}`
          : confirmedHeldObject.label;
        objectDetectionStatusEl.textContent =
          `已確認：${shownLabel}・${confirmedHeldObject.hand === "left" ? "左手" : "右手"}・物件 ${pct}%・接觸 ${contactPct}%`;
      }
    } else if (candidate) {
      objectDetectionStatusEl.textContent =
        candidate.gripProxy
          ? `握持姿態確認中：${candidate.hand === "left" ? "左手" : "右手"}`
          : `確認中：${candidate.genericObject ? `未知物件 / ${candidate.label}` : candidate.label}`;
    } else {
      objectDetectionStatusEl.textContent = latestObjects?.length ? "已偵測物件，等待手部接觸" : "未偵測持物";
    }
  }
}


function detectHandsIfNeeded(nowMs) {
  if (extensionModeEl?.value !== "auto") return;
  if (!handLandmarker) return;
  if (nowMs - lastHandDetectionAt < HAND_INFERENCE_INTERVAL_MS) return;

  lastHandDetectionAt = nowMs;

  try {
    const result = handLandmarker.detectForVideo(video, performance.now());
    latestHands = result?.landmarks || [];
  } catch (err) {
    console.warn("Hand landmark detection failed", err);
  }
}

async function detectObjectsIfNeeded(nowMs, lm, b) {
  if (extensionModeEl?.value !== "auto") return;
  if (!objectDetector) return;
  if (nowMs - lastObjectDetectionAt < OBJECT_INFERENCE_INTERVAL_MS) return;

  lastObjectDetectionAt = nowMs;

  try {
    const result = objectDetector.detectForVideo(video, performance.now());
    latestObjects = result?.detections || [];
    let candidate = findHeldObjectNearHands(lm, b, latestObjects);

    // Reference-inspired hybrid fallback:
    // if the detector produces no usable held-object bbox, infer only
    // "probable holding" from persistent hand-grip posture.
    if (!candidate) {
      candidate = findGripProxyCandidate(lm, b);
    }

    updateHoldState(candidate);
  } catch (err) {
    console.warn("Object detection failed", err);
  }
}

function extensionShouldBeActive() {
  const mode = extensionModeEl?.value || "auto";
  if (mode === "off") return false;
  if (mode === "manual") return true;

  if (confirmedHeldObject) return true;

  if (manualObjectBox && latestPoseLandmarks) {
    const b = bodyBounds(latestPoseLandmarks);
    const held = b ? manualHeldObject(latestPoseLandmarks, b) : null;
    return !!held?.validGrip;
  }

  return false;
}

function resolvedExtensionHand(lm, b) {
  const mode = extensionModeEl?.value || "auto";

  if (mode === "auto" && confirmedHeldObject?.hand) {
    return confirmedHeldObject.hand;
  }

  if (mode === "auto" && manualObjectBox) {
    const held = manualHeldObject(lm, b);
    if (held?.validGrip) return held.hand;
  }

  return chooseToolHand(lm, b);
}

function resolvedToolType() {
  const mode = extensionModeEl?.value || "auto";

  if (mode === "auto" && confirmedHeldObject) {
    if (confirmedHeldObject.label === "tennis racket") return "racket";
    if (confirmedHeldObject.label === "cell phone") return "phone";
    if (confirmedHeldObject.label === "baseball bat") return "tool";

    // If the detector found an object but classified it as an unrelated COCO
    // category, keep the bbox/contact evidence and let the user specify its
    // physical tool type (e.g. 桌球拍).
    if (confirmedHeldObject.genericObject) {
      return toolTypeEl?.value || "unknown";
    }
    return "tool";
  }

  if (mode === "auto" && manualObjectBox) {
    return toolTypeEl?.value || "unknown";
  }

  return toolTypeEl?.value || "racket";
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
    minTrackingConfidence: 0.45,
    outputSegmentationMasks: true
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

    if (extensionModeEl?.value === "auto") {
      await Promise.all([
        initHandLandmarker(),
        initObjectDetector()
      ]);
    }

    await openCamera({ facing: facingMode });

    running = true;
    stopBtn.disabled = false;
    if (fullscreenBtn) fullscreenBtn.disabled = false;
    if (stageFlipBtn) stageFlipBtn.hidden = !isFullscreenActive();
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
  document.body.classList.remove("analysis-fullscreen");
  if (document.fullscreenElement && document.exitFullscreen) {
    document.exitFullscreen().catch(() => {});
  } else if (document.webkitFullscreenElement && document.webkitExitFullscreen) {
    try { document.webkitExitFullscreen(); } catch (_) {}
  }
  if (fullscreenBtn) fullscreenBtn.disabled = true;
  if (analysisHud) analysisHud.hidden = true;
  if (exitFullscreenBtn) exitFullscreenBtn.hidden = true;
  if (stageFlipBtn) stageFlipBtn.hidden = true;
  if (animationId) cancelAnimationFrame(animationId);
  animationId = null;

  stopTracks();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  resetMetrics();

  latestObjects = [];
  currentHoldCandidate = null;
  confirmedHeldObject = null;
  holdConfirmFrames = 0;
  holdMissFrames = 0;
  latestPoseLandmarks = null;
  latestHands = [];
  lastHandDetectionAt = 0;
  manualObjectBox = null;
  endManualBoxMode();
  if (objectDetectionStatusEl) {
    objectDetectionStatusEl.textContent =
      extensionModeEl?.value === "auto" ? "未偵測持物" :
      extensionModeEl?.value === "manual" ? "手動模式" : "自動偵測已關閉";
  }

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



function canvasPointFromPointer(event) {
  const rect = canvas.getBoundingClientRect();
  let x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);

  // Canvas is visually mirrored for front camera, so invert pointer X back
  // into the intrinsic canvas coordinate system.
  if ((canvas.style.transform || "").includes("scaleX(-1)")) {
    x = canvas.width - x;
  }

  return {
    x: clamp(x, 0, canvas.width),
    y: clamp(y, 0, canvas.height)
  };
}

function normalizedManualBox(a, b) {
  if (!a || !b) return null;
  const x1 = Math.min(a.x, b.x);
  const y1 = Math.min(a.y, b.y);
  const x2 = Math.max(a.x, b.x);
  const y2 = Math.max(a.y, b.y);

  if (x2 - x1 < 16 || y2 - y1 < 16) return null;

  return {
    originX: x1,
    originY: y1,
    width: x2 - x1,
    height: y2 - y1
  };
}

function manualDetectionFromBox() {
  if (!manualObjectBox) return null;
  return {
    boundingBox: { ...manualObjectBox },
    categories: [{
      categoryName: "manual object",
      displayName: "manual object",
      score: 1
    }],
    __manual: true
  };
}

function nearestHandToBox(lm, box) {
  if (!lm || !box) return null;

  const candidates = [];
  const lw = wristPixels(lm, "left");
  const rw = wristPixels(lm, "right");

  for (const item of [
    { hand: "left", wrist: lw },
    { hand: "right", wrist: rw }
  ]) {
    if (!item.wrist) continue;
    const d = pointToBoxDistance(item.wrist, box, 0);
    candidates.push({ ...item, distance: d });
  }

  candidates.sort((a, b) => a.distance - b.distance);
  return candidates[0] || null;
}

function manualHeldObject(lm, b) {
  if (!manualObjectBox || !lm || !b) return null;

  const nearest = nearestHandToBox(lm, manualObjectBox);
  if (!nearest) return null;

  const bodyWidth =
    visible(lm, 11) && visible(lm, 12)
      ? Math.hypot(
          (lm[11].x - lm[12].x) * canvas.width,
          (lm[11].y - lm[12].y) * canvas.height
        )
      : Math.max(60, b.w * 0.4);

  const mappedHands = assignDetectedHandsToPose(lm);
  let best = null;

  for (const side of ["left", "right"]) {
    const handPts = mappedHands[side];
    if (!handPts) continue;

    const f = handObjectContactFeatures(handPts, manualObjectBox, bodyWidth);
    if (!f) continue;

    if (!best || f.score > best.contactScore) {
      best = {
        hand: side,
        distance: f.minTipDistance,
        contactScore: f.score,
        fingertipHits: f.fingertipHits
      };
    }
  }

  if (best) {
    return {
      label: "manual object",
      confidence: 1,
      hand: best.hand,
      distance: best.distance,
      contactScore: best.contactScore,
      fingertipHits: best.fingertipHits,
      detection: manualDetectionFromBox(),
      rank: 999,
      validGrip: best.contactScore >= HOLDING_SCORE_THRESHOLD,
      manual: true,
      contactBased: true
    };
  }

  // Pose-wrist fallback if the hand model temporarily loses the hand.
  const allowed = Math.max(35, bodyWidth * 0.22);
  return {
    label: "manual object",
    confidence: 1,
    hand: nearest.hand,
    distance: nearest.distance,
    contactScore: Math.max(0, 0.4 - nearest.distance / Math.max(1, bodyWidth)),
    fingertipHits: 0,
    detection: manualDetectionFromBox(),
    rank: 999,
    validGrip: nearest.distance <= allowed,
    manual: true,
    contactBased: false
  };
}

function updateManualBoxStatus() {
  if (!manualBoxStatusEl) return;

  if (manualBoxDrawing) {
    manualBoxStatusEl.textContent = "拖曳框選中…";
  } else if (manualObjectBox) {
    manualBoxStatusEl.textContent = "已框選；靠近手腕時視為持物";
  } else {
    manualBoxStatusEl.textContent = "未框選";
  }
}

function beginManualBoxMode() {
  manualBoxDrawing = true;
  manualBoxStart = null;
  manualBoxCurrent = null;
  canvas.style.pointerEvents = "auto";
  canvas.style.cursor = "crosshair";
  if (manualBoxBtn) manualBoxBtn.textContent = "請在畫面拖曳框選";
  updateManualBoxStatus();
}

function endManualBoxMode() {
  manualBoxDrawing = false;
  manualBoxStart = null;
  manualBoxCurrent = null;
  canvas.style.pointerEvents = "none";
  canvas.style.cursor = "";
  if (manualBoxBtn) manualBoxBtn.textContent = "手動框選持物";
  updateManualBoxStatus();
}

function clearManualObjectBox() {
  manualObjectBox = null;
  endManualBoxMode();
  if (manualBoxStatusEl) manualBoxStatusEl.textContent = "未框選";
}

function sameDetection(a, b) {
  if (!a || !b) return false;
  const ab = a.boundingBox;
  const bb = b.boundingBox;
  if (!ab || !bb) return false;

  const acx = ab.originX + ab.width / 2;
  const acy = ab.originY + ab.height / 2;
  const bcx = bb.originX + bb.width / 2;
  const bcy = bb.originY + bb.height / 2;

  const centerDistance = Math.hypot(acx - bcx, acy - bcy);
  const sizeRef = Math.max(20, Math.min(ab.width + ab.height, bb.width + bb.height) / 2);

  return centerDistance < sizeRef * 0.18;
}

function closestPointOnBox(p, box) {
  if (!p || !box) return null;

  return {
    x: clamp(p.x, box.originX, box.originX + box.width),
    y: clamp(p.y, box.originY, box.originY + box.height)
  };
}

function drawHeldObjectBoxes(lm) {
  const detectionsToDraw = [...(latestObjects || [])];

  const manualDet = manualDetectionFromBox();
  if (manualDet) detectionsToDraw.push(manualDet);

  // Hysteresis can keep a held object confirmed for several missed detector
  // cycles. Keep drawing its last known box so the UI never says "held"
  // without showing where the detected object was.
  if (
    confirmedHeldObject?.detection &&
    !confirmedHeldObject?.gripProxy &&
    !detectionsToDraw.some(d => sameDetection(d, confirmedHeldObject.detection))
  ) {
    detectionsToDraw.push(confirmedHeldObject.detection);
  }

  if (!detectionsToDraw.length) return;

  const leftWrist = lm ? wristPixels(lm, "left") : null;
  const rightWrist = lm ? wristPixels(lm, "right") : null;

  ctx.save();
  ctx.textBaseline = "bottom";
  ctx.font = `700 ${Math.max(14, Math.round(canvas.width / 65))}px system-ui, sans-serif`;

  for (const det of detectionsToDraw) {
    const cat = categoryOfDetection(det);
    const box = det?.boundingBox;

    if (!cat || !box) continue;
    const isManual = !!det.__manual;

    const isConfirmed =
      confirmedHeldObject?.detection &&
      sameDetection(det, confirmedHeldObject.detection);

    const isCandidate =
      !isConfirmed &&
      currentHoldCandidate?.detection &&
      sameDetection(det, currentHoldCandidate.detection);

    const isKnownTool = EXTENDABLE_OBJECT_LABELS.has(cat.name);

    // Keep normal screen uncluttered: known tools are always shown; unknown
    // categories are shown only when contact logic says they are a held
    // candidate/confirmed object.
    if (!isManual && !isKnownTool && !isCandidate && !isConfirmed) continue;

    let stroke = "rgba(80, 205, 255, 0.95)";
    let fill = "rgba(80, 205, 255, 0.10)";
    let prefix = "物件";

    if (isManual) {
      const b = bodyBounds(lm);
      const held = b ? manualHeldObject(lm, b) : null;
      stroke = held?.validGrip
        ? "rgba(70, 245, 145, 1)"
        : "rgba(255, 170, 70, 1)";
      fill = held?.validGrip
        ? "rgba(70, 245, 145, 0.12)"
        : "rgba(255, 170, 70, 0.10)";
      prefix = held?.validGrip ? "手動持物" : "手動框選";
    }

    if (isCandidate) {
      stroke = "rgba(255, 205, 65, 0.98)";
      fill = "rgba(255, 205, 65, 0.12)";
      prefix = "持物候選";
    }

    if (isConfirmed) {
      stroke = "rgba(70, 245, 145, 1)";
      fill = "rgba(70, 245, 145, 0.14)";
      prefix = "已持物";
    }

    const x = clamp(box.originX, 0, canvas.width);
    const y = clamp(box.originY, 0, canvas.height);
    const w = clamp(box.width, 0, canvas.width - x);
    const h = clamp(box.height, 0, canvas.height - y);

    ctx.fillStyle = fill;
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = stroke;
    ctx.lineWidth = Math.max(3, canvas.width / 320);
    ctx.setLineDash(isCandidate ? [10, 7] : []);
    ctx.strokeRect(x, y, w, h);

    const pct = Math.round(cat.score * 100);
    const label = isManual
      ? `${prefix}：${toolTypeEl?.selectedOptions?.[0]?.textContent || "未知工具"}`
      : `${prefix}：${isKnownTool ? cat.name : `未知物件(${cat.name})`} ${pct}%`;

    const metrics = ctx.measureText(label);
    const padX = 7;
    const padY = 5;
    const labelH = Math.max(24, Math.round(canvas.width / 42));
    const labelW = metrics.width + padX * 2;
    const labelX = clamp(x, 2, canvas.width - labelW - 2);
    const labelY = clamp(y - labelH, 2, canvas.height - labelH - 2);

    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(10, 10, 10, 0.78)";
    ctx.fillRect(labelX, labelY, labelW, labelH);

    ctx.strokeStyle = stroke;
    ctx.lineWidth = 2;
    ctx.strokeRect(labelX, labelY, labelW, labelH);

    ctx.fillStyle = stroke;
    ctx.fillText(label, labelX + padX, labelY + labelH - padY);

    // For candidate/confirmed holding, draw association from wrist to object.
    if (isConfirmed || isCandidate) {
      const hand = isConfirmed
        ? confirmedHeldObject.hand
        : currentHoldCandidate?.hand;

      const wrist = hand === "left" ? leftWrist : rightWrist;
      const target = closestPointOnBox(wrist, box);

      if (wrist && target) {
        ctx.beginPath();
        ctx.moveTo(wrist.x, wrist.y);
        ctx.lineTo(target.x, target.y);
        ctx.strokeStyle = stroke;
        ctx.lineWidth = Math.max(2, canvas.width / 500);
        ctx.setLineDash([7, 5]);
        ctx.stroke();

        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(target.x, target.y, Math.max(4, canvas.width / 180), 0, Math.PI * 2);
        ctx.fillStyle = stroke;
        ctx.fill();
      }
    }
  }

  ctx.restore();
}

function drawResults(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  latestPoseLandmarks = result?.landmarks?.[0] || null;

  if (!result?.landmarks?.length) {
    resetMetrics();
    return;
  }

  const drawingUtils = new DrawingUtils(ctx);

  // Draw space classification first, then the pose skeleton on top.
  if (result.landmarks[0]) {
    drawSpaceZones(result.landmarks[0], result);
  }

  // V10: show detected held-object bounding boxes on top of the video.
  drawHeldObjectBoxes(result.landmarks[0]);

  for (const lm of result.landmarks) {
    drawingUtils.drawConnectors(
      lm,
      PoseLandmarker.POSE_CONNECTIONS,
      { color: "#ffffff", lineWidth: 3 }
    );
    drawingUtils.drawLandmarks(
      lm,
      { color: "#ffffff", fillColor: "#111111", radius: 4, lineWidth: 2 }
    );
  }

  if (manualBoxDrawing && manualBoxStart && manualBoxCurrent) {
    const preview = normalizedManualBox(manualBoxStart, manualBoxCurrent);
    if (preview) {
      ctx.save();
      ctx.strokeStyle = "rgba(255, 170, 70, 1)";
      ctx.fillStyle = "rgba(255, 170, 70, 0.08)";
      ctx.lineWidth = Math.max(2, canvas.width / 400);
      ctx.setLineDash([10, 7]);
      ctx.fillRect(preview.originX, preview.originY, preview.width, preview.height);
      ctx.strokeRect(preview.originX, preview.originY, preview.width, preview.height);
      ctx.restore();
    }
  }

  showAnalysis(result.landmarks[0]);
  updateAnalysisHud();
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

    if (result?.landmarks?.[0]) {
      const b = bodyBounds(result.landmarks[0]);
      if (b) {
        detectHandsIfNeeded(now);
        detectObjectsIfNeeded(now, result.landmarks[0], b);
      }
    }

    drawResults(result);
  } catch (err) {
    console.error("Inference error", err);
    setStatus("分析暫停", friendlyError(err));
  }
}


if (ppsScaleEl && ppsScaleValueEl) {
  const updatePpsScaleLabel = () => {
    ppsScaleValueEl.textContent = `${Number(ppsScaleEl.value).toFixed(2)} × 身寬`;
    updateAnalysisHud();
  };
  ppsScaleEl.addEventListener("input", updatePpsScaleLabel);
  updatePpsScaleLabel();
}



function updateExtensionMode() {
  const mode = extensionModeEl?.value || "auto";

  confirmedHeldObject = null;
  currentHoldCandidate = null;
  holdConfirmFrames = 0;
  holdMissFrames = 0;

  if (mode === "off") {
    if (extensionStatusEl) extensionStatusEl.textContent = "關閉";
    if (objectDetectionStatusEl) objectDetectionStatusEl.textContent = "自動偵測已關閉";
  } else if (mode === "manual") {
    if (extensionStatusEl) extensionStatusEl.textContent = "手動開啟";
    if (objectDetectionStatusEl) objectDetectionStatusEl.textContent = "手動模式";
  } else {
    if (extensionStatusEl) extensionStatusEl.textContent = "等待持物";
    if (objectDetectionStatusEl) objectDetectionStatusEl.textContent =
      objectDetector ? "未偵測持物" : "等待載入物件模型";

    if (running) {
      if (!handLandmarker) initHandLandmarker();
      if (!objectDetector) initObjectDetector();
    }
  }
}

function applyToolPreset() {
  const preset = TOOL_PRESETS[toolTypeEl?.value || "racket"] || TOOL_PRESETS.custom;
  if (toolTypeEl?.value !== "custom" && extensionScaleEl) {
    extensionScaleEl.value = String(preset.scale);
  }
  updateExtensionScaleLabel();
}

function updateExtensionScaleLabel() {
  if (extensionScaleEl && extensionScaleValueEl) {
    extensionScaleValueEl.textContent =
      `${Number(extensionScaleEl.value).toFixed(2)} × 身高`;
  }
}

extensionModeEl?.addEventListener("change", updateExtensionMode);
toolTypeEl?.addEventListener("change", applyToolPreset);
toolHandEl?.addEventListener("change", () => {
  if (extensionStatusEl && extensionModeEl?.value === "manual") {
    extensionStatusEl.textContent = "手動更新中";
  }
});
extensionScaleEl?.addEventListener("input", updateExtensionScaleLabel);

updateExtensionMode();
applyToolPreset();


manualBoxBtn?.addEventListener("click", beginManualBoxMode);
clearManualBoxBtn?.addEventListener("click", clearManualObjectBox);

canvas.addEventListener("pointerdown", (event) => {
  if (!manualBoxDrawing) return;
  event.preventDefault();
  canvas.setPointerCapture?.(event.pointerId);
  manualBoxStart = canvasPointFromPointer(event);
  manualBoxCurrent = manualBoxStart;
});

canvas.addEventListener("pointermove", (event) => {
  if (!manualBoxDrawing || !manualBoxStart) return;
  event.preventDefault();
  manualBoxCurrent = canvasPointFromPointer(event);
});

canvas.addEventListener("pointerup", (event) => {
  if (!manualBoxDrawing || !manualBoxStart) return;
  event.preventDefault();

  const end = canvasPointFromPointer(event);
  const box = normalizedManualBox(manualBoxStart, end);

  if (box) {
    manualObjectBox = box;
    if (manualBoxStatusEl) {
      manualBoxStatusEl.textContent = "已框選；等待手腕靠近物件";
    }
  }

  endManualBoxMode();
});

canvas.addEventListener("pointercancel", () => {
  if (manualBoxDrawing) endManualBoxMode();
});

fullscreenBtn?.addEventListener("click", toggleImmersiveAnalysis);
exitFullscreenBtn?.addEventListener("click", exitImmersiveAnalysis);
stageFlipBtn?.addEventListener("click", switchCameraByFacing);

document.addEventListener("fullscreenchange", syncFullscreenUi);
document.addEventListener("webkitfullscreenchange", syncFullscreenUi);

window.addEventListener("orientationchange", () => {
  // Let mobile browser settle its viewport before recalculating overlays.
  setTimeout(syncFullscreenUi, 120);
});

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

setStatus("等待啟動", "V12：自動偵測 + 手動框選 fallback；支援未收錄於 COCO 的桌球拍等工具。");
