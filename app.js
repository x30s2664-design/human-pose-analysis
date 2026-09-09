import {
  FilesetResolver,
  PoseLandmarker,
  ObjectDetector,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm";

const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const OBJECT_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite";

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



let poseLandmarker = null;
let objectDetector = null;
let objectDetectorLoading = false;
let lastObjectDetectionAt = 0;
let latestObjects = [];
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

// Mobile/tablet: limiting inference rate reduces heat and browser stalls.
const MIN_INFERENCE_INTERVAL_MS = 85;
const OBJECT_INFERENCE_INTERVAL_MS = 450;

// Hysteresis: avoid extension flicker.
// Detection must be confirmed across multiple object-detection cycles.
const HOLD_CONFIRM_CYCLES = 2;
const HOLD_RELEASE_CYCLES = 4;

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
  racket: { scale: 0.42, width: 0.11, label: "球拍" },
  pen:    { scale: 0.12, width: 0.025, label: "筆" },
  phone:  { scale: 0.16, width: 0.055, label: "手機" },
  tool:   { scale: 0.28, width: 0.055, label: "工具" },
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

function drawExtensionZone(lm, b) {
  if (!extensionShouldBeActive()) {
    if (extensionStatusEl) {
      extensionStatusEl.textContent =
        extensionModeEl?.value === "off" ? "關閉" : "未偵測持物";
    }
    return;
  }

  const hand = resolvedExtensionHand(lm, b);
  if (!hand) {
    if (extensionStatusEl) extensionStatusEl.textContent = "手部未偵測";
    return;
  }

  const vector = toolVectorFromPose(lm, hand);
  if (!vector) {
    if (extensionStatusEl) extensionStatusEl.textContent = "等待手臂";
    return;
  }

  const toolType = resolvedToolType();
  const preset = TOOL_PRESETS[toolType] || TOOL_PRESETS.custom;
  const scale = parseFloat(extensionScaleEl?.value || String(preset.scale));

  // Tool-use extension is estimated from body height so it remains responsive
  // on phone/tablet cameras without requiring metric calibration.
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
  ctx.fillStyle = "rgba(65, 235, 145, 0.13)";
  ctx.fill();

  ctx.lineWidth = Math.max(2, canvas.width / 480);
  ctx.strokeStyle = "rgba(75, 240, 150, 0.98)";
  ctx.setLineDash([11, 8]);
  ctx.stroke();

  // Tool axis and endpoint
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(vector.wrist.x, vector.wrist.y);
  ctx.lineTo(end.x, end.y);
  ctx.strokeStyle = "rgba(120, 255, 180, 0.92)";
  ctx.lineWidth = Math.max(2, canvas.width / 650);
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(end.x, end.y, Math.max(5, radius * 0.18), 0, Math.PI * 2);
  ctx.fillStyle = "rgba(120, 255, 180, 1)";
  ctx.fill();

  const fontSize = Math.max(15, Math.round(canvas.width / 58));
  ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "rgba(120, 255, 180, 0.98)";

  const labelX = clamp(end.x + 10, 8, canvas.width - fontSize * 6);
  const labelY = clamp(end.y - 8, fontSize + 8, canvas.height - 8);
  ctx.fillText(`展延 ${preset.label}`, labelX, labelY);

  ctx.restore();

  if (extensionStatusEl) {
    extensionStatusEl.textContent =
      `${hand === "left" ? "左手" : "右手"}・${preset.label}`;
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

function drawSpaceZones(lm) {
  const b = bodyBounds(lm);
  if (!b) return;

  const refW = bodyReferenceWidth(lm, b);
  const scale = parseFloat(ppsScaleEl?.value || "0.18");

  // Use shoulder/body WIDTH. 0.20 gives a compact envelope around the body.
  const ppsMargin = Math.max(10, refW * scale);

  // Far space is only a subtle background cue.
  ctx.save();
  ctx.fillStyle = "rgba(65, 120, 255, 0.025)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();

  const ppsMask = buildBodyMask(lm, b, ppsMargin);
  const bodyMask = buildBodyMask(lm, b, 0);

  paintMaskColor(ppsMask, "rgba(255, 214, 70, 0.16)");
  // PPS comes from the reconstructed body envelope.
  // The visible body itself is rendered as articulated mannequin blocks.
  drawMannequinBody(lm, b);

  const fontSize = Math.max(13, Math.round(canvas.width / 70));
  ctx.save();
  ctx.font = `700 ${fontSize}px system-ui, sans-serif`;
  ctx.textBaseline = "top";

  ctx.fillStyle = "rgba(255, 110, 110, 0.96)";
  ctx.fillText("本體", clamp(b.maxX - 48, 8, canvas.width - 60),
               clamp(b.minY + b.h * 0.52, 8, canvas.height - 26));

  ctx.fillStyle = "rgba(255, 230, 110, 0.96)";
  ctx.fillText("近體 PPS", clamp(b.maxX - 85, 8, canvas.width - 105),
               clamp(b.minY - ppsMargin * 0.45, 8, canvas.height - 26));

  ctx.fillStyle = "rgba(120, 175, 255, 0.92)";
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
      scoreThreshold: 0.25,
      maxResults: 8
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
  const cat = det?.categories?.[0];
  if (!cat) return null;
  return {
    name: normalizeCategoryName(cat.categoryName || cat.displayName),
    score: Number(cat.score || 0)
  };
}

function findHeldObjectNearHands(lm, b, detections) {
  if (!lm || !b || !detections?.length) return null;

  const wrists = [];
  const lw = wristPixels(lm, "left");
  const rw = wristPixels(lm, "right");
  if (lw) wrists.push({ hand: "left", ...lw });
  if (rw) wrists.push({ hand: "right", ...rw });
  if (!wrists.length) return null;

  const bodyWidth =
    visible(lm, 11) && visible(lm, 12)
      ? Math.hypot(
          (lm[11].x - lm[12].x) * canvas.width,
          (lm[11].y - lm[12].y) * canvas.height
        )
      : Math.max(60, b.w * 0.4);

  let best = null;

  for (const det of detections) {
    const cat = categoryOfDetection(det);
    if (!cat || !EXTENDABLE_OBJECT_LABELS.has(cat.name)) continue;

    for (const wrist of wrists) {
      const relation = wristTouchesObject(wrist, det, bodyWidth);
      if (!relation?.touches) continue;

      // Prefer high confidence and closer wrist/object contact.
      const rank = cat.score * 3 - relation.distance / Math.max(1, bodyWidth);

      if (!best || rank > best.rank) {
        best = {
          label: cat.name,
          confidence: cat.score,
          hand: wrist.hand,
          distance: relation.distance,
          detection: det,
          rank
        };
      }
    }
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

    if (holdConfirmFrames >= HOLD_CONFIRM_CYCLES) {
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
      objectDetectionStatusEl.textContent =
        `已確認：${confirmedHeldObject.label}・${confirmedHeldObject.hand === "left" ? "左手" : "右手"}・${pct}%`;
    } else if (candidate) {
      objectDetectionStatusEl.textContent =
        `確認中：${candidate.label}`;
    } else {
      objectDetectionStatusEl.textContent = "未偵測持物";
    }
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
    const candidate = findHeldObjectNearHands(lm, b, latestObjects);
    updateHoldState(candidate);
  } catch (err) {
    console.warn("Object detection failed", err);
  }
}

function extensionShouldBeActive() {
  const mode = extensionModeEl?.value || "auto";
  if (mode === "off") return false;
  if (mode === "manual") return true;
  return !!confirmedHeldObject;
}

function resolvedExtensionHand(lm, b) {
  const mode = extensionModeEl?.value || "auto";

  if (mode === "auto" && confirmedHeldObject?.hand) {
    return confirmedHeldObject.hand;
  }

  return chooseToolHand(lm, b);
}

function resolvedToolType() {
  const mode = extensionModeEl?.value || "auto";

  if (mode === "auto" && confirmedHeldObject) {
    if (confirmedHeldObject.label === "tennis racket") return "racket";
    if (confirmedHeldObject.label === "cell phone") return "phone";
    if (confirmedHeldObject.label === "baseball bat") return "tool";
    return "tool";
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

    if (extensionModeEl?.value === "auto") {
      await initObjectDetector();
    }

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

  latestObjects = [];
  currentHoldCandidate = null;
  confirmedHeldObject = null;
  holdConfirmFrames = 0;
  holdMissFrames = 0;
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

function drawResults(result) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!result?.landmarks?.length) {
    resetMetrics();
    return;
  }

  const drawingUtils = new DrawingUtils(ctx);

  // Draw space classification first, then the pose skeleton on top.
  if (result.landmarks[0]) {
    drawSpaceZones(result.landmarks[0]);
  }

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

    if (result?.landmarks?.[0]) {
      const b = bodyBounds(result.landmarks[0]);
      if (b) {
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

    if (running && !objectDetector) {
      initObjectDetector();
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

setStatus("等待啟動", "V8：本體輪廓展延；自動判斷手機／球拍等持物。");
