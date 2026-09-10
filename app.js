import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const cameraSelect = document.getElementById("cameraSelect");

const statusEl = document.getElementById("status");
const headEl = document.getElementById("head");
const shoulderEl = document.getElementById("shoulder");
const hipEl = document.getElementById("hip");
const leftKneeEl = document.getElementById("leftKnee");
const rightKneeEl = document.getElementById("rightKnee");
const supportEl = document.getElementById("support");

let poseLandmarker = null;
let stream = null;
let running = false;
let lastVideoTime = -1;

function deg(v) {
  return `${v.toFixed(1)}°`;
}

function angleLine(a, b) {
  return Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
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
  // MediaPipe Pose indices:
  // nose 0, left eye 2, right eye 5
  if (![0, 2, 5].every(i => visible(lm, i))) return "UNKNOWN";
  const nose = lm[0];
  const le = lm[2];
  const re = lm[5];
  const midX = (le.x + re.x) / 2;
  const span = Math.abs(le.x - re.x);
  if (span < 0.002) return "UNKNOWN";
  const dx = (nose.x - midX) / span;

  if (dx < -0.18) return "LEFT";
  if (dx > 0.18) return "RIGHT";
  return "FORWARD";
}

function supportLeg(lm) {
  // hips 23/24, ankles 27/28
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

  if ([11, 12].every(i => visible(lm, i))) {
    shoulderEl.textContent = deg(angleLine(lm[11], lm[12]));
  } else {
    shoulderEl.textContent = "—";
  }

  if ([23, 24].every(i => visible(lm, i))) {
    hipEl.textContent = deg(angleLine(lm[23], lm[24]));
  } else {
    hipEl.textContent = "—";
  }

  if ([23, 25, 27].every(i => visible(lm, i))) {
    leftKneeEl.textContent = deg(jointAngle(lm[23], lm[25], lm[27]));
  } else {
    leftKneeEl.textContent = "—";
  }

  if ([24, 26, 28].every(i => visible(lm, i))) {
    rightKneeEl.textContent = deg(jointAngle(lm[24], lm[26], lm[28]));
  } else {
    rightKneeEl.textContent = "—";
  }
}

async function createPoseLandmarker() {
  statusEl.textContent = "模型載入中…";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numPoses: 2,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  statusEl.textContent = "模型已就緒";
}

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter(d => d.kind === "videoinput");
    cameraSelect.innerHTML = "";
    cams.forEach((cam, index) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `相機 ${index + 1}`;
      cameraSelect.appendChild(opt);
    });
  } catch (e) {
    console.warn(e);
  }
}

async function startCamera() {
  if (!poseLandmarker) {
    statusEl.textContent = "模型尚未就緒";
    return;
  }

  stopCamera();

  const selected = cameraSelect.value;
  const constraints = {
    audio: false,
    video: selected
      ? { deviceId: { exact: selected }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } }
  };

  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();

    await listCameras();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    running = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusEl.textContent = "即時分析中";
    requestAnimationFrame(predict);
  } catch (e) {
    console.error(e);
    statusEl.textContent = `無法開啟相機：${e.name}`;
  }
}

function stopCamera() {
  running = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

async function predict() {
  if (!running) return;

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const nowMs = performance.now();

    const result = poseLandmarker.detectForVideo(video, nowMs);

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const drawingUtils = new DrawingUtils(ctx);

    if (result.landmarks?.length) {
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

      // Detailed metrics are shown for the first detected person.
      showAnalysis(result.landmarks[0]);
    } else {
      headEl.textContent = "—";
      shoulderEl.textContent = "—";
      hipEl.textContent = "—";
      leftKneeEl.textContent = "—";
      rightKneeEl.textContent = "—";
      supportEl.textContent = "—";
    }
  }

  requestAnimationFrame(predict);
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
cameraSelect.addEventListener("change", () => {
  if (running) startCamera();
});

if (!navigator.mediaDevices?.getUserMedia) {
  statusEl.textContent = "此瀏覽器不支援相機 API";
  startBtn.disabled = true;
} else {
  createPoseLandmarker()
    .then(listCameras)
    .catch(err => {
      console.error(err);
      statusEl.textContent = "模型載入失敗，請檢查網路或瀏覽器主控台";
    });
}
