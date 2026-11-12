// 전체 흐름: 세션 정보 입력 → 카메라 · 모델 준비 → 프레임마다 인식 · 제스처 · 그리기
import { askSessionInfo, describe, HAND_KO } from './setup-form.js';
import { startCamera } from './camera.js';
import { createHandTracker } from './hand-tracker.js';
import { extractHands, renderTable, saveJson } from './landmarks.js';
import { createOverlay } from './draw.js';
import { FpsCounter } from './fps.js';
import { setupPip } from './pip.js';
import { GestureMouse } from './gestures.js';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const stateEl = document.getElementById('state');
const pauseBtn = document.getElementById('pause');
const warnEl = document.getElementById('warn');
const rowsEl = document.getElementById('rows');

// 입력 폼을 받는 동안 모델을 미리 불러온다
const trackerReady = createHandTracker();

const meta = await askSessionInfo();
const who = describe(meta);
document.title = who;
document.getElementById('who').textContent = who;

const LABELS = {
  none: `✋ ${HAND_KO[meta.hand]}손을 보여주세요`,
  move: '👆 이동',
  click: '🤏 클릭',
  drag: '✊ 드래그',
  scroll: '✌️ 스크롤',
  grab: '🖐 창 이동',
  fist: '✊ 주먹 (유지하면 일시정지)',
  paused: '⏸ 일시정지 — 주먹 1초 또는 ⌃⌥P',
};
const label = (status) =>
  status.mode === 'grab' && !status.grabOk ? '🖐 옮길 창이 없어요' : LABELS[status.mode] ?? status.mode;

// --- 커서 · 창 제어 (메인 프로세스로 전달) ---
const gesture = new GestureMouse({
  move: (x, y, vx = 0, vy = 0) => window.handMouse.move(x, y, vx, vy),
  stop: () => window.handMouse.stop(),
  left: (down) => window.handMouse.left(down),
  right: () => window.handMouse.rightClick(),
  scroll: (dy) => window.handMouse.scroll(dy),
  windowGrab: (x, y) => window.handMouse.windowGrab(x, y),
  windowMove: (x, y) => window.handMouse.windowMove(x, y),
  windowRelease: () => window.handMouse.windowRelease(),
});

function togglePause() {
  gesture.togglePause();
  setPausedUI();
}
function setPausedUI() {
  pauseBtn.textContent = gesture.paused ? '다시 시작' : '일시정지';
}
pauseBtn.addEventListener('click', togglePause);
window.handMouse.onTogglePause(togglePause);
window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !gesture.paused) togglePause();
});

warnEl.hidden = await window.handMouse.isAccessibilityTrusted();

// --- 카메라 & 모델 ---
const size = await startCamera(video);
canvas.width = size.width;
canvas.height = size.height;
stateEl.textContent = '모델 로딩 중…';
const tracker = await trackerReady;

const drawOverlay = createOverlay(canvas);
const fps = new FpsCounter(document.getElementById('fps'));
const renderPip = setupPip({
  video,
  overlay: canvas,
  button: document.getElementById('pip'),
  pipVideo: document.getElementById('pip-video'),
  label: who,
});

let latest = []; // 마지막 프레임에서 인식한 손들 (최대 2개)
document.getElementById('save').addEventListener('click', () =>
  saveJson(meta, latest.filter((h) => h.handedness === meta.hand)));

// 카메라 프레임 하나당 정확히 한 번만 인식한다.
// (웹캠 스트림은 새 프레임이 없어도 video.currentTime 이 계속 늘어서, 그걸로 거르면 같은 프레임을 여러 번
//  처리하게 되고 손 속도가 0 ↔ 급증을 반복해 커서가 끊겨 보였다)
let tableAt = 0;
let lastDetectTs = 0;
function processFrame(frameTime) {
  // detectForVideo 는 타임스탬프가 계속 증가해야 함
  const now = Math.max(performance.now(), lastDetectTs + 1);
  lastDetectTs = now;
  const result = tracker.detectForVideo(video, now);
  fps.tick(now);

  latest = extractHands(result, frameTime);
  // 커서와 창은 선택한 손으로만 제어. 속도 계산이 정확하도록 프레임이 찍힌 시각을 쓴다
  const hand = latest.find((h) => h.handedness === meta.hand);
  const status = gesture.update(
    { landmarks: hand ? [hand.landmarks] : [], worldLandmarks: hand ? [hand.worldLandmarks] : [] },
    frameTime / 1000,
  );
  for (const h of latest) h.mode = h === hand ? status.mode : null;

  stateEl.textContent = label(status);
  setPausedUI();
  drawOverlay(latest, hand, status);
  renderPip(stateEl.textContent, status.mode, fps.el.textContent);
  // 표는 사람이 읽을 용도라 10Hz 로만 갱신
  if (now - tableAt > 100) {
    tableAt = now;
    renderTable(rowsEl, latest.filter((h) => h === hand));
  }
}

let lastFrame = -1;
let lastFrameAt = 0;
let lastFrameTime = 0;
function onVideoFrame(_now, frame) {
  if (frame.presentedFrames !== lastFrame) {
    lastFrame = frame.presentedFrames;
    lastFrameAt = performance.now();
    // captureTime: 카메라가 프레임을 찍은 시각 (없으면 화면에 나온 시각)
    lastFrameTime = Math.max(frame.captureTime ?? frame.expectedDisplayTime, lastFrameTime + 1);
    processFrame(lastFrameTime);
  }
  video.requestVideoFrameCallback(onVideoFrame);
}
video.requestVideoFrameCallback(onVideoFrame);

// 창을 최소화하면 프레임 콜백이 멈출 수 있어서, 그때만 타이머로 대신 돈다
let lastVideoTime = -1;
setInterval(() => {
  if (performance.now() - lastFrameAt < 200 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  lastFrameTime = Math.max(performance.now(), lastFrameTime + 1);
  processFrame(lastFrameTime);
}, 33);
