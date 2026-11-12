const {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  net,
  powerSaveBlocker,
  protocol,
  screen,
  session,
  systemPreferences,
} = require('electron');
const path = require('path');
const { pathToFileURL } = require('url');
const mouse = require('./mouse');
const windowMover = require('./window-mover');

const ROOT = path.join(__dirname, '..', '..');
// app:// 로 내보내는 폴더 (렌더러 코드, MediaPipe 라이브러리/wasm, 모델)
const SERVED = ['src', 'node_modules/@mediapipe/tasks-vision', 'Models'].map((p) => path.join(ROOT, p));

// file:// 에서는 wasm·모델 fetch 가 막힐 수 있어 전용 프로토콜로 로컬 파일을 제공
protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } },
]);

// 창이 다른 앱 뒤에 있어도 카메라 인식이 멈추지 않도록
app.commandLine.appendSwitch('disable-renderer-backgrounding');
app.commandLine.appendSwitch('disable-background-timer-throttling');
app.commandLine.appendSwitch('disable-backgrounding-occluded-windows');

const PAUSE_SHORTCUT = 'Control+Alt+P';
const TICK_MS = 8;        // 커서 갱신 주기 (~120Hz)
const FOLLOW = 0.6;       // 매 틱마다 목표 위치로 다가가는 비율 (클수록 빠름)
const PREDICT_MAX = 0.04; // 다음 카메라 프레임이 올 때까지 손 속도로 앞질러 예측하는 최대 시간 (초)

let win = null;
let target = null;        // 손이 가리키는 화면 좌표 (point 단위)
let velocity = { x: 0, y: 0 }; // point/초
let targetAt = 0;
let cursor = null;        // 우리가 마지막으로 보낸 커서 위치
let grabHand = null;      // 창을 잡은 순간의 손 위치 (정규화 좌표)

// 카메라 인식은 ~30fps 라서, 프레임 사이에는 손 속도로 위치를 예측해 지연을 줄인다
setInterval(() => {
  if (!target) return;
  cursor ??= mouse.getPosition();
  const ahead = Math.min((performance.now() - targetAt) / 1000, PREDICT_MAX);
  const dx = target.x + velocity.x * ahead - cursor.x;
  const dy = target.y + velocity.y * ahead - cursor.y;
  if (Math.abs(dx) < 0.2 && Math.abs(dy) < 0.2) return;
  cursor = { x: cursor.x + dx * FOLLOW, y: cursor.y + dy * FOLLOW };
  mouse.move(cursor);
}, TICK_MS);

function currentPos() {
  return cursor ?? mouse.getPosition();
}

ipcMain.on('mouse:move', (_e, nx, ny, vx = 0, vy = 0) => {
  // 정규화 좌표(0~1)를 메인 디스플레이 좌표로
  const { x, y, width, height } = screen.getPrimaryDisplay().bounds;
  target = { x: x + nx * (width - 1), y: y + ny * (height - 1) };
  velocity = { x: vx * width, y: vy * height };
  targetAt = performance.now();
});

ipcMain.on('mouse:stop', () => {
  target = null;
  mouse.leftUp(currentPos());
  cursor = null; // 다음에 다시 시작할 때 실제 커서 위치부터
});

ipcMain.on('mouse:left', (_e, down) => {
  // 버튼 이벤트는 보간 중인 위치가 아니라 목표 위치에서 바로
  if (target) {
    cursor = { ...target };
    mouse.move(cursor);
  }
  if (down) mouse.leftDown(currentPos());
  else mouse.leftUp(currentPos());
});

ipcMain.on('mouse:right', () => mouse.rightClick(currentPos()));

ipcMain.on('mouse:scroll', (_e, dy) => mouse.scroll(dy));

// --- 맨 앞 창 잡아서 옮기기 ---
ipcMain.handle('window:grab', (_e, nx, ny) => {
  grabHand = windowMover.grab() ? { nx, ny } : null;
  return grabHand !== null;
});

ipcMain.on('window:move', (_e, nx, ny) => {
  if (!grabHand) return;
  // 손이 움직인 거리(정규화)를 화면 크기만큼 늘려서 창을 같은 비율로 이동
  const { width, height } = screen.getPrimaryDisplay().bounds;
  windowMover.moveBy((nx - grabHand.nx) * width, (ny - grabHand.ny) * height);
});

ipcMain.on('window:release', () => {
  grabHand = null;
  windowMover.release();
});

ipcMain.handle('accessibility', () => systemPreferences.isTrustedAccessibilityClient(false));

function serveLocalFiles() {
  protocol.handle('app', (request) => {
    const file = path.normalize(path.join(ROOT, decodeURIComponent(new URL(request.url).pathname)));
    if (!SERVED.some((dir) => file.startsWith(dir + path.sep))) {
      return new Response('Not found', { status: 404 });
    }
    return net.fetch(pathToFileURL(file).toString());
  });
}

function createWindow() {
  const { workArea } = screen.getPrimaryDisplay();
  const width = 860;
  const height = 400;

  win = new BrowserWindow({
    width,
    height,
    x: workArea.x + workArea.width - width - 16,
    y: workArea.y + workArea.height - height - 16,
    title: 'Hand Window Mover',
    backgroundColor: '#111111',
    webPreferences: {
      preload: path.join(__dirname, '..', 'preload', 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });
  win.setAlwaysOnTop(true, 'floating');
  win.setVisibleOnAllWorkspaces(true);
  win.loadURL('app://local/src/renderer/index.html');
}

app.whenReady().then(async () => {
  // 다른 앱을 쓰는 동안 macOS App Nap 이 이 앱을 느리게 만들지 않도록
  powerSaveBlocker.start('prevent-app-suspension');

  await systemPreferences.askForMediaAccess('camera');
  // 손쉬운 사용 권한이 없으면 시스템 설정 안내 창을 띄움
  systemPreferences.isTrustedAccessibilityClient(true);

  session.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
    callback(permission === 'media');
  });

  serveLocalFiles();
  createWindow();

  globalShortcut.register(PAUSE_SHORTCUT, () => win?.webContents.send('toggle-pause'));
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  windowMover.release();
  mouse.leftUp(currentPos());
});

app.on('window-all-closed', () => app.quit());
