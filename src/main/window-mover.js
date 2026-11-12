// macOS 손쉬운 사용(Accessibility) API 로 맨 앞 앱의 창 위치를 바꾼다 (koffi FFI)
const koffi = require('koffi');

const AX = koffi.load('/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices');
const CF = koffi.load('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation');
koffi.load('/System/Library/Frameworks/AppKit.framework/AppKit'); // NSWorkspace 클래스를 쓰려면 로드만 해두면 됨
const objc = koffi.load('/usr/lib/libobjc.A.dylib');

// mouse.js 의 CGPoint 와 이름이 겹치지 않게 익명 구조체로
const Point = koffi.struct({ x: 'double', y: 'double' });

const AXUIElementCreateSystemWide = AX.func('AXUIElementCreateSystemWide', 'void *', []);
const AXUIElementCreateApplication = AX.func('AXUIElementCreateApplication', 'void *', ['int32_t']);
const AXUIElementCopyAttributeValue = AX.func('AXUIElementCopyAttributeValue', 'int32_t', [
  'void *', 'void *', koffi.out(koffi.pointer('void *')),
]);
const AXUIElementSetAttributeValue = AX.func('AXUIElementSetAttributeValue', 'int32_t', ['void *', 'void *', 'void *']);
const AXValueCreate = AX.func('AXValueCreate', 'void *', ['uint32_t', koffi.pointer(Point)]);
const AXValueGetValue = AX.func('AXValueGetValue', 'bool', ['void *', 'uint32_t', koffi.out(koffi.pointer(Point))]);
const CFStringCreateWithCString = CF.func('void *CFStringCreateWithCString(void *alloc, const char *s, uint32_t encoding)');
const CFRelease = CF.func('void CFRelease(void *cf)');

// [[NSWorkspace sharedWorkspace] frontmostApplication].processIdentifier
const objc_getClass = objc.func('void *objc_getClass(const char *name)');
const sel = objc.func('void *sel_registerName(const char *name)');
const sendPtr = objc.func('objc_msgSend', 'void *', ['void *', 'void *']);
const sendInt = objc.func('objc_msgSend', 'int32_t', ['void *', 'void *']);
const NSWorkspace = objc_getClass('NSWorkspace');
const SEL_SHARED = sel('sharedWorkspace');
const SEL_FRONTMOST = sel('frontmostApplication');
const SEL_PID = sel('processIdentifier');

function frontmostPid() {
  const app = sendPtr(sendPtr(NSWorkspace, SEL_SHARED), SEL_FRONTMOST);
  return app ? sendInt(app, SEL_PID) : 0;
}

const UTF8 = 0x08000100;
const AX_VALUE_CGPOINT = 1;
const attr = (name) => CFStringCreateWithCString(null, name, UTF8); // 프로그램이 끝날 때까지 재사용
const FOCUSED_APP = attr('AXFocusedApplication');
const FOCUSED_WINDOW = attr('AXFocusedWindow');
const MAIN_WINDOW = attr('AXMainWindow');
const POSITION = attr('AXPosition');

const systemWide = AXUIElementCreateSystemWide();

// 성공하면 CFTypeRef (호출한 쪽에서 CFRelease), 실패하면 null
function copy(element, attribute) {
  const out = [null];
  return AXUIElementCopyAttributeValue(element, attribute, out) === 0 ? out[0] : null;
}

function frontWindow() {
  // 시스템 전체 포커스 조회는 -25204 로 실패하는 경우가 있어 NSWorkspace 의 맨 앞 앱을 먼저 쓴다
  const pid = frontmostPid();
  const app = pid > 0 ? AXUIElementCreateApplication(pid) : copy(systemWide, FOCUSED_APP);
  if (!app) return null;
  const win = copy(app, FOCUSED_WINDOW) ?? copy(app, MAIN_WINDOW);
  CFRelease(app);
  return win;
}

function getPosition(win) {
  const value = copy(win, POSITION);
  if (!value) return null;
  const pos = {};
  const ok = AXValueGetValue(value, AX_VALUE_CGPOINT, pos);
  CFRelease(value);
  return ok ? pos : null;
}

function setPosition(win, pos) {
  const value = AXValueCreate(AX_VALUE_CGPOINT, pos);
  const err = AXUIElementSetAttributeValue(win, POSITION, value);
  CFRelease(value);
  return err === 0;
}

let grabbed = null; // { win, start: 창의 처음 위치 }

// 맨 앞 창을 잡는다. 잡을 창이 없거나 권한이 없으면 false
function grab() {
  release();
  const win = frontWindow();
  if (!win) return false;
  const start = getPosition(win);
  if (!start) {
    CFRelease(win);
    return false;
  }
  grabbed = { win, start };
  return true;
}

// 잡은 순간부터 손이 움직인 거리(point)만큼 창을 옮긴다
function moveBy(dx, dy) {
  if (!grabbed) return;
  setPosition(grabbed.win, { x: Math.round(grabbed.start.x + dx), y: Math.round(grabbed.start.y + dy) });
}

function release() {
  if (!grabbed) return;
  CFRelease(grabbed.win);
  grabbed = null;
}

module.exports = { grab, moveBy, release, isGrabbing: () => grabbed !== null };
