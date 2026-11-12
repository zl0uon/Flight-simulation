// macOS CoreGraphics 를 직접 호출해서 실제 마우스 이벤트를 만든다 (koffi FFI)
const koffi = require('koffi');

const CG = koffi.load('/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics');
const CF = koffi.load('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation');

const CGPoint = koffi.struct('CGPoint', { x: 'double', y: 'double' });

const CGEventCreate = CG.func('void *CGEventCreate(void *source)');
const CGEventGetLocation = CG.func('CGPoint CGEventGetLocation(void *event)');
const CGEventCreateMouseEvent = CG.func(
  'void *CGEventCreateMouseEvent(void *source, uint32_t type, CGPoint pos, uint32_t button)'
);
const CGEventCreateScrollWheelEvent2 = CG.func(
  'void *CGEventCreateScrollWheelEvent2(void *source, uint32_t units, uint32_t count, int32_t w1, int32_t w2, int32_t w3)'
);
const CGEventSetIntegerValueField = CG.func(
  'void CGEventSetIntegerValueField(void *event, uint32_t field, int64_t value)'
);
const CGEventPost = CG.func('void CGEventPost(uint32_t tap, void *event)');
const CFRelease = CF.func('void CFRelease(void *cf)');

// CGEventType
const LEFT_DOWN = 1;
const LEFT_UP = 2;
const RIGHT_DOWN = 3;
const RIGHT_UP = 4;
const MOVED = 5;
const LEFT_DRAGGED = 6;

const BUTTON_LEFT = 0;
const BUTTON_RIGHT = 1;
const HID_TAP = 0;
const FIELD_CLICK_STATE = 1;
const SCROLL_UNIT_PIXEL = 0;

const DOUBLE_CLICK_MS = 450;
const DOUBLE_CLICK_DIST = 10;

let leftHeld = false;
let lastClick = { time: 0, x: 0, y: 0, count: 0 };

function post(type, pos, button, clickState) {
  const event = CGEventCreateMouseEvent(null, type, pos, button);
  if (clickState) CGEventSetIntegerValueField(event, FIELD_CLICK_STATE, clickState);
  CGEventPost(HID_TAP, event);
  CFRelease(event);
}

function getPosition() {
  const event = CGEventCreate(null);
  const pos = CGEventGetLocation(event);
  CFRelease(event);
  return pos;
}

// 왼쪽 버튼을 누르고 있으면 드래그 이벤트로 보내야 앱들이 드래그로 인식한다
function move(pos) {
  post(leftHeld ? LEFT_DRAGGED : MOVED, pos, BUTTON_LEFT);
}

function leftDown(pos) {
  if (leftHeld) return;
  const now = Date.now();
  const near = Math.hypot(pos.x - lastClick.x, pos.y - lastClick.y) < DOUBLE_CLICK_DIST;
  // 짧은 시간 안에 같은 곳을 다시 누르면 더블/트리플 클릭
  const count = now - lastClick.time < DOUBLE_CLICK_MS && near ? lastClick.count + 1 : 1;
  lastClick = { time: now, x: pos.x, y: pos.y, count };
  leftHeld = true;
  post(LEFT_DOWN, pos, BUTTON_LEFT, count);
}

function leftUp(pos) {
  if (!leftHeld) return;
  leftHeld = false;
  post(LEFT_UP, pos, BUTTON_LEFT, lastClick.count);
}

function rightClick(pos) {
  post(RIGHT_DOWN, pos, BUTTON_RIGHT, 1);
  post(RIGHT_UP, pos, BUTTON_RIGHT, 1);
}

function scroll(dy) {
  const event = CGEventCreateScrollWheelEvent2(null, SCROLL_UNIT_PIXEL, 1, Math.round(dy), 0, 0);
  CGEventPost(HID_TAP, event);
  CFRelease(event);
}

module.exports = { getPosition, move, leftDown, leftUp, rightClick, scroll, isLeftHeld: () => leftHeld };
