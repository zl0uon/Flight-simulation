import { OneEuro } from './one-euro.js';

// 카메라 화면 중 이 영역이 전체 화면에 대응 (손을 조금만 움직여도 화면 끝까지)
export const REGION = { x0: 0.2, x1: 0.8, y0: 0.15, y1: 0.7 };

export const PINCH_ON = 0.38; // 손가락 끝 거리 / 손 크기 가 이보다 작으면 꼬집기 (3D 추정 오차가 커서 너무 작으면 잘 안 눌림)
const PINCH_OFF = 0.55;  // 이보다 커지면 놓기
const RIGHT_INDEX_AWAY = 0.45; // 오른쪽 클릭: 검지가 엄지에서 이만큼은 떨어져 있어야 함 (아니면 왼쪽 클릭으로 봄)
const DRAG_START = 0.035; // 꼬집은 채 이만큼(화면 비율) 움직이면 클릭 대신 드래그
const QUICK_CLICK = 0.3; // 이보다 짧게(초) 꼬집었다 놓으면 드래그로 흔들렸어도 누른 자리에서 클릭
const THUMB_AWAY = 0.45; // 스크롤 자세: 엄지가 검지에서 이만큼 떨어져 있어야 함
const PINCH_LOOKBACK = 0.1; // 꼬집는 순간 손이 흔들리므로, 이만큼 전 위치에서 클릭 (초)
const SCROLL_SPEED = 2500;  // 손을 카메라 화면 높이만큼 움직였을 때 스크롤 픽셀
const SCROLL_NATURAL = true; // 트랙패드처럼 손을 올리면 내용이 올라감
const FIST_HOLD = 1.0;   // 주먹을 이만큼(초) 유지하면 일시정지 전환
const GRAB_ON = 0.45;    // 엄지 끝과 나머지 네 손가락 끝의 최대 거리 / 손 크기 가 이보다 작으면 창 잡기
const GRAB_OFF = 0.75;   // 이보다 커지면 창 놓기
const GRAB_REACH = 1.15; // 중지 끝-손목 거리 / 손 크기 가 이보다 커야 잡기 (주먹은 손끝이 손바닥에 붙어 있어 더 작음)

const clamp01 = (v) => Math.min(1, Math.max(0, v));

export class GestureMouse {
  constructor(api) {
    this.api = api;
    // minCutoff: 낮을수록 멈춰 있을 때 덜 떨림 / beta: 클수록 빠르게 움직일 때 지연이 적음
    this.fx = new OneEuro({ minCutoff: 2.0, beta: 25 });
    this.fy = new OneEuro({ minCutoff: 2.0, beta: 25 });
    this.paused = false;
    this.history = [];
    this.release();
  }

  // 손을 놓쳤거나 일시정지할 때 모든 상태 정리
  release() {
    if (this.grabbing) this.api.windowRelease();
    this.grabbing = false;
    this.grabOk = false;
    if (this.leftHeld) this.api.left(false);
    this.leftHeld = false;
    this.dragging = false;
    this.rightArmed = true;
    this.scrollY = null;
    this.fistSince = null;
    this.fistLatched = false;
    this.fx.reset();
    this.fy.reset();
    this.history = [];
    this.velocity = { x: 0, y: 0 };
    this.api.stop();
  }

  togglePause() {
    this.paused = !this.paused;
    this.release();
  }

  update(result, t) {
    const lm = result.landmarks[0];
    const w = result.worldLandmarks?.[0];
    if (!lm || !w) {
      if (this.hadHand) this.release();
      this.hadHand = false;
      return { mode: this.paused ? 'paused' : 'none' };
    }
    this.hadHand = true;

    // 3D 월드 좌표(미터)로 판정해서 손과 카메라 거리와 무관하게
    const d3 = (a, b) => Math.hypot(w[a].x - w[b].x, w[a].y - w[b].y, w[a].z - w[b].z);
    const palm = d3(0, 9) || 1;
    const extended = (tip, pip) => d3(0, tip) > d3(0, pip);
    const fingers = {
      index: extended(8, 6),
      middle: extended(12, 10),
      ring: extended(16, 14),
      pinky: extended(20, 18),
    };
    const pinchL = d3(4, 8) / palm;
    const pinchR = d3(4, 12) / palm;
    const bunch = Math.max(d3(4, 8), d3(4, 12), d3(4, 16), d3(4, 20)) / palm;
    const grabPose = bunch < (this.grabbing ? GRAB_OFF : GRAB_ON) && d3(0, 12) / palm > GRAB_REACH;
    const status = { pinchL, pinchR, bunch };

    // --- 주먹: 일시정지 전환 ---
    const fist = !fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky && !grabPose;
    if (fist) {
      this.fistSince ??= t;
      status.fistProgress = Math.min(1, (t - this.fistSince) / FIST_HOLD);
      if (!this.fistLatched && t - this.fistSince >= FIST_HOLD) {
        this.fistLatched = true;
        this.togglePause();
        this.fistLatched = true; // release()가 초기화하므로 다시 설정
        this.fistSince = t;
      }
      if (this.leftHeld) {
        this.api.left(false);
        this.leftHeld = false;
      }
      if (this.grabbing) {
        this.api.windowRelease();
        this.grabbing = false;
      }
      return { ...status, mode: this.paused ? 'paused' : 'fist' };
    }
    this.fistSince = null;
    this.fistLatched = false;
    if (this.paused) return { ...status, mode: 'paused' };

    // --- 커서 위치: 검지 뿌리(꼬집어도 덜 움직임), 거울 반전 ---
    const ax = 1 - lm[5].x;
    const ay = lm[5].y;
    const nx = this.fx.filter(clamp01((ax - REGION.x0) / (REGION.x1 - REGION.x0)), t);
    const ny = this.fy.filter(clamp01((ay - REGION.y0) / (REGION.y1 - REGION.y0)), t);
    // 커서 속도 (화면 비율/초) — 메인 프로세스가 프레임 사이 위치를 예측하는 데 사용
    const prev = this.history.at(-1);
    if (prev && t > prev.t) {
      this.velocity = {
        x: 0.5 * this.velocity.x + 0.5 * ((nx - prev.x) / (t - prev.t)),
        y: 0.5 * this.velocity.y + 0.5 * ((ny - prev.y) / (t - prev.t)),
      };
    }
    this.history.push({ t, x: nx, y: ny });
    this.history = this.history.filter((h) => t - h.t <= PINCH_LOOKBACK + 0.1);
    status.cursor = { x: nx, y: ny };

    // --- 창 잡기: 다섯 손가락 끝을 한데 모으면 맨 앞 창이 손을 따라 움직임 ---
    if (this.grabbing && !grabPose) {
      this.api.windowRelease();
      this.grabbing = false;
    } else if (!this.grabbing && grabPose) {
      // 손가락을 모으는 도중 엄지+검지가 먼저 닿아 눌린 왼쪽 버튼은 놓는다
      if (this.leftHeld) this.api.left(false);
      this.leftHeld = false;
      this.dragging = false;
      this.api.stop();
      this.grabbing = true;
      this.grabOk = false;
      this.api.windowGrab(nx, ny).then((ok) => (this.grabOk = ok && this.grabbing));
    }
    if (this.grabbing) {
      this.api.windowMove(nx, ny);
      return { ...status, mode: 'grab', grabOk: this.grabOk };
    }

    // --- 스크롤: 검지+중지만 펴기 ---
    const scrollPose = fingers.index && fingers.middle && !fingers.ring && !fingers.pinky
      && pinchL > THUMB_AWAY && !this.leftHeld;
    if (scrollPose) {
      if (this.scrollY !== null) {
        const dy = (ay - this.scrollY) * SCROLL_SPEED;
        if (Math.abs(dy) >= 1) this.api.scroll(SCROLL_NATURAL ? dy : -dy);
      }
      this.scrollY = ay;
      this.api.stop();
      return { ...status, mode: 'scroll' };
    }
    this.scrollY = null;

    // --- 오른쪽 클릭: 엄지+중지 (검지는 엄지에서 확실히 떨어져 있어야 왼쪽 클릭과 헷갈리지 않음) ---
    if (!this.leftHeld && this.rightArmed && pinchR < PINCH_ON && pinchL > RIGHT_INDEX_AWAY) {
      this.rightArmed = false;
      this.api.right();
    } else if (!this.rightArmed && pinchR > PINCH_OFF) {
      this.rightArmed = true;
    }

    // --- 왼쪽 클릭 / 드래그: 엄지+검지 ---
    // 꼬집을 때 중지가 검지 옆에 붙어 엄지에 더 가까울 때가 많으므로, 검지가 닿았으면 항상 왼쪽 클릭
    if (!this.leftHeld && pinchL < PINCH_ON) {
      // 꼬집는 동작으로 손이 흔들리기 전 위치에서 누르기
      const before = this.history.find((h) => t - h.t <= PINCH_LOOKBACK) ?? this.history.at(-1);
      this.pinchAt = { x: before.x, y: before.y };
      this.api.move(before.x, before.y);
      this.api.left(true);
      this.leftHeld = true;
      this.dragging = false;
      this.pinchStart = t;
    } else if (this.leftHeld && pinchL > PINCH_OFF) {
      // 짧게 꼬집은 건 클릭이 목적이므로 누른 자리로 돌아가서 놓는다 (버튼 밖으로 미끄러져 클릭이 취소되지 않게)
      if (t - this.pinchStart < QUICK_CLICK) this.api.move(this.pinchAt.x, this.pinchAt.y);
      this.api.left(false);
      this.leftHeld = false;
      this.dragging = false;
    }

    if (this.leftHeld) {
      if (!this.dragging && Math.hypot(nx - this.pinchAt.x, ny - this.pinchAt.y) > DRAG_START) {
        this.dragging = true;
      }
      // 드래그로 판정되기 전엔 커서를 고정해서 클릭이 미끄러지지 않게
      if (this.dragging) this.api.move(nx, ny, this.velocity.x, this.velocity.y);
      return { ...status, mode: this.dragging ? 'drag' : 'click' };
    }

    this.api.move(nx, ny, this.velocity.x, this.velocity.y);
    return { ...status, mode: 'move' };
  }
}
