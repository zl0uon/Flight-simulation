// 카메라 위 오버레이: 손 21개 점과 선, 커서 영역, 주먹 진행 막대
import { DrawingUtils, HandLandmarker } from '/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs';
import { PINCH_ON, REGION } from './gestures.js';

export const COLORS = {
  move: '#00e676', click: '#ffea00', drag: '#ff9100', scroll: '#40c4ff', grab: '#e040fb', paused: '#9e9e9e',
};
const OTHER_HAND = '#9e9e9e'; // 조작에 쓰지 않는 손

export function createOverlay(canvas) {
  const ctx = canvas.getContext('2d');
  const drawer = new DrawingUtils(ctx);

  return function draw(hands, controlHand, status) {
    const { width: W, height: H } = canvas;
    ctx.clearRect(0, 0, W, H);

    // 커서 영역 (영상이 좌우반전이라 x를 뒤집어 그림)
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = 2;
    ctx.strokeRect((1 - REGION.x1) * W, REGION.y0 * H, (REGION.x1 - REGION.x0) * W, (REGION.y1 - REGION.y0) * H);
    ctx.setLineDash([]);

    for (const hand of hands) {
      const mirrored = hand.landmarks.map((p) => ({ ...p, x: 1 - p.x }));
      const color = hand === controlHand ? COLORS[status.mode] ?? '#ffffff' : OTHER_HAND;
      drawer.drawConnectors(mirrored, HandLandmarker.HAND_CONNECTIONS, { color, lineWidth: 3 });
      drawer.drawLandmarks(mirrored, { color: '#fff', fillColor: color, radius: 3 });
    }

    // 꼬집기 게이지: 엄지-검지 거리가 기준선(PINCH_ON) 아래로 내려가면 클릭
    if (status.pinchL !== undefined) {
      const MAX = 1.2;
      const barW = 90;
      const x = 10;
      const y = H - 22;
      ctx.fillStyle = 'rgba(0,0,0,0.55)';
      ctx.fillRect(x - 4, y - 4, barW + 8, 16);
      ctx.fillStyle = status.pinchL < PINCH_ON ? COLORS.click : 'rgba(255,255,255,0.7)';
      ctx.fillRect(x, y, barW * Math.min(1, status.pinchL / MAX), 8);
      ctx.fillStyle = '#ff1744';
      ctx.fillRect(x + barW * (PINCH_ON / MAX) - 1, y - 3, 2, 14);
    }

    if (status.fistProgress && status.mode !== 'paused') {
      ctx.fillStyle = 'rgba(255,255,255,0.8)';
      ctx.fillRect(0, H - 8, W * status.fistProgress, 8);
    }
  };
}
