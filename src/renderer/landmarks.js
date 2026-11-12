// HandLandmarker 결과 → 손별 21개 관절 3D 좌표, 좌표 표, JSON 저장

// MediaPipe Hands 21개 관절 순서
export const JOINTS = [
  'WRIST',
  'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP',
  'INDEX_MCP', 'INDEX_PIP', 'INDEX_DIP', 'INDEX_TIP',
  'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
  'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
  'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP',
];

/**
 * 인식된 모든 손(최대 2개)을 정리
 * - landmarks: x,y 는 영상 크기로 정규화(0~1), z 는 손목 기준 상대 깊이 (x 와 같은 스케일, 작을수록 카메라에 가까움)
 * - worldLandmarks: 손의 기하학적 중심을 원점으로 한 실제 미터 단위 3D 좌표
 */
export function extractHands(result, timestamp) {
  return result.landmarks.map((landmarks, i) => ({
    // 웹캠 원본(반전 전) 영상 기준으로 MediaPipe 라벨이 실제 손과 일치함
    handedness: result.handednesses[i]?.[0]?.categoryName ?? 'Unknown',
    score: result.handednesses[i]?.[0]?.score ?? 0,
    timestamp,
    landmarks,
    worldLandmarks: result.worldLandmarks[i],
  }));
}

export const toJoints = (hand) => hand.landmarks.map((p, j) => {
  const w = hand.worldLandmarks[j];
  return { id: j, name: JOINTS[j], image: { x: p.x, y: p.y, z: p.z }, world: { x: w.x, y: w.y, z: w.z } };
});

export function renderTable(rowsEl, hands) {
  const f = (v) => v.toFixed(3);
  rowsEl.innerHTML = hands
    .map((hand) =>
      toJoints(hand)
        .map(({ id, name, image: p, world: w }) =>
          `<tr><td>${id}</td><td>${name}</td><td>${f(p.x)}</td><td>${f(p.y)}</td><td>${f(p.z)}</td>` +
          `<td>${f(w.x)}</td><td>${f(w.y)}</td><td>${f(w.z)}</td></tr>`)
        .join(''))
    .join('');
}

export function saveJson(meta, hands) {
  if (!hands.length) return;
  const data = hands.map(({ landmarks, worldLandmarks, ...rest }) =>
    ({ ...meta, ...rest, joints: toJoints({ landmarks, worldLandmarks }) }));
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${meta.participant}_S${meta.session}_${meta.hand}_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}
