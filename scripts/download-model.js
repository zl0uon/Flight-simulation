// npm install 직후(postinstall) 실행: HandLandmarker 모델을 Models 폴더에 받는다
const fs = require('fs');
const path = require('path');

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const DEST = path.join(__dirname, '..', 'Models', 'hand_landmarker.task');

async function main() {
  if (fs.existsSync(DEST) && fs.statSync(DEST).size > 0) {
    console.log(`모델이 이미 있어요: ${path.relative(process.cwd(), DEST)}`);
    return;
  }
  fs.mkdirSync(path.dirname(DEST), { recursive: true });
  console.log(`모델 다운로드 중… ${MODEL_URL}`);
  const res = await fetch(MODEL_URL);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  // 받다가 끊기면 깨진 파일이 남지 않도록 임시 파일에 받은 뒤 이름을 바꾼다
  const tmp = `${DEST}.download`;
  fs.writeFileSync(tmp, Buffer.from(await res.arrayBuffer()));
  fs.renameSync(tmp, DEST);
  console.log(`저장 완료: ${path.relative(process.cwd(), DEST)} (${fs.statSync(DEST).size} bytes)`);
}

main().catch((err) => {
  console.error(`모델 다운로드 실패: ${err.message}\n'npm run download-model' 로 다시 시도하세요.`);
  process.exitCode = 1;
});
