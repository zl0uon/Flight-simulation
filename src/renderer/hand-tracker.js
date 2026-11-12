// MediaPipe HandLandmarker 생성 (npm 으로 받은 라이브러리와 Models 폴더의 모델을 app:// 로 불러옴)
import { FilesetResolver, HandLandmarker } from '/node_modules/@mediapipe/tasks-vision/vision_bundle.mjs';

const WASM_DIR = '/node_modules/@mediapipe/tasks-vision/wasm';
const MODEL_PATH = '/Models/hand_landmarker.task';

export async function createHandTracker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_DIR);
  return HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numHands: 2,
  });
}
