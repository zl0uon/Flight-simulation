"""종료: 창의 닫기 버튼(macOS는 창 왼쪽 위 빨간 버튼), q 또는 ESC, 터미널에서 Ctrl+C
"""

import argparse
import math
import shutil
import ssl
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"

# 21개 랜드마크 사이의 연결선 (손 뼈대)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),          # 검지
    (5, 9), (9, 10), (10, 11), (11, 12),     # 중지
    (9, 13), (13, 14), (14, 15), (15, 16),   # 약지
    (13, 17), (17, 18), (18, 19), (19, 20),  # 새끼
    (0, 17),                                 # 손바닥
]
FINGER_TIPS = [4, 8, 12, 16, 20]

WINDOW_NAME = "MediaPipe Hand Tracking"

# 펼친 손가락 개수 -> 스로틀(%)
# 엄지 판정이 가장 불안정하므로 4개와 5개는 둘 다 100%로 처리
THROTTLE_MAP = {0: 0, 1: 25, 2: 50, 3: 75, 4: 100, 5: 100}


class FlightController:
    """손 랜드마크를 조종 입력으로 변환.

    roll(-1.0~1.0)   손을 좌우로 기울인 각도 (화면에서 본 손목 -> 중지 뿌리 방향)
    pitch(-1.0~1.0)  손바닥을 앞뒤로 기울인 각도. 손끝을 내 쪽(카메라 반대쪽)으로 젖히면 +(상승),
                     손끝을 카메라 쪽으로 숙이면 -(하강). 조종간을 당기고 미는 것과 같은 방향이다.
    스로틀(0~100%)   펼친 손가락 개수
    """

    HOLD_FRAMES = 12   # 손을 잠깐 놓쳐도 이만큼(약 0.4초)은 직전 조종 입력을 유지한다
    KNUCKLES = (5, 9, 13, 17)   # 검지~새끼 손가락 뿌리
    CALIB_FRAMES = 20        # 중립 자세를 재는 프레임 수 (약 0.7초)
    MAX_NEUTRAL = 25.0       # 중립으로 인정하는 좌우 기울기의 한계 (도)
    MAX_PITCH_NEUTRAL = 20.0

    def __init__(self, window: int = 5, deadzone_deg: float = 5.0,
                 max_roll_deg: float = 30.0, pitch_deadzone_deg: float = 10.0,
                 max_pitch_deg: float = 40.0, invert_pitch: bool = False):
        self.angles = deque(maxlen=window)   # 최근 N프레임 손목 기울기(도)
        self.pitch_angles = deque(maxlen=window)   # 최근 N프레임 손바닥 앞뒤 기울기(도)
        self.fingers = deque(maxlen=window)  # 최근 N프레임 손가락 개수
        self.deadzone = deadzone_deg
        self.max_roll = max_roll_deg
        self.pitch_deadzone = pitch_deadzone_deg
        self.max_pitch = max_pitch_deg
        self.invert_pitch = invert_pitch
        self.angle = 0.0     # 평활화된 기울기(도)
        self.pitch_angle = 0.0   # 평활화된 앞뒤 기울기(도)
        self.roll = 0.0      # -1.0(좌) ~ 1.0(우)
        self.pitch = 0.0     # -1.0(하강) ~ 1.0(상승)
        self.throttle = 0    # 0, 25, 50, 75, 100
        self.missed = 0      # 손을 연속으로 놓친 프레임 수
        # 중립 자세: 편하게 든 손은 똑바로 서지 않고 한쪽으로 기울어 있다(팔이 옆에서 올라오므로).
        # 그 각도를 0으로 삼지 않으면 한쪽은 조금만 기울여도 돌고 반대쪽은 한참 꺾어야 돈다.
        self.neutral = 0.0
        self.pitch_neutral = 0.0
        self._calib = []     # 중립을 재는 중이면 [(좌우 각, 앞뒤 각), ...], 끝났으면 None

    @staticmethod
    def palm_pitch(world) -> float:
        """손바닥의 앞뒤 기울기(도). 손끝이 카메라에서 멀어지게 젖혀지면 +, 카메라 쪽으로 숙이면 -.

        world = MediaPipe의 월드 랜드마크 (미터 단위 3D 좌표, z는 카메라에 가까울수록 작다).
        손목(0) -> 중지 뿌리(9) 벡터가 화면 평면에서 앞뒤로 얼마나 벗어났는지를 잰다.
        좌우로 기울인 상태에서도 재려고 화면 평면 성분 전체(x, y)의 길이와 비교한다.
        """
        dx = world[9].x - world[0].x
        dy = world[9].y - world[0].y
        dz = world[9].z - world[0].z
        return math.degrees(math.atan2(dz, math.hypot(dx, dy)))

    @staticmethod
    def _scaled(angle: float, deadzone: float, full: float) -> float:
        """데드존 안은 0, 밖은 0부터 부드럽게 시작해 full도에서 ±1."""
        a = abs(angle)
        if a < deadzone:
            return 0.0
        return math.copysign(min((a - deadzone) / (full - deadzone), 1.0), angle)

    @classmethod
    def wrist_angle(cls, landmarks, w: int, h: int) -> float:
        """손목(0) -> 손가락 뿌리 네 점의 중심 벡터의 수직 대비 기울기(도). 오른쪽으로 기울면 +.

        한 점(중지 뿌리)만 쓰면 그 점이 떨릴 때 각도가 같이 떨려서, 네 점의 평균을 쓴다.
        """
        # 정규화 좌표는 가로세로 비율이 달라 픽셀 단위로 바꿔서 계산
        n = len(cls.KNUCKLES)
        dx = (sum(landmarks[i].x for i in cls.KNUCKLES) / n - landmarks[0].x) * w
        dy = (sum(landmarks[i].y for i in cls.KNUCKLES) / n - landmarks[0].y) * h
        return math.degrees(math.atan2(dx, -dy))  # 손이 똑바로 서 있으면 0도

    def update(self, landmarks, handedness: str, w: int, h: int, world=None):
        """world(월드 랜드마크)를 주면 손바닥 앞뒤 기울기로 pitch도 정한다."""
        self.missed = 0
        angle = self.wrist_angle(landmarks, w, h)
        tilt = None
        if world is not None:
            tilt = self.palm_pitch(world)
            tilt = -tilt if self.invert_pitch else tilt
        if self._calib is not None:   # 처음 손이 보인 동안의 자세를 중립으로 삼는다
            self._calib.append((angle, tilt))
            if len(self._calib) >= self.CALIB_FRAMES:
                self._finish_calibration()
        self.angles.append(angle - self.neutral)
        self.fingers.append(count_fingers(landmarks, handedness))
        if tilt is not None:
            self.pitch_angles.append(tilt - self.pitch_neutral)

        # 이동 평균
        self.angle = sum(self.angles) / len(self.angles)
        avg_fingers = sum(self.fingers) / len(self.fingers)
        if self.pitch_angles:
            self.pitch_angle = sum(self.pitch_angles) / len(self.pitch_angles)

        # 데드존: 작은 기울기는 0으로, 데드존 밖은 0부터 부드럽게 시작하도록 재스케일
        self.roll = self._scaled(self.angle, self.deadzone, self.max_roll)
        self.pitch = self._scaled(self.pitch_angle, self.pitch_deadzone, self.max_pitch)

        self.throttle = THROTTLE_MAP[int(round(avg_fingers))]

    @property
    def calibrating(self) -> bool:
        return self._calib is not None

    def recalibrate(self):
        """지금부터 보이는 손 자세를 새 중립으로 삼는다."""
        self._calib = []

    def _finish_calibration(self):
        median = lambda v: sorted(v)[len(v) // 2]
        clamp = lambda v, lim: max(-lim, min(lim, v))
        self.neutral = clamp(median([a for a, _ in self._calib]), self.MAX_NEUTRAL)
        tilts = [t for _, t in self._calib if t is not None]
        self.pitch_neutral = clamp(median(tilts), self.MAX_PITCH_NEUTRAL) if tilts else 0.0
        self._calib = None
        self.angles.clear()
        self.pitch_angles.clear()

    def lost(self):
        """이번 프레임에 손을 못 찾았다. 잠깐 놓친 것이면 직전 입력을 유지하고, 길어지면 중립으로.

        손을 크게 기울이면 인식이 한두 프레임씩 끊기는데, 그때마다 중립으로 돌아가면
        선회가 계속 풀려 버린다.
        """
        self.missed += 1
        if self.missed > self.HOLD_FRAMES:
            self.reset()

    def reset(self):
        """손을 놓치면 조종간(roll, pitch)은 중립으로 복귀, 스로틀은 마지막 값 유지."""
        self.angles.clear()
        self.pitch_angles.clear()
        self.fingers.clear()
        self.angle = 0.0
        self.pitch_angle = 0.0
        self.roll = 0.0
        self.pitch = 0.0


def draw_hud(frame, ctrl: FlightController, active: bool):
    h, w = frame.shape[:2]
    color = (0, 255, 255) if active else (128, 128, 128)

    # Roll 바 (화면 하단 가운데)
    cx, y, half = w // 2, h - 40, 150
    cv2.rectangle(frame, (cx - half, y - 10), (cx + half, y + 10), (80, 80, 80), 1)
    dz = int(half * ctrl.deadzone / ctrl.max_roll)
    cv2.rectangle(frame, (cx - dz, y - 10), (cx + dz, y + 10), (60, 60, 60), -1)
    cv2.line(frame, (cx, y - 14), (cx, y + 14), (200, 200, 200), 1)
    px = int(cx + ctrl.roll * half)
    cv2.rectangle(frame, (min(cx, px), y - 8), (max(cx, px), y + 8), color, -1)
    cv2.putText(frame, f"ROLL {ctrl.roll:+.2f} ({ctrl.angle:+.0f} deg)",
                (cx - half, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Pitch 바 (화면 왼쪽, 위 = 상승)
    px0, ptop, pbottom = 25, 80, h - 100
    pmid = (ptop + pbottom) // 2
    cv2.rectangle(frame, (px0, ptop), (px0 + 25, pbottom), (80, 80, 80), 1)
    pdz = int((pbottom - ptop) / 2 * ctrl.pitch_deadzone / ctrl.max_pitch)
    cv2.rectangle(frame, (px0, pmid - pdz), (px0 + 25, pmid + pdz), (60, 60, 60), -1)
    py = int(pmid - ctrl.pitch * (pbottom - ptop) / 2)
    cv2.rectangle(frame, (px0 + 2, min(pmid, py)), (px0 + 23, max(pmid, py)), color, -1)
    cv2.line(frame, (px0 - 4, pmid), (px0 + 29, pmid), (200, 200, 200), 1)
    cv2.putText(frame, f"PITCH {ctrl.pitch:+.2f} ({ctrl.pitch_angle:+.0f} deg)", (px0 - 15, ptop - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 스로틀 바 (화면 오른쪽)
    x, top, bottom = w - 50, 80, h - 100
    cv2.rectangle(frame, (x, top), (x + 25, bottom), (80, 80, 80), 1)
    fill = int((bottom - top) * ctrl.throttle / 100)
    cv2.rectangle(frame, (x, bottom - fill), (x + 25, bottom), color, -1)
    cv2.putText(frame, f"{ctrl.throttle}%", (x - 20, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if not active:
        cv2.putText(frame, "NO HAND", (cx - 60, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)


def ensure_model(path: Path = MODEL_PATH) -> Path:
    """모델 파일이 없으면 다운로드."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"모델 다운로드 중... -> {path}")
        # python.org판 macOS Python은 시스템 루트 인증서를 쓰지 않아
        # CERTIFICATE_VERIFY_FAILED가 나므로 certifi 인증서 묶음을 사용
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ctx = ssl.create_default_context()
        # 중간에 끊겨도 깨진 모델 파일이 남지 않도록 임시 파일에 받은 뒤 교체
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, context=ctx) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.replace(path)
    return path


def forbid_frame_saving():
    """웹캠 영상은 메모리에서만 처리한다: 이 프로세스에서 OpenCV 이미지/영상 저장을 막는다.

    현재 코드에는 프레임을 저장하는 곳이 없지만, 나중에 디버깅용 저장 코드가
    실수로 들어가도 실행 즉시 오류가 나도록 하는 안전장치.
    """
    def blocked(*_args, **_kwargs):
        raise RuntimeError("개인정보 보호: 웹캠 프레임을 디스크에 저장하는 것은 금지되어 있습니다.")

    for name in ("imwrite", "imwritemulti", "VideoWriter"):
        if hasattr(cv2, name):
            setattr(cv2, name, blocked)


def count_fingers(landmarks, handedness: str) -> int:
    """펴진 손가락 개수를 셉니다 (손바닥이 카메라를 향한다고 가정)."""
    count = 0
    # 엄지: x좌표 비교 (좌우 반전된 화면 기준)
    thumb_tip, thumb_ip = landmarks[4], landmarks[3]
    if handedness == "Right":
        count += thumb_tip.x < thumb_ip.x
    else:
        count += thumb_tip.x > thumb_ip.x
    # 나머지 손가락: 끝마디가 두 번째 마디보다 위(y가 작음)면 펴진 것
    for tip in FINGER_TIPS[1:]:
        count += landmarks[tip].y < landmarks[tip - 2].y
    return int(count)


def draw_hand(frame, landmarks, label: str):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
    for i, p in enumerate(pts):
        color = (0, 0, 255) if i in FINGER_TIPS else (0, 255, 0)
        cv2.circle(frame, p, 5, color, -1)

    x, y = pts[0]
    cv2.putText(frame, label, (x - 40, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def main():
    parser = argparse.ArgumentParser(description="MediaPipe 손 인식")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--max-hands", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--smooth", type=int, default=5, help="이동 평균 프레임 수")
    parser.add_argument("--deadzone", type=float, default=5.0, help="roll 데드존(도)")
    parser.add_argument("--pitch-deadzone", type=float, default=10.0, help="pitch 데드존(도)")
    parser.add_argument("--max-pitch", type=float, default=40.0,
                        help="pitch가 최대(±1.0)가 되는 손바닥 앞뒤 기울기(도)")
    parser.add_argument("--invert-pitch", action="store_true", help="pitch 방향을 반대로")
    parser.add_argument("--max-roll", type=float, default=30.0,
                        help="roll이 최대(±1.0)가 되는 기울기(도)")
    args = parser.parse_args()

    forbid_frame_saving()
    print("웹캠 영상은 메모리에서만 처리되며 디스크에 저장되지 않습니다.")
    ctrl = FlightController(args.smooth, args.deadzone, args.max_roll,
                            args.pitch_deadzone, args.max_pitch, args.invert_pitch)

    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"카메라 {args.camera}번을 열 수 없습니다. "
                         "(macOS: 시스템 설정 > 개인정보 보호 > 카메라 권한 확인)")

    prev_time = time.time()
    start = time.monotonic()

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("프레임을 읽지 못했습니다.")
                    break

                frame = cv2.flip(frame, 1)  # 거울 모드
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.monotonic() - start) * 1000)

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    # 거울 모드로 뒤집었으므로 MediaPipe의 Left/Right가 실제 손과 일치
                    side = handedness[0].category_name
                    fingers = count_fingers(landmarks, side)
                    draw_hand(frame, landmarks, f"{side}: {fingers}")

                # 조종은 첫 번째로 인식된 손으로 함
                h, w = frame.shape[:2]
                if result.hand_landmarks:
                    world = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
                    ctrl.update(result.hand_landmarks[0],
                                result.handedness[0][0].category_name, w, h, world)
                else:
                    ctrl.lost()
                draw_hud(frame, ctrl, bool(result.hand_landmarks))

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}  Hands: {len(result.hand_landmarks)}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
                # 창의 닫기 버튼으로 닫으면 종료
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    except KeyboardInterrupt:
        pass  # 터미널에서 Ctrl+C
    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # macOS에서 창이 실제로 닫히도록 이벤트 한 번 처리
        print("종료했습니다.")


if __name__ == "__main__":
    main()
