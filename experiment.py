"""제스처 비행 조종 시뮬레이션. 시작할 때 세 가지 비행 모드 중 하나를 고른다 (--flight).

    staged  단계별 시뮬레이션: 이륙 -> 직선 -> 선회 -> 착륙 과제를 차례로 수행하는 실험.
            참가자 코드와 동의가 필요하고 시도마다 결과를 trials.csv에 누적 저장한다.
    free    자유 비행: 과제·제한 시간·판정 없이 손으로 자유롭게 난다. 기록하지 않는다.
            R 키로 활주로 출발 위치로 돌아간다.
    auto    자동 조종: 컴퓨터가 스스로 비행하고 사용자는 지켜본다. 카메라를 켜지 않고
            기록하지 않는다. 창을 닫을 때까지 반복한다.

개인정보 보호
- 시작 시 동의 안내문을 출력하고, 참가자가 동의해야만 카메라를 켠다.
- 참가자는 익명 코드(예: P001)로만 기록한다. 이름 같은 형식은 거부한다.
- 웹캠 프레임은 메모리에서 손 인식과 화면 미리보기에만 쓰고 디스크에 저장하지 않는다.
  (forbid_frame_saving()이 실행 중 OpenCV 이미지/영상 저장 함수를 막아 둔다.)

사용 예
    python3 experiment.py                                           # 비행 모드와 낮/밤을 물어본다
    python3 experiment.py --flight free --time night                # 밤에 자유 비행
    python3 experiment.py --flight free                             # 자유 비행
    python3 experiment.py --flight auto --scenario landing          # 착륙 과제를 자동 조종으로
    python3 experiment.py --participant P001 --scenario full
    python3 experiment.py --participant P002 --scenario landing --repeats 3
    python3 experiment.py --participant P003 --scenario circuit    # 이륙부터 착륙까지 이어서 비행
    python3 experiment.py --participant P003 --scenario route      # 복잡한 경로 + 상승·강하 (500 m까지)
    python3 experiment.py --participant P004 --env rough           # 강한 바람과 난기류
    python3 experiment.py --participant P005 --env off             # 바람·물리 효과 없는 처음의 단순 모델

환경 (--env): 물리 효과(뱅크 선회, 선회 중 고도 손실, 에너지 교환, 추력 한계, 실속, 지면 효과)와
바람·난기류. off = 단순 모델, calm = 물리 효과만, normal = 약한 바람(기본), rough = 강한 바람.
어떤 환경에서 수행했는지는 trials.csv의 environment 열에 기록된다.

손 조종
    좌우 기울임 = 선회,  펼친 손가락 개수 = 스로틀(0~5개 -> 0~100%)
    손바닥 앞뒤 기울임 = 상승·하강: 손끝을 내 쪽으로 젖히면 상승, 카메라 쪽으로 숙이면 하강
    (방향이 반대로 느껴지면 --invert-pitch)

기체 계통 (키보드): F = 플랩 다음 단계(UP -> 1 -> 2 -> FULL), G = 착륙 장치 올림/내림.
    건드리지 않으면 플랩 UP·착륙 장치 자동으로, 계통이 없던 때와 똑같이 비행한다.

교관석 기능 (자유 비행 모드에서만): P 일시 정지, X 엔진 고장, T 돌풍 주입,
    R 활주로에서 다시 시작, L 최종 접근 3 km 지점, H 공항 상공 300 m.

화면 안내 문구는 영어다 (OpenCV 기본 글꼴은 한글을 그리지 못함).
화면 시점 (기본은 3D 추적 화면)
    마우스 왼쪽 드래그 / 트랙패드 두 손가락 스크롤 / 방향키 / 4 6 8 2   카메라 회전
    + - / Ctrl(또는 Option) + 스크롤                                   확대 / 축소
    1 뒤 / 3 옆 / 7 위 / 9 앞             미리 정한 시점,  5 또는 0 = 기본 시점
    V                                     3D <-> 2D 지도 전환
    A 또는 C, 화면 위 시점 버튼 클릭       조종사 시점 <-> 외부 시점 (--cam pilot 으로 조종사 시점에서 시작)
    N 또는 화면 위 DAY/NIGHT 버튼 클릭     낮 <-> 밤 전환 (--time night 으로 밤에서 시작)
    W 또는 그 옆 날씨 버튼 클릭           날씨 전환 (--weather rain 처럼 시작 날씨 지정)

종료: 창의 닫기 버튼(macOS는 창 왼쪽 위 빨간 버튼), q 또는 ESC, 터미널에서 Ctrl+C.
중단하면 진행 중이던 시도는 outcome=aborted로 기록된다.
"""

import argparse
import csv
import math
import random
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from flight_sim import (DEFAULT_WIND_FROM, ENVIRONMENTS, SCENARIO_ALL, SCENARIO_ORDER, SCENARIOS,
                        Aircraft, Environment, FreeFlight)
from autopilot import Autopilot
import airway
import cockpit as ck
from cockpit import FlightEffects
from hand_tracking import (FlightController, draw_hand, ensure_model, forbid_frame_saving)
from view3d import WEATHERS, View, normalize_key, render

CONSENT_TEXT = """
==================================================================
                      실험 참여 동의 안내
==================================================================
본 실험은 손 제스처로 가상 비행기를 조종하는 방식의 사용성을
평가합니다. 이륙, 직선 비행, 원형 선회, 착륙, 복잡한 경로 비행(상승·강하 포함) 과제를 수행합니다.

[수집하는 정보]
  - 익명 참가자 코드 ({participant})
  - 과제 수행 기록: 시작 시각, 반응 시간, 목표 도달 시간,
    경로 오차, 착륙 성공 여부, 손 인식 실패 비율
  - 비행 궤적: 과제 수행 중 0.1초 간격의 기체 위치·고도·속도와 조종 입력 값
    (숫자만 기록하며, 손 모양이나 영상은 포함되지 않습니다. tracks 폴더에 저장)

[수집하지 않는 정보]
  - 이름, 연락처 등 개인을 식별할 수 있는 정보
  - 얼굴 또는 웹캠 영상·사진: 웹캠 영상은 손 위치 인식을 위해
    메모리에서만 처리되며 디스크에 저장되지 않습니다.

[참가자의 권리]
  - 참여는 자발적이며, 언제든지 창을 닫거나 q 또는 ESC 키로 중단할 수 있고
    중단해도 불이익이 없습니다.
  - 기록은 {output} 파일에 저장되며, 참가자 코드를 알려주시면
    본인 기록의 삭제를 요청할 수 있습니다.
==================================================================
"""
CONSENT_YES = {"y", "yes", "예", "네", "동의"}

CSV_FIELDS = [
    "participant_id", "session_id", "mode", "scenario", "trial_no",
    "start_time",           # GO 신호가 표시된 시각 (ISO 8601, 로컬 시간)
    "reaction_time_s",      # GO 신호 후 첫 조작까지: GO 순간 대비 roll 또는 pitch가 ROLL_CHANGE 이상 바뀌거나 스로틀 단계 변경
    "target_time_s",        # GO 후 목표 도달까지 (착륙은 접지까지), 미도달이면 빈칸
    "path_error_m",         # 목표 경로와의 평균 수평 편차 (시간 가중 평균)
    "outcome",              # success / fail / timeout / aborted
    "fail_reason",
    "landing_success",      # 착륙·장주 시나리오만: True / False
    "touchdown_sink_ms",    # 접지 하강률
    "touchdown_speed_ms",   # 접지 속도
    "touchdown_offset_m",   # 접지 시 활주로 중심선 이탈 (+ 오른쪽)
    "hand_lost_pct",        # 비행 중 손이 인식되지 않은 프레임 비율
    "environment",          # off / calm / normal / rough (flight_sim.ENVIRONMENTS)
    "wind_speed_ms",        # 평균 풍속 (off·calm이면 0)
    "wind_from_deg",        # 바람이 불어오는 방위 (바람이 없으면 빈칸)
    "lighting",             # day / night (시도가 끝난 시점의 화면)
    "weather",              # clear / cloudy / rain / storm / fog / snow (시도가 끝난 시점의 화면)
    "alt_error_m",          # route 과제만: 목표 고도와의 평균 차이 (절댓값, 시간 가중 평균)
]
# 나중에 추가한 열과, 그 열이 없던 예전 기록에 채울 값 (예전 기록은 단순 모델·낮 화면이었다)
ADDED_FIELDS = {"environment": "off", "wind_speed_ms": 0.0, "wind_from_deg": "", "lighting": "day",
                "weather": "clear", "alt_error_m": ""}

PARTICIPANT_RE = re.compile(r"^[A-Za-z]{1,4}\d{1,4}$")   # 예: P001, S12
READY_DELAY = (1.5, 3.5)   # GO 신호 전 무작위 대기 (초): 예측해서 미리 움직이는 것 방지
RESULT_SECONDS = 2.5
ROLLOUT_SECONDS = 5.0     # 지상에서 끝난 시도는 결과 화면 동안 활주로 감속 장면을 보여준다
ROLLOUT_DECEL = 4.0       # 그때의 감속도 (m/s^2)
ROLL_CHANGE = 0.1         # 반응으로 인정하는 roll·pitch 변화량


class GestureInput:
    """웹캠 + MediaPipe 손 인식 -> FlightController(roll, pitch, throttle)."""

    def __init__(self, args):
        options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=args.min_confidence,
            min_hand_presence_confidence=args.min_confidence,
            min_tracking_confidence=args.min_confidence,
        )
        self.cap = cv2.VideoCapture(args.camera)
        if not self.cap.isOpened():
            raise SystemExit(f"카메라 {args.camera}번을 열 수 없습니다. "
                             "(macOS: 시스템 설정 > 개인정보 보호 > 카메라 권한 확인)")
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.ctrl = FlightController(args.smooth, args.deadzone, args.max_roll,
                                     args.pitch_deadzone, args.max_pitch, args.invert_pitch)
        self.start = time.monotonic()
        self.last_ts = -1
        # 카메라 읽기와 손 인식은 별도 스레드에서 돈다. 화면은 카메라 속도(어두우면 15 fps까지 떨어진다)에
        # 묶이지 않고 FRAME_SECONDS 간격으로 그려지고, 조종 입력은 가장 최근 인식 결과를 쓴다.
        self._raw = FlightController(args.smooth, args.deadzone, args.max_roll,
                                     args.pitch_deadzone, args.max_pitch, args.invert_pitch)
        self._latest = None      # (미리보기 프레임, 손 인식 여부)
        self._failed = False
        self._stop = threading.Event()
        self._next = self._last_read = time.monotonic()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    FRAME_SECONDS = 1 / 60   # 화면을 그리는 최소 간격
    INPUT_TAU = 0.08         # 조종 입력이 새 인식 결과를 따라가는 시간 상수 (s): 계단처럼 튀지 않게
    FIRST_FRAME_TIMEOUT = 10.0

    def _capture_loop(self):
        while not self._stop.is_set():
            got = self._capture()
            if got is None:
                self._failed = True
                return
            self._latest = got

    def _capture(self):
        """카메라 한 프레임을 읽어 손을 인식한다. (미리보기 프레임, 손 인식 여부), 카메라 오류 시 None."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # VIDEO 모드는 타임스탬프가 반드시 증가해야 함
        ts = max(int((time.monotonic() - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        h, w = frame.shape[:2]
        if result.hand_landmarks:
            lms, side = result.hand_landmarks[0], result.handedness[0][0].category_name
            world = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
            self._raw.update(lms, side, w, h, world)
            draw_hand(frame, lms, side)
            return frame, True
        self._raw.lost()
        return frame, False

    def read(self):
        """(미리보기 프레임, 손 인식 여부) 반환. 카메라 오류 시 None.

        새 카메라 프레임을 기다리지 않고 가장 최근 것을 돌려준다. 조종 입력(self.ctrl)은
        최근 인식 결과 쪽으로 부드럽게 따라간다.
        """
        deadline = time.monotonic() + self.FIRST_FRAME_TIMEOUT
        while self._latest is None and not self._failed and time.monotonic() < deadline:
            time.sleep(0.01)   # 첫 프레임을 기다린다
        if self._failed or self._latest is None:
            return None
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        now = time.monotonic()
        self._next = max(self._next + self.FRAME_SECONDS, now)
        k = 1.0 - math.exp(-min(now - self._last_read, 0.1) / self.INPUT_TAU)
        self._last_read = now
        raw, ctrl = self._raw, self.ctrl
        ctrl.roll += (raw.roll - ctrl.roll) * k
        ctrl.pitch += (raw.pitch - ctrl.pitch) * k
        ctrl.angle += (raw.angle - ctrl.angle) * k
        ctrl.pitch_angle += (raw.pitch_angle - ctrl.pitch_angle) * k
        ctrl.throttle = raw.throttle
        ctrl._calib = raw._calib   # 화면에 "중립 재는 중" 표시용
        return self._latest

    def recenter(self):
        """Z 키: 지금 손 자세를 중립으로 다시 잡는다."""
        self._raw.recalibrate()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.landmarker.close()
        self.cap.release()


MODES = {"gesture": GestureInput}


class AutoInput:
    """자동 조종 입력: 카메라 없이 컴퓨터(Autopilot)가 roll과 스로틀을 정한다."""
    FRAME_SECONDS = 1 / 30   # 카메라가 없으므로 화면 갱신 속도를 직접 맞춘다

    def __init__(self, args):
        self.ctrl = FlightController(args.smooth, args.deadzone, args.max_roll)
        self.pilot = Autopilot()
        self._next = time.monotonic()

    def read(self):
        """(미리보기 프레임 없음, 손 인식 표시는 항상 정상) 반환."""
        delay = self._next - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._next = max(self._next + self.FRAME_SECONDS, time.monotonic())
        return None, True

    def fly(self, sc, ac):
        """현재 상황을 보고 조종 입력을 정한다. 시도 루프가 프레임마다 부른다."""
        self.ctrl.roll, self.ctrl.throttle = self.pilot.control(sc, ac)
        self.ctrl.pitch = self.pilot.pitch(sc, ac)
        self.ctrl.angle = self.ctrl.roll * self.ctrl.max_roll

    def new_trial(self):
        self.pilot = Autopilot()

    def close(self):
        pass


# ---------------------------------------------------------------- 창과 입력

WINDOW_NAME = "Gesture Flight Experiment"
_view = View()   # 카메라 시점: 시도가 바뀌어도 유지된다
_keys = []       # 시점 조작이 아닌 키 입력 (모드별 기능 키용)


def _on_mouse(event, x, y, _flags, _param):
    _view.on_mouse(event, x, y, _flags)


def open_window():
    """창을 만들고 마우스 시점 조작을 연결."""
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)


def user_wants_quit() -> bool:
    """창 닫기 버튼 또는 q/ESC 키면 True. 시점 조작 키도 여기서 처리."""
    key = normalize_key(cv2.waitKeyEx(1))   # 방향키를 구분하려면 waitKeyEx가 필요
    if key != -1:
        if key & 0xFF in (ord("q"), ord("Q"), 27) and key < 256:
            return True
        if not _view.on_key(key):
            _keys.append(key)
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


# ---------------------------------------------------------------- 시도 1회

def coast(ac, ctrl, dt, env=None):
    """판정이 끝난 뒤 결과 화면 동안 기체를 계속 움직인다 (표시 전용, 기록에는 영향 없음).

    지상이면 제동하며 직진으로 굴러가고, 공중이면 조종 입력대로 계속 난다.
    """
    if ac.airborne:
        return ac.step(ctrl.roll, ctrl.throttle, dt, env, ctrl.pitch)
    ac.speed = max(ac.speed - ROLLOUT_DECEL * dt, 0.0)
    ac.ground_speed = ac.speed
    h = np.radians(ac.heading)
    ac.x += ac.speed * np.sin(h) * dt
    ac.y += ac.speed * np.cos(h) * dt
    ac.vs = 0.0
    return None


TRACK_SECONDS = 0.1   # 비행 궤적을 남기는 간격 (초)
TRACK_FIELDS = ["t_s", "x_m", "y_m", "alt_m", "heading_deg", "speed_ms", "vs_ms", "bank_deg",
                "roll_input", "pitch_input", "throttle_pct", "path_error_m", "target_alt_m", "hand"]


def track_sample(t, sc, ac, ctrl, hand) -> dict:
    """비행 궤적 표본 하나: 위치·자세·조종 입력. 웹캠 영상이나 손 좌표는 들어가지 않는다."""
    target = getattr(sc, "target_alt", None)
    return {"t_s": round(t, 2), "x_m": round(ac.x, 1), "y_m": round(ac.y, 1), "alt_m": round(ac.alt, 1),
            "heading_deg": round(ac.heading % 360.0, 1), "speed_ms": round(ac.speed, 1),
            "vs_ms": round(ac.vs, 2), "bank_deg": round(ac.bank, 1),
            "roll_input": round(ctrl.roll, 3), "pitch_input": round(ctrl.pitch, 3),
            "throttle_pct": ctrl.throttle, "path_error_m": round(sc.path_error(ac), 1),
            "target_alt_m": "" if target is None else round(target, 1), "hand": int(bool(hand))}


def save_track(folder: Path, row: dict, track: list) -> Path:
    """시도 하나의 궤적을 CSV로 저장한다. 파일 이름으로 trials.csv의 행과 짝지을 수 있다."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (f"{row['participant_id']}_{row['session_id']}_"
                     f"{int(row['trial_no']):03d}_{row['scenario']}.csv")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACK_FIELDS)
        writer.writeheader()
        writer.writerows(track)
    return path


def run_trial(inp, name, info, rng):
    """시나리오 1회 수행. (CSV 행 또는 None, 중단 여부) 반환."""
    sc = SCENARIOS[name]()
    ac = sc.initial_aircraft()
    trail = [(ac.x, ac.y, ac.alt)]   # 3D 화면에서 궤적을 고도까지 그리기 위해 고도도 저장
    ctrl = inp.ctrl
    # 환경(바람·난기류·물리 효과). off면 None이고 처음의 단순 모델로 움직인다
    env = None
    if ENVIRONMENTS[info["env"]] is not None:
        env = Environment(info["env"], info["wind_from"], seed=rng.random())
    fx = FlightEffects(ac, ctrl, env, info.get("motion", True))   # 이륙·착륙 연출, 움직임 효과 (표시 전용)
    # 조종 안내 십자선: 사람이 조종할 때만 (자동 조종은 안내가 필요 없다)
    guide = Autopilot() if info.get("guide", True) and not hasattr(inp, "fly") else None
    info["fd"] = None
    track = info["track"] = []   # 비행 궤적 표본 (TRACK_SECONDS 간격)

    phase = "ready"
    ready_until = time.monotonic() + rng.uniform(*READY_DELAY)
    go_time = start_iso = base_roll = base_pitch = base_throttle = reaction = result_until = None
    err_sum = alt_sum = run_time = 0.0
    frames = lost = 0
    t = 0.0
    last = time.monotonic()
    quit_requested = False

    try:
        while True:
            got = inp.read()
            if got is None:
                print("프레임을 읽지 못했습니다.")
                quit_requested = True
                break
            cam, hand = got
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now
            event = None
            if hasattr(inp, "fly"):   # 자동 조종이면 컴퓨터가 조종 입력을 정한다
                inp.fly(sc, ac)

            if phase == "ready" and now >= ready_until:
                phase, go_time = "run", now
                start_iso = datetime.now().isoformat(timespec="milliseconds")
                base_roll, base_pitch, base_throttle = ctrl.roll, ctrl.pitch, ctrl.throttle
            elif phase == "run":
                t = now - go_time
                if reaction is None and (abs(ctrl.roll - base_roll) >= ROLL_CHANGE
                                         or abs(ctrl.pitch - base_pitch) >= ROLL_CHANGE
                                         or ctrl.throttle != base_throttle):
                    reaction = t
                event = ac.step(ctrl.roll, ctrl.throttle, dt, env, ctrl.pitch)
                sc.update(ac, event, t)
                if guide is not None:
                    want_roll, want_pitch = guide.guidance(sc, ac)
                    info["fd"] = (want_roll - ctrl.roll, want_pitch - ctrl.pitch)
                if not track or t - track[-1]["t_s"] >= TRACK_SECONDS:
                    track.append(track_sample(t, sc, ac, ctrl, hand))
                err_sum += sc.path_error(ac) * dt
                if hasattr(sc, "alt_error"):
                    alt_sum += abs(sc.alt_error(ac)) * dt
                run_time += dt
                frames += 1
                lost += not hand
                if np.hypot(ac.x - trail[-1][0], ac.y - trail[-1][1]) > 5:
                    trail.append((ac.x, ac.y, ac.alt))
                if not sc.done and t >= sc.timeout:
                    sc.finish("timeout", "time limit")
                if sc.done:
                    info["fd"] = None
                    track.append(track_sample(t, sc, ac, ctrl, hand))
                    phase = "result"
                    result_until = now + (RESULT_SECONDS if ac.airborne else ROLLOUT_SECONDS)
            elif phase == "result":
                if now >= result_until:
                    break
                event = coast(ac, ctrl, dt, env)
                if np.hypot(ac.x - trail[-1][0], ac.y - trail[-1][1]) > 5:
                    trail.append((ac.x, ac.y, ac.alt))
            fx.update(ac, ctrl, event, dt)

            cv2.imshow(WINDOW_NAME,
                       render(_view, sc, ac, trail, ctrl, cam, hand, phase, t, info, fx))
            if user_wants_quit():
                quit_requested = True
                break
            while _keys:
                system_key(_keys.pop(), ac, inp)
    except KeyboardInterrupt:   # 터미널에서 Ctrl+C
        quit_requested = True

    if phase == "ready":   # GO 전에 중단되면 시작하지 않은 시도이므로 기록하지 않음
        return None, quit_requested
    sc.finish("aborted", "user quit")

    td = sc.touchdown or {}
    fmt = lambda v, n=3: "" if v is None else round(v, n)
    row = {
        "participant_id": info["participant"],
        "session_id": info["session_id"],
        "mode": info["mode"],
        "scenario": name,
        "trial_no": info["trial_no"],
        "start_time": start_iso,
        "reaction_time_s": fmt(reaction),
        "target_time_s": fmt(sc.reached_time),
        "path_error_m": fmt(err_sum / run_time if run_time > 0 else None, 2),
        "outcome": sc.outcome,
        "fail_reason": sc.reason if sc.outcome != "success" else "",
        "landing_success": "" if sc.landing_success is None else sc.landing_success,
        "touchdown_sink_ms": fmt(td.get("sink"), 2),
        "touchdown_speed_ms": fmt(td.get("speed"), 2),
        "touchdown_offset_m": fmt(td.get("offset"), 2),
        "hand_lost_pct": fmt(100.0 * lost / frames if frames else None, 1),
        "environment": info["env"],
        "wind_speed_ms": env.wind_speed if env is not None else 0.0,
        "wind_from_deg": env.wind_from if env is not None and env.wind_speed > 0 else "",
        "lighting": "night" if _view.night else "day",
        "weather": _view.weather,
        "alt_error_m": fmt(alt_sum / run_time, 1) if hasattr(sc, "alt_error") and run_time > 0 else "",
    }
    return row, quit_requested


# ---------------------------------------------------------------- 자유 비행

FREE_NOTICE_SECONDS = 5.0   # 접지 결과를 보여주는 시간
FREE_TRAIL_POINTS = 2000    # 궤적은 최근 것만 남긴다 (약 10 km)


def system_key(key, ac, inp=None) -> bool:
    """모든 비행 모드 공통 키: F = 플랩 다음 단계, G = 착륙 장치 올림/내림, Z = 손 중립 다시 잡기."""
    if key in (ord("z"), ord("Z")):
        if hasattr(inp, "recenter"):
            inp.recenter()
    elif key in (ord("f"), ord("F")):
        ac.next_flaps()
    elif key in (ord("g"), ord("G")):
        ac.toggle_gear()
    else:
        return False
    return True


# 교관석에서 기체를 옮겨 놓을 수 있는 위치: 키 -> (설명, x, y, 방위, 속도, 고도)
REPOSITIONS = {
    ord("l"): ("최종 접근 3 km", 0.0, -3000.0, 0.0, 36.0, 160.0),
    ord("h"): ("공항 상공 300 m", 0.0, -800.0, 0.0, 36.0, 300.0),
}
GUST_KICK = (7.0, 0.0, -8.0)   # 교관석의 돌풍 주입: 돌풍 성분(동, 북, 위)에 더하는 양 (m/s)


def run_free(inp, info, rng):
    """자유 비행: 과제 없이 창을 닫을 때까지 난다. 아무것도 기록하지 않는다.

    훈련 시뮬레이터의 교관석(IOS)처럼 일시 정지, 위치 재배치, 비상 상황 주입을 키로 할 수 있다.
        P 일시 정지(프리즈)   X 엔진 고장 걸기/풀기   T 강한 돌풍 주입
        R 활주로에서 다시 시작   L 최종 접근 3 km 지점으로   H 공항 상공 300 m로
    """
    def start():
        sc = FreeFlight()
        ac = sc.initial_aircraft()
        env = None
        if ENVIRONMENTS[info["env"]] is not None:
            env = Environment(info["env"], info["wind_from"], seed=rng.random())
        return (sc, ac, env, FlightEffects(ac, inp.ctrl, env, info.get("motion", True)),
                [(ac.x, ac.y, ac.alt)])

    def reposition(key):
        """기체를 공중의 정해진 위치로 옮긴다 (계통 상태와 날씨는 그대로)."""
        _name, x, y, heading, speed, alt = REPOSITIONS[key]
        ac.x, ac.y, ac.heading, ac.speed, ac.alt = x, y, heading, speed, alt
        ac.airborne, ac.vs, ac.bank, ac.ground_speed, ac.stalled = True, 0.0, 0.0, speed, False
        trail[:] = [(ac.x, ac.y, ac.alt)]
        fx.touchdown = fx.liftoff = None
        fx._prev_alt, fx._prev_speed = ac.alt, ac.speed

    ctrl = inp.ctrl
    sc, ac, env, fx, trail = start()
    t = 0.0
    notice_until = 0.0
    landings = 0   # 다시 시작(R)해도 이어서 센다
    frozen = False
    gust_until = 0.0
    last = time.monotonic()
    try:
        while True:
            got = inp.read()
            if got is None:
                print("프레임을 읽지 못했습니다.")
                break
            cam, hand = got
            now = time.monotonic()
            dt = min(now - last, 0.1)
            last = now

            if not frozen:   # 프리즈 중에는 시간이 멈춘다 (화면과 시점 조작은 계속된다)
                event = ac.step(ctrl.roll, ctrl.throttle, dt, env, ctrl.pitch)
                t += dt
                sc.update(ac, event, t)
                if event == "touchdown":
                    landings += 1
                    notice_until = now + FREE_NOTICE_SECONDS
                if np.hypot(ac.x - trail[-1][0], ac.y - trail[-1][1]) > 5:
                    trail.append((ac.x, ac.y, ac.alt))
                    del trail[:-FREE_TRAIL_POINTS]
                fx.update(ac, ctrl, event, dt)

            # 교관석 상태 표시: 지금 걸려 있는 상황들
            info["ios"] = [word for word, on in (("FREEZE", frozen), ("ENG FAIL", ac.engine_failed),
                                                 ("GUST", now < gust_until)) if on]
            phase = "result" if now < notice_until else "run"   # 접지 직후에만 결과를 띄운다
            cv2.imshow(WINDOW_NAME,
                       render(_view, sc, ac, trail, ctrl, cam, hand, phase, t, info, fx))
            if user_wants_quit():
                break
            while _keys:
                key = _keys.pop()
                lower = key | 0x20 if 0 < key < 128 else key   # 대문자도 같은 키로
                if system_key(key, ac, inp):
                    pass
                elif lower == ord("r"):   # 활주로 출발 위치에서 다시 시작 (고장도 풀린다)
                    sc, ac, env, fx, trail = start()
                    t, notice_until, frozen = 0.0, 0.0, False
                elif lower == ord("p"):   # 프리즈
                    frozen = not frozen
                elif lower == ord("x"):   # 엔진 고장 걸기 / 풀기
                    ac.engine_failed = not ac.engine_failed
                elif lower == ord("t"):   # 강한 돌풍 (옆바람 + 하강 기류). 몇 초에 걸쳐 잦아든다
                    if env is not None:
                        for i, kick in enumerate(GUST_KICK):
                            env.gust[i] += kick
                        gust_until = now + 4.0
                elif lower in REPOSITIONS:
                    reposition(lower)
                    notice_until = 0.0
    except KeyboardInterrupt:   # 터미널에서 Ctrl+C
        pass
    return landings


def append_row(path: Path, row: dict):
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        missing = CSV_FIELDS[len(header):]
        if header == CSV_FIELDS[:len(header)] and missing and set(missing) <= set(ADDED_FIELDS):
            upgrade_csv(path, missing)   # 열이 추가되기 전에 만든 파일
        elif header != CSV_FIELDS:
            raise SystemExit(f"{path}의 열 구성이 현재 형식과 다릅니다. 다른 --output을 지정하세요.")
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def upgrade_csv(path: Path, missing: list):
    """예전 형식의 기록 파일에 나중에 생긴 열(missing)을 추가한다.

    그 열이 없던 때의 기록은 당시 조건(단순 모델 off, 낮 화면 day)으로 채운다.
    바꾸기 전 파일은 같은 폴더에 .bak 파일로 남긴다 (이미 있으면 번호를 붙인다).
    """
    backup = path.with_name(path.name + ".bak")
    n = 2
    while backup.exists():
        backup = path.with_name(f"{path.name}.bak{n}")
        n += 1
    backup.write_bytes(path.read_bytes())
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **{name: ADDED_FIELDS[name] for name in missing}})
    filled = ", ".join(f"{name}={ADDED_FIELDS[name]!r}" for name in missing)
    print(f"{path.name}: 예전 형식이라 열을 추가했습니다 (기존 {len(rows)}개 기록은 {filled}, "
          f"바꾸기 전 파일은 {backup.name}).")


def next_trial_no(path: Path, participant: str) -> int:
    """같은 참가자의 기존 기록 다음 번호부터 시작 (여러 번 실행해도 번호가 이어짐)."""
    if not path.exists():
        return 1
    with path.open(newline="", encoding="utf-8") as f:
        nums = [int(r["trial_no"]) for r in csv.DictReader(f)
                if r.get("participant_id") == participant and r.get("trial_no", "").isdigit()]
    return max(nums, default=0) + 1


PARTICIPANT_HELP = "참가자 ID는 익명 코드 형식이어야 합니다 (영문 1~4자 + 숫자 1~4자리, 예: P001)."


def ask_participant() -> str:
    """--participant 없이 실행한 경우(예: VS Code ▶ 버튼) 터미널에서 익명 코드를 입력받는다."""
    while True:
        try:
            code = input("익명 참가자 코드를 입력하세요 (예: P001, 이름은 넣지 마세요): ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n참가자 코드가 없어 실험을 시작하지 않습니다.")
        if PARTICIPANT_RE.match(code):
            return code
        print(PARTICIPANT_HELP)


def ask_consent(participant: str, output: Path) -> bool:
    """동의 안내문을 출력하고 참가자의 명시적 동의를 받는다. 동의하지 않으면 False."""
    print(CONSENT_TEXT.format(participant=participant, output=output.name))
    try:
        answer = input("위 내용을 읽고 실험 참여에 동의하십니까? (y/n): ").strip().lower()
    except EOFError:   # 입력을 받을 수 없는 환경이면 동의하지 않은 것으로 처리
        answer = ""
    return answer in CONSENT_YES


FLIGHT_MODES = {
    "staged": "단계별 시뮬레이션 - 이륙 → 직선 → 선회 → 착륙 과제를 차례로 수행 (실험 기록 저장)",
    "free": "자유 비행         - 과제 없이 손으로 자유롭게 비행 (기록 없음)",
    "auto": "자동 조종         - 컴퓨터가 실제 항로(예: 인천 → 로스앤젤레스)를 비행, 지켜보기 (카메라·기록 없음)",
}
VIEW_HELP = ("화면 시점: 마우스 드래그·트랙패드 두 손가락 스크롤·방향키(또는 4 6 8 2) = 회전,\n"
             "           +/- 또는 Ctrl+스크롤 = 확대/축소,\n"
             "           1 3 7 9 = 미리 정한 시점, 5 = 기본 시점, V = 3D/2D 전환,\n"
             "낮/밤 전환: N 키 또는 화면 위 가운데의 DAY/NIGHT 버튼 클릭\n"
             "날씨 전환: W 키 또는 그 옆의 날씨 버튼 클릭\n"
             "조종석 시점: C 키 (뒤에서 따라가는 시점과 전환, 조종석에서는 드래그로 고개 돌리기)\n"
             "기체 계통: F = 플랩 다음 단계, G = 착륙 장치 올림/내림\n"
             "손 중립 다시 잡기: 손을 편하게 든 자세에서 Z 키 (시작할 때 한 번은 자동으로 잡습니다)\n"
             "종료: 창의 닫기 버튼, q 또는 ESC\n")


def ask_flight_mode() -> str:
    """--flight 없이 실행하면 터미널에서 비행 모드를 고르게 한다."""
    names = list(FLIGHT_MODES)
    print("비행 모드를 고르세요")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {FLIGHT_MODES[name]}")
    while True:
        try:
            answer = input(f"번호 입력 (1~{len(names)}): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n비행 모드를 고르지 않아 시작하지 않습니다.")
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f"1~{len(names)} 중에서 골라 주세요.")


TIMES = {"day": "낮", "night": "밤 (활주로 등화, 기체 등, 별과 달)"}
WEATHER_NAMES = {"clear": "맑음", "cloudy": "흐림 (구름 많음)", "rain": "비", "storm": "뇌우 (폭우와 번개)",
                 "fog": "안개 (시야 매우 나쁨)", "snow": "눈"}


def ask_weather() -> str:
    """--weather 없이 실행하면 터미널에서 날씨를 고르게 한다."""
    names = list(WEATHER_NAMES)
    print("날씨를 고르세요 (실행 중에도 W 키로 바꿀 수 있습니다)")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {WEATHER_NAMES[name]}")
    while True:
        try:
            answer = input(f"번호 입력 (1~{len(names)}, 그냥 Enter = 맑음): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n날씨를 고르지 않아 시작하지 않습니다.")
        if answer == "":
            return "clear"
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f"1~{len(names)} 중에서 골라 주세요.")


BUNDLES = {"full": SCENARIO_ORDER, "all": SCENARIO_ALL}   # 여러 과제를 차례로 하는 묶음

CAMS = {"pilot": "조종사 시점 (조종석에서 앞을 봄)", "outside": "외부 시점 (기체 뒤에서 따라가며 봄)"}


def ask_cam() -> str:
    """--cam 없이 실행하면 터미널에서 시점을 고르게 한다."""
    names = list(CAMS)
    print("시점을 고르세요 (실행 중에도 A 키나 화면 위 버튼으로 바꿀 수 있습니다)")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {CAMS[name]}")
    while True:
        try:
            answer = input(f"번호 입력 (1~{len(names)}, 그냥 Enter = 외부 시점): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n시점을 고르지 않아 시작하지 않습니다.")
        if answer == "":
            return "outside"
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f"1~{len(names)} 중에서 골라 주세요.")


def ask_time() -> str:
    """--time 없이 실행하면 터미널에서 낮/밤을 고르게 한다."""
    names = list(TIMES)
    print("\n시간대를 고르세요 (실행 중에도 N 키로 바꿀 수 있습니다)")
    for i, name in enumerate(names, 1):
        print(f"  {i}) {TIMES[name]}")
    while True:
        try:
            answer = input(f"번호 입력 (1~{len(names)}, 그냥 Enter = 낮): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n시간대를 고르지 않아 시작하지 않습니다.")
        if answer == "":
            return "day"
        if answer in names:
            return answer
        if answer in ("낮", "밤"):
            return "day" if answer == "낮" else "night"
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print(f"1~{len(names)} 중에서 골라 주세요.")


def print_environment(args):
    wind, gust = ENVIRONMENTS[args.env] or (0.0, 0.0)
    print(f"환경: {args.env}" + (f" (바람 {args.wind_from:03.0f}도에서 {wind:.0f} m/s, 돌풍 ±{gust:.1f} m/s)"
                                if wind > 0 else "") + "\n")


def close_window():
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # macOS에서 창이 실제로 닫히도록 이벤트 한 번 처리


# ---------------------------------------------------------------- 손동작 연습과 결과 요약

PRACTICE_HOLD = 1.0   # 동작을 이만큼(초) 유지하면 다음 단계로
# (제목, 설명, 통과 조건). 화면 글자는 영어다 (OpenCV 기본 글꼴은 한글을 그리지 못함)
PRACTICE_STEPS = [
    ("HOLD YOUR HAND UP", "Open hand, relaxed, facing the camera. Keep it still.",
     lambda c: not c.calibrating and abs(c.roll) < 0.15 and abs(c.pitch) < 0.3),
    ("TILT RIGHT", "Lean your hand to the right: the aircraft turns right.", lambda c: c.roll > 0.6),
    ("TILT LEFT", "Lean your hand to the left: the aircraft turns left.", lambda c: c.roll < -0.6),
    ("TILT PALM BACK", "Fingertips toward you: the aircraft climbs.", lambda c: c.pitch > 0.5),
    ("TILT PALM FORWARD", "Fingertips toward the camera: the aircraft descends.",
     lambda c: c.pitch < -0.5),
    ("MAKE A FIST", "No fingers = throttle 0%.", lambda c: c.throttle == 0),
    ("SHOW 2 FINGERS", "Two fingers = throttle 50% (level flight).", lambda c: c.throttle == 50),
    ("OPEN YOUR HAND", "Four or five fingers = throttle 100%.", lambda c: c.throttle == 100),
]
SKIP_STEP_KEYS = (32,)        # Space: 이 동작 건너뛰기
SKIP_ALL_KEYS = (13, 10, 3)   # Enter: 연습 전체 건너뛰기


def draw_practice(cam, hand, ctrl, step, hold, done):
    """연습 화면: 왼쪽에 웹캠 미리보기(저장하지 않음), 오른쪽에 조종 입력, 아래에 지금 할 동작."""
    img = np.full((ck.CANVAS_H, ck.CANVAS_W, 3), ck.PANEL, np.uint8)
    ck.text(img, "HAND GESTURE PRACTICE", (24, 40), 0.9, ck.WHITE, 2)
    ck.text(img, "NOT RECORDED", (ck.CANVAS_W - 24, 40), 0.5, ck.GRAY, 1, "r")
    if cam is not None:
        img[64:64 + 420, 24:24 + 560] = cv2.resize(cam, (560, 420), interpolation=cv2.INTER_AREA)
    cv2.rectangle(img, (23, 63), (584, 484), ck.RED if not hand else ck.DIM, 2 if not hand else 1)
    if not hand:
        ck.text(img, "NO HAND - show your hand to the camera", (304, 474), 0.55, ck.WHITE, 2, "c")

    x = 624   # 조종 입력 (오른쪽)
    ck.draw_axis_input(img, x, 110, 340, "ROLL (tilt left / right)", ctrl.roll,
                       ctrl.deadzone / ctrl.max_roll, ("L", "R"), hand)
    ck.draw_axis_input(img, x, 190, 340, "PITCH (palm forward / back)", ctrl.pitch,
                       ctrl.pitch_deadzone / ctrl.max_pitch, ("DN", "UP"), hand)
    ck.draw_throttle(img, x, 272, 130, ctrl.throttle)
    ck.text(img, "THROTTLE = number of fingers", (x + 90, 300), 0.45, ck.WHITE)
    ck.text(img, "0 = 0%    1 = 25%    2 = 50%", (x + 90, 330), 0.42, ck.GRAY)
    ck.text(img, "3 = 75%    4 or 5 = 100%", (x + 90, 352), 0.42, ck.GRAY)

    for i, (title, _text, _ok) in enumerate(PRACTICE_STEPS):   # 단계 목록
        col, row = divmod(i, 4)
        color = ck.GREEN if i < step or done else ck.WHITE if i == step else ck.DIM
        mark = "OK " if i < step or done else ">  " if i == step else "   "
        ck.text(img, f"{mark}{i + 1}. {title}", (x + col * 185, 470 + row * 22), 0.4, color)

    ck.shade(img, 24, 560, ck.CANVAS_W - 24, 660, (60, 50, 40), 0.9)   # 지금 할 동작 (아래)
    cv2.rectangle(img, (24, 560), (ck.CANVAS_W - 24, 660), ck.DIM, 1)
    if done:
        ck.text(img, "PRACTICE COMPLETE", (ck.CANVAS_W // 2, 606), 1.0, ck.GREEN, 2, "c")
        ck.text(img, "The first task starts now.", (ck.CANVAS_W // 2, 640), 0.55, ck.WHITE, 1, "c")
    else:
        title, text, _ok = PRACTICE_STEPS[step]
        ck.text(img, f"{step + 1} / {len(PRACTICE_STEPS)}   {title}", (44, 598), 0.9, ck.WHITE, 2)
        ck.text(img, text, (44, 630), 0.55, ck.WHITE)
        cv2.rectangle(img, (44, 642), (ck.CANVAS_W - 44, 650), (70, 70, 70), -1)   # 유지 시간 막대
        cv2.rectangle(img, (44, 642), (44 + int((ck.CANVAS_W - 88) * min(hold / PRACTICE_HOLD, 1.0)), 650),
                      ck.GREEN, -1)
    ck.text(img, "SPACE: skip this step    ENTER: skip practice    Z: recenter hand    Q: quit",
            (ck.CANVAS_W // 2, 686), 0.42, ck.GRAY, 1, "c")
    return img


def run_practice(inp) -> bool:
    """기록 없이 손동작을 하나씩 따라 해 본다. 끝까지 했거나 건너뛰면 True, 종료를 원하면 False."""
    ctrl = inp.ctrl
    step, hold, done_until = 0, 0.0, None
    last = time.monotonic()
    try:
        while True:
            got = inp.read()
            if got is None:
                print("프레임을 읽지 못했습니다.")
                return False
            cam, hand = got
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            if done_until is None:
                ok = hand and PRACTICE_STEPS[step][2](ctrl)
                hold = hold + dt if ok else 0.0
                if hold >= PRACTICE_HOLD:
                    step, hold = step + 1, 0.0
            elif now >= done_until:
                return True
            cv2.imshow(WINDOW_NAME, draw_practice(cam, hand, ctrl, min(step, len(PRACTICE_STEPS) - 1),
                                                  hold, done_until is not None))
            if user_wants_quit():
                return False
            while _keys:
                key = _keys.pop()
                if key in SKIP_ALL_KEYS:
                    return True
                if key in SKIP_STEP_KEYS and done_until is None:
                    step, hold = step + 1, 0.0
                elif key in (ord("z"), ord("Z")) and hasattr(inp, "recenter"):
                    inp.recenter()
            if step >= len(PRACTICE_STEPS) and done_until is None:
                done_until = now + 1.5
    except KeyboardInterrupt:
        return False


OUTCOME_COLORS = {"success": ck.GREEN, "fail": ck.RED, "timeout": ck.AMBER}


def draw_summary(rows, info):
    """모든 과제를 마친 뒤의 결과 요약표."""
    img = np.full((ck.CANVAS_H, ck.CANVAS_W, 3), ck.PANEL, np.uint8)
    ck.text(img, "SESSION SUMMARY", (40, 56), 1.0, ck.WHITE, 2)
    ck.text(img, f"{info['participant']}   {str(info['mode']).upper()} CONTROL", (ck.CANVAS_W - 40, 56),
            0.55, ck.GRAY, 1, "r")
    cols = [("STAGE", 40, "l"), ("RESULT", 250, "l"), ("TIME s", 500, "r"), ("REACTION s", 630, "r"),
            ("PATH ERR m", 770, "r"), ("ALT ERR m", 900, "r")]
    for label, x, anchor in cols:
        ck.text(img, label, (x, 110), 0.5, ck.GRAY, 1, anchor)
    cv2.line(img, (40, 122), (ck.CANVAS_W - 40, 122), ck.DIM, 1)
    shown = rows[-13:]   # 화면에 들어가는 만큼만 (반복이 많을 때는 최근 것)
    blank = lambda v: "-" if v in ("", None) else str(v)
    for i, row in enumerate(shown):
        y = 156 + i * 34
        ck.text(img, f"{row['trial_no']}. {row['scenario'].upper()}", (40, y), 0.6, ck.WHITE, 1)
        ck.text(img, row["outcome"].upper(), (250, y), 0.6, OUTCOME_COLORS.get(row["outcome"], ck.GRAY), 2)
        for key, x in (("target_time_s", 500), ("reaction_time_s", 630), ("path_error_m", 770),
                       ("alt_error_m", 900)):
            ck.text(img, blank(row[key]), (x, y), 0.6, ck.WHITE, 1, "r")
        if row["fail_reason"]:
            ck.text(img, row["fail_reason"], (250, y + 15), 0.38, ck.GRAY)
    wins = sum(row["outcome"] == "success" for row in rows)
    y = 156 + len(shown) * 34 + 20
    cv2.line(img, (40, y - 26), (ck.CANVAS_W - 40, y - 26), ck.DIM, 1)
    ck.text(img, f"{wins} / {len(rows)} STAGES COMPLETED", (40, y + 8), 0.8,
            ck.GREEN if wins == len(rows) else ck.WHITE, 2)
    ck.text(img, "Press ENTER or SPACE, or close the window, to finish", (ck.CANVAS_W // 2, 676), 0.5,
            ck.GRAY, 1, "c")
    return img


def show_summary(inp, rows, info):
    """결과 요약표를 키를 누르거나 창을 닫을 때까지 보여준다."""
    img = draw_summary(rows, info)
    try:
        while True:
            if inp.read() is None:   # 화면 갱신 속도를 맞추는 용도 (영상은 쓰지 않는다)
                return
            cv2.imshow(WINDOW_NAME, img)
            if user_wants_quit():
                return
            while _keys:
                if _keys.pop() in SKIP_STEP_KEYS + SKIP_ALL_KEYS:
                    return
    except KeyboardInterrupt:
        return


def main_staged(args, parser):
    """단계별 시뮬레이션: 참가자 코드와 동의를 받고 과제를 차례로 수행해 기록한다."""
    if args.participant is None:
        args.participant = ask_participant()
    elif not PARTICIPANT_RE.match(args.participant):
        parser.error(PARTICIPANT_HELP)

    name = args.scenario or "all"   # 기본: 기본 과제 4개에 이어 복잡한 경로까지
    scenarios = BUNDLES.get(name, [name]) * (args.repeats or 1)
    rng = random.Random(args.seed)
    info = {
        "participant": args.participant.upper(),
        "mode": args.mode,
        "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "env": args.env,
        "wind_from": args.wind_from,
        "motion": not args.no_motion,
        "guide": not args.no_guide,
    }
    if not ask_consent(info["participant"], args.output):
        print("동의하지 않아 실험을 시작하지 않습니다. 카메라는 켜지지 않았고 기록도 남지 않았습니다.")
        return
    print("동의가 확인되었습니다. 실험을 시작합니다.\n")
    print(VIEW_HELP)
    print_environment(args)

    forbid_frame_saving()
    trial_no = next_trial_no(args.output, info["participant"])
    inp = MODES[args.mode](args)
    saved = 0
    rows = []
    track_dir = args.output.parent / "tracks"
    try:
        open_window()
        quit_requested = args.practice and not run_practice(inp)
        for name in scenarios:
            if quit_requested:
                break
            info["trial_no"] = trial_no
            row, quit_requested = run_trial(inp, name, info, rng)
            if row is not None:
                append_row(args.output, row)
                if not args.no_tracks and info["track"]:
                    save_track(track_dir, row, info["track"])
                rows.append(row)
                saved += 1
                trial_no += 1
                print(f"[{row['scenario']}] {row['outcome']}  RT={row['reaction_time_s']}s  "
                      f"time={row['target_time_s']}s  err={row['path_error_m']}m")
            if quit_requested:
                break
        if rows and not quit_requested:
            show_summary(inp, rows, info)
    finally:
        inp.close()
        close_window()
    print(f"{saved}개 시도를 {args.output}에 저장했습니다.")
    if saved and not args.no_tracks:
        print(f"비행 궤적은 {track_dir} 폴더에 시도별 파일로 저장했습니다.")


def main_free(args):
    """자유 비행: 손으로 조종하지만 과제도 기록도 없다."""
    print("자유 비행 모드입니다. 과제나 판정 없이 자유롭게 비행합니다.")
    print("웹캠 영상은 손 인식을 위해 메모리에서만 처리되며 저장되지 않습니다. 비행 기록도 남기지 않습니다.")
    print("교관석 기능 (키): P = 일시 정지,  X = 엔진 고장 걸기/풀기,  T = 강한 돌풍 주입\n"
          "                  R = 활주로에서 다시 시작,  L = 최종 접근 3 km 지점,  H = 공항 상공 300 m\n")
    print(VIEW_HELP)
    print_environment(args)
    info = {"participant": "FREE FLIGHT", "mode": args.mode, "subtitle": "NOT RECORDED",
            "trial_no": None, "env": args.env, "wind_from": args.wind_from, "hint": "R: RESTART",
            "ios": [], "motion": not args.no_motion}
    forbid_frame_saving()
    inp = MODES[args.mode](args)
    try:
        open_window()
        landings = run_free(inp, info, random.Random(args.seed))
    finally:
        inp.close()
        close_window()
    print(f"자유 비행을 마쳤습니다 (착륙 {landings}회). 기록은 저장하지 않았습니다.")


# ---------------------------------------------------------------- 실제 항로 비행 (자동 조종)

ENROUTE_LOW_ALT = 150.0        # 이 고도까지는 화면도 실제 고도 그대로 그린다 (m)
ENROUTE_ALT_SCALE = 0.051      # 그 위로는 화면의 고도를 줄여 그린다 (10,700 m -> 약 690 m)
ENROUTE_BLEND_SECONDS = 1.5    # 해안을 지날 때 바다와 육지가 바뀌는 데 걸리는 시간


class EnrouteView:
    """순항 구간의 화면 표시용 정보 (왼쪽 위 상자). 과제 판정은 없다."""
    timeout = None
    error_label = "DIST TO GO"
    instruction = ""
    reason = ""
    touchdown = outcome = None
    done = False

    def __init__(self, route):
        self.route = route
        self.name = f"{route.origin} > {route.dest}"
        self.km = 0.0            # 지나온 거리
        self.elapsed = 0.0       # 실제 기준 비행 시간 (초)

    @property
    def progress(self):
        return min(self.km / self.route.distance, 1.0)

    @property
    def time_text(self):
        return airway.format_hm(self.elapsed)

    def error_text(self, ac):
        return f"{max(self.route.distance - self.km, 0.0):,.0f} km"

    def path_error(self, ac):
        return 0.0

    def targets(self):
        return []


GUIDE_AHEAD = 14000.0   # 순항 화면에서 길 안내선을 그리는 거리 (화면 기준 m)
GUIDE_STEP = 500.0


def guide_path(route, km, ac, visual):
    """순항 화면의 길 안내선: 지금 위치에서 앞으로 이어지는 항로 (화면 좌표, 기체보다 조금 아래).

    화면은 축소해서 그리므로, 화면의 거리를 실제 거리로 바꿔 그 지점의 진행 방위를 따라 이어 간다.
    """
    real_per_visual = ac.alt / max(visual.alt, 1.0)   # 화면 1 m가 실제 몇 m인지
    z = visual.alt - 60.0   # 조종사 시점에서도 앞으로 뻗은 선으로 보이도록 기체보다 충분히 아래에 둔다
    x, y = visual.x, visual.y
    pts = [(x, y, z)]
    for i in range(int(GUIDE_AHEAD / GUIDE_STEP)):
        ahead_km = km + (i + 0.5) * GUIDE_STEP * real_per_visual / 1000.0
        if ahead_km >= route.distance:
            break
        h = math.radians(route.bearing(ahead_km / route.distance))
        x, y = x + GUIDE_STEP * math.sin(h), y + GUIDE_STEP * math.cos(h)
        pts.append((x, y, z))
    return pts


def run_enroute(inp, route, maps, info, rng) -> bool:
    """순항 구간: 대권 항로를 따라 실제 고도·속도로 날되 시간을 압축해 보여준다. 중단을 원하면 True.

    기체 상태(ac)는 실제 값(고도 10 km, 속도 250 m/s)이고 계기도 그 값을 보인다. 3D 화면은 고도를
    줄인 별도의 기체(visual)로 그린다: 실제 축척으로는 지면이 전부 안개에 묻혀 보이지 않기 때문이다.
    """
    sc = EnrouteView(route)
    ctrl = inp.ctrl
    ctrl.roll = ctrl.pitch = 0.0
    total = route.flight_seconds
    warp = total / route.show_seconds
    env = None
    if ENVIRONMENTS[info["env"]] is not None:
        env = Environment(info["env"], info["wind_from"], seed=rng.random())
    ac = Aircraft(x=0.0, y=0.0, heading=route.bearing(0.0), speed=route.airspeed(0.0),
                  alt=route.altitude(0.0), airborne=True)
    ac.physical = True
    visual = Aircraft(x=0.0, y=0.0, heading=ac.heading, speed=ac.speed, alt=ac.alt, airborne=True)
    fx = FlightEffects(ac, ctrl, env, info.get("motion", True))
    land = 1.0
    hint = info.get("hint")
    info["hint"] = f"CRUISE TIME x{warp:.0f}"
    last = time.monotonic()
    quit_requested = False
    try:
        while sc.km < route.distance:
            if inp.read() is None:
                quit_requested = True
                break
            now = time.monotonic()
            dt, last = min(now - last, 0.1), now
            # 이번 프레임에 흐른 실제 시간. 상승·강하 구간은 천천히 보여준다 (순항의 1/5 속도에서 시작)
            sim_dt = dt * warp * (0.2 + 0.8 * route.climb_progress(sc.km))
            sc.km = min(sc.km + route.ground_speed(sc.km) * sim_dt / 1000.0, route.distance)
            sc.elapsed += sim_dt
            frac = sc.progress
            new_alt = route.altitude(sc.km)
            ac.vs = (new_alt - ac.alt) / sim_dt if sim_dt > 0 else 0.0
            ac.alt, ac.speed = new_alt, route.airspeed(sc.km)
            ac.ground_speed = route.ground_speed(sc.km)
            ac.heading = route.bearing(frac)
            ctrl.throttle = 75 if ac.vs > 1.0 else 25 if ac.vs < -1.0 else 50
            if env is not None:
                env.step(min(ac.alt, 500.0), dt)
            # 화면용 기체: 고도를 줄이고, 지면이 흐르는 속도도 같은 비율로 줄인다
            visual.alt = min(ac.alt, ENROUTE_LOW_ALT + (ac.alt - ENROUTE_LOW_ALT) * ENROUTE_ALT_SCALE)
            step = ac.ground_speed * (visual.alt / ac.alt) * dt
            h = math.radians(ac.heading)
            visual.x, visual.y = visual.x + step * math.sin(h), visual.y + step * math.cos(h)
            visual.heading, visual.speed = ac.heading, ac.speed
            # 공항 근처는 항상 육지로 본다 (거친 해안선으로는 바닷가 공항이 바다 위로 판정되기도 한다)
            near_airport = min(sc.km, route.distance - sc.km) < 40.0
            target = 1.0 if near_airport or airway.over_land(*route.position(frac)) else 0.0
            land += (target - land) * min(dt / ENROUTE_BLEND_SECONDS, 1.0)
            info["airway"] = {"map": maps[0], "minimap": maps[1], "frac": frac, "land": land,
                              "visual": visual, "guide": guide_path(route, sc.km, ac, visual),
                              "banner": f"TO {route.dest}   {route.distance - sc.km:,.0f} km   "
                                        f"{airway.format_hm(max(total - sc.elapsed, 0.0))} LEFT"}
            fx.update(ac, ctrl, None, dt)
            cv2.imshow(WINDOW_NAME,
                       render(_view, sc, ac, [], ctrl, None, True, "run", sc.elapsed, info, fx))
            if user_wants_quit():
                quit_requested = True
                break
            _keys.clear()
    except KeyboardInterrupt:
        quit_requested = True
    info["airway"] = None
    info["hint"] = hint
    return quit_requested


def ask_airport(prompt, exclude=None) -> str:
    """터미널에서 실제 공항 하나를 고르게 한다.

    자주 쓰는 공항은 번호로, 그 밖의 전 세계 공항은 공항 코드(LHR 등)나 도시 이름 검색(paris 등)으로 고른다.
    """
    shown = [c for c in airway.KOREAN_NAMES if c != exclude and c in airway.AIRPORTS]
    describe = lambda c: f"{c} {airway.AIRPORTS[c][3]}"
    print(prompt)
    for i in range(0, len(shown), 3):   # 세 칸씩 나눠서 보여준다
        print("  " + "".join(f"{k + 1:2d}) {describe(shown[k]):<18}" for k in range(i, min(i + 3, len(shown)))))
    print(f"  이 밖의 공항(전 세계 {len(airway.AIRPORTS):,}곳)은 공항 코드(예: AMS)나 영문 도시 이름(예: paris)을 입력하세요.")
    while True:
        try:
            answer = input("번호, 공항 코드 또는 도시 이름: ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\n공항을 고르지 않아 시작하지 않습니다.")
        if answer.isdigit():
            if 1 <= int(answer) <= len(shown):
                return shown[int(answer) - 1]
            print(f"번호는 1~{len(shown)} 사이입니다.")
            continue
        code = answer.upper()
        if code in airway.AIRPORTS and code != exclude:
            return code
        if len(answer) < 2:
            continue
        found = [c for c in airway.search_airports(answer) if c != exclude]
        if not found:
            print("찾지 못했습니다. 영문 도시 이름이나 세 글자 공항 코드를 입력해 주세요.")
        elif len(found) == 1:
            return found[0]
        else:   # 여러 개면 목록을 바꿔서 다시 고르게 한다
            shown = found
            print("검색 결과:")
            for k, c in enumerate(shown, 1):
                print(f"  {k:2d}) {c} {airway.AIRPORTS[c][6]} ({airway.AIRPORTS[c][4]}, {airway.AIRPORTS[c][5]})")


def airport_code(text) -> str:
    """명령행의 공항 코드 확인."""
    code = text.upper()
    if code not in airway.AIRPORTS:
        raise argparse.ArgumentTypeError(f"{text!r}는 모르는 공항 코드입니다 (예: ICN, LAX, LHR)")
    return code


def ask_route(origin=None, dest=None):
    """출발 공항과 도착 공항을 차례로 정해 항로를 만든다. 너무 가까운 두 공항이면 다시 고르게 한다."""
    asked = origin is None or dest is None
    while True:
        a = origin or ask_airport("\n출발 공항을 고르세요")
        print(f"출발: {airway.AIRPORTS[a][3]} ({a})")
        b = dest or ask_airport("\n도착 공항을 고르세요", exclude=a)
        print(f"도착: {airway.AIRPORTS[b][3]} ({b})")
        route = airway.Route(a, b) if a != b else None
        if route is not None and route.distance >= airway.MIN_DISTANCE:
            return route
        message = (f"두 공항이 너무 가깝습니다 ({airway.MIN_DISTANCE:.0f} km 이상 떨어진 공항을 골라 주세요)."
                   if route is not None else "출발 공항과 도착 공항이 같습니다.")
        if not asked:
            raise SystemExit(message)
        print(message)
        origin = dest = None


def main_airway(args):
    """자동 조종 - 실제 항로 비행: 출발 공항 이륙 -> 순항(시간 압축) -> 도착 공항 착륙."""
    route = ask_route(args.origin, args.dest)
    origin, dest = airway.AIRPORTS[route.origin][0], airway.AIRPORTS[route.dest][0]
    print("자동 조종 모드입니다. 컴퓨터가 실제 항로를 따라 스스로 비행합니다. 카메라는 켜지 않고 기록도 남기지 않습니다.")
    print(f"항로: {route.label}")
    print(f"  거리 {route.distance:,.0f} km, 순항 고도 {route.cruise_alt:,.0f} m, "
          f"실제 비행 시간 약 {airway.format_hm(route.flight_seconds)} "
          f"(화면에서는 약 {route.show_seconds:.0f}초로 압축)")
    print("  V 키: 3D 화면 <-> 항로 지도\n")
    print(VIEW_HELP)
    print_environment(args)
    info = {"participant": "AUTOPILOT", "mode": "auto", "subtitle": f"{route.origin} > {route.dest}   NOT RECORDED",
            "session_id": "", "trial_label": "FLIGHT", "autopilot": True, "airway": None,
            "env": args.env, "wind_from": args.wind_from, "motion": not args.no_motion}
    maps = (airway.RouteMap(route, ck.MAP_W, ck.CANVAS_H), airway.RouteMap(route, 170, 170, labels=False))
    rng = random.Random(args.seed)
    inp = AutoInput(args)
    flights = 0
    try:
        open_window()
        quit_requested = False
        while not quit_requested and (args.repeats is None or flights < args.repeats):
            flights += 1
            info["trial_no"] = flights
            inp.new_trial()
            print(f"[항로 비행 {flights}] {origin} 이륙")
            info["hint"] = f"DEPARTURE: {origin} ({route.origin})"
            row, quit_requested = run_trial(inp, "takeoff", info, rng)
            if quit_requested or row is None or row["outcome"] != "success":
                continue
            print(f"[항로 비행 {flights}] 순항 시작")
            quit_requested = run_enroute(inp, route, maps, info, rng)
            if quit_requested:
                break
            inp.new_trial()
            print(f"[항로 비행 {flights}] {dest} 접근")
            info["hint"] = f"ARRIVAL: {dest} ({route.dest})"
            row, quit_requested = run_trial(inp, "landing", info, rng)
            if row is not None and row["outcome"] != "aborted":
                print(f"[항로 비행 {flights}] {dest} 착륙: {row['outcome']}  {row['fail_reason']}")
    finally:
        inp.close()
        close_window()
    print("자동 조종을 마쳤습니다. 기록은 저장하지 않았습니다.")


def main_auto(args):
    """자동 조종: 컴퓨터가 비행하는 모습을 보여준다. 카메라도 기록도 쓰지 않는다.

    기본은 실제 항로 비행(main_airway). --scenario 를 주면 그 과제를 자동 조종으로 수행한다.
    """
    if args.scenario is None:
        return main_airway(args)
    name = args.scenario
    scenarios = BUNDLES.get(name, [name])
    print("자동 조종 모드입니다. 컴퓨터가 스스로 비행합니다. 카메라는 켜지 않고 기록도 남기지 않습니다.")
    print(f"과제: {' → '.join(scenarios)}" + ("" if args.repeats else "  (창을 닫을 때까지 반복)") + "\n")
    print(VIEW_HELP)
    print_environment(args)
    info = {"participant": "AUTOPILOT", "mode": "auto", "subtitle": "NOT RECORDED",
            "session_id": "", "trial_label": "RUN", "autopilot": True,
            "env": args.env, "wind_from": args.wind_from, "motion": not args.no_motion}
    rng = random.Random(args.seed)
    inp = AutoInput(args)
    runs = 0
    try:
        open_window()
        quit_requested = False
        while not quit_requested and (args.repeats is None or runs < args.repeats * len(scenarios)):
            for name in scenarios:
                runs += 1
                info["trial_no"] = runs
                inp.new_trial()
                row, quit_requested = run_trial(inp, name, info, rng)
                if row is not None and row["outcome"] != "aborted":
                    print(f"[자동 조종 {runs}] {row['scenario']}: {row['outcome']}  "
                          f"time={row['target_time_s']}s  err={row['path_error_m']}m  {row['fail_reason']}")
                if quit_requested:
                    break
    finally:
        inp.close()
        close_window()
    print("자동 조종을 마쳤습니다. 기록은 저장하지 않았습니다.")


def main():
    parser = argparse.ArgumentParser(description="제스처 비행 조종 시뮬레이션")
    parser.add_argument("--flight", choices=list(FLIGHT_MODES), default=None,
                        help="비행 모드: staged = 단계별 시뮬레이션(실험 기록), free = 자유 비행, "
                             "auto = 자동 조종. 생략하면 실행 후 고릅니다.")
    parser.add_argument("--participant", default=None,
                        help="익명 참가자 코드 (예: P001). 이름 등 실명 정보는 넣지 마세요. "
                             "단계별 시뮬레이션에서만 쓰며, 생략하면 실행 후 입력받습니다.")
    parser.add_argument("--mode", choices=sorted(MODES), default="gesture", help="조종 입력 방식")
    parser.add_argument("--scenario", choices=list(SCENARIOS) + list(BUNDLES), default=None,
                        help="full = 이륙 -> 직선 비행 -> 원형 선회 -> 착륙 순서로 진행. "
                             "route = 좌우 선회가 이어지는 복잡한 경로를 따라 500 m까지 올라갔다 내려온다. "
                             "all = full 다음 단계로 route까지 (단계별 모드의 기본). "
                             "circuit = 이륙부터 착륙까지 한 번에 이어서 비행. "
                             "기본값: 단계별은 all. 자동 조종은 생략하면 실제 항로 비행(--from, --to), "
                             "주면 그 과제를 자동 조종으로 수행 (자유 비행은 과제 없음)")
    parser.add_argument("--from", dest="origin", type=airport_code, default=None, metavar="공항코드",
                        help="자동 조종 항로 비행의 출발 공항 (세 글자 IATA 코드, 예: ICN). 생략하면 실행 후 고릅니다")
    parser.add_argument("--to", dest="dest", type=airport_code, default=None, metavar="공항코드",
                        help="자동 조종 항로 비행의 도착 공항 (예: LAX). 생략하면 실행 후 고릅니다")
    parser.add_argument("--repeats", type=int, default=None,
                        help="시나리오(또는 full 묶음) 반복 횟수. 기본값: 단계별 1회, 자동 조종은 계속 반복")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "trials.csv")
    parser.add_argument("--seed", type=int, default=None, help="GO 신호 대기 시간·난기류 난수 시드")
    parser.add_argument("--env", choices=list(ENVIRONMENTS), default="normal",
                        help="환경: off = 바람·물리 효과 없는 단순 모델, calm = 물리 효과만, "
                             "normal = 약한 바람과 난기류, rough = 강한 바람과 난기류")
    parser.add_argument("--wind-from", type=float, default=DEFAULT_WIND_FROM,
                        help="바람이 불어오는 방위(도). 활주로는 북쪽(360도) 방향")
    parser.add_argument("--time", choices=list(TIMES), default=None,
                        help="시간대: day = 낮, night = 밤. 생략하면 실행 후 고릅니다 "
                             "(실행 중 N 키로도 전환)")
    parser.add_argument("--weather", choices=list(WEATHERS), default=None,
                        help="날씨(화면 효과와 시야): clear 맑음, cloudy 흐림, rain 비, storm 뇌우, "
                             "fog 안개, snow 눈. 생략하면 실행 후 고릅니다 (실행 중 W 키로도 전환). "
                             "바람·난기류는 --env 로 따로 정합니다")
    parser.add_argument("--cam", choices=list(CAMS), default=None,
                        help="시점: pilot = 조종사 시점, outside = 외부 시점. 생략하면 실행 후 고릅니다 "
                             "(실행 중 A 키나 화면 위 버튼으로도 전환)")
    parser.add_argument("--cockpit", action="store_true", help="--cam pilot 과 같다")
    parser.add_argument("--practice", action="store_true",
                        help="단계별 모드 맨 앞에 손동작 연습 화면을 넣는다 (기본은 바로 과제 시작)")
    parser.add_argument("--no-tracks", action="store_true",
                        help="시도별 비행 궤적 파일(tracks 폴더)을 저장하지 않는다")
    parser.add_argument("--no-guide", action="store_true",
                        help="조종 안내 십자선(자세계와 조종석 앞 유리의 마젠타 십자)을 끈다")
    parser.add_argument("--no-motion", action="store_true",
                        help="움직임 효과(가속·난기류·접지 때의 화면 흔들림)를 끈다")
    parser.add_argument("--view", choices=["3d", "2d"], default="3d",
                        help="시작 화면: 3d = 3D 추적 화면, 2d = 위에서 본 지도 (실행 중 V 키로 전환)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.3,
                        help="손 인식 신뢰도 기준. 낮출수록 기울인 손도 덜 놓치지만 오인식이 늘 수 있다")
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--deadzone", type=float, default=5.0)
    parser.add_argument("--max-roll", type=float, default=30.0,
                        help="roll이 최대가 되는 손 기울기(도). 작을수록 덜 기울여도 크게 돈다")
    parser.add_argument("--pitch-deadzone", type=float, default=10.0,
                        help="손바닥 앞뒤 기울기의 중립 범위(도)")
    parser.add_argument("--max-pitch", type=float, default=40.0,
                        help="pitch가 최대가 되는 손바닥 앞뒤 기울기(도)")
    parser.add_argument("--invert-pitch", action="store_true",
                        help="pitch 방향을 반대로 (손끝을 카메라 쪽으로 숙이면 상승)")
    args = parser.parse_args()

    # 참가자 코드를 준 경우는 실험이므로 묻지 않고 단계별 시뮬레이션으로 간다
    flight = args.flight or ("staged" if args.participant else ask_flight_mode())
    _view.mode = args.view
    _view.night = (args.time or ask_time()) == "night"
    print(f"시간대: {'밤' if _view.night else '낮'}\n")
    _view.weather = args.weather or ask_weather()
    print(f"날씨: {WEATHER_NAMES[_view.weather]}\n")
    cam = "pilot" if args.cockpit else (args.cam or ("outside" if args.view == "2d" else ask_cam()))
    _view.cam = "cockpit" if cam == "pilot" else "chase"
    print(f"시점: {CAMS[cam]}\n")
    if flight == "staged":
        main_staged(args, parser)
    elif flight == "free":
        main_free(args)
    else:
        main_auto(args)


if __name__ == "__main__":
    main()
