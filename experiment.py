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
    python3 experiment.py --participant P004 --env rough           # 강한 바람과 난기류
    python3 experiment.py --participant P005 --env off             # 바람·물리 효과 없는 처음의 단순 모델

환경 (--env): 물리 효과(뱅크 선회, 선회 중 고도 손실, 에너지 교환, 추력 한계, 실속, 지면 효과)와
바람·난기류. off = 단순 모델, calm = 물리 효과만, normal = 약한 바람(기본), rough = 강한 바람.
어떤 환경에서 수행했는지는 trials.csv의 environment 열에 기록된다.

손 조종
    좌우 기울임 = 선회,  펼친 손가락 개수 = 스로틀(0~5개 -> 0~100%)
    손바닥 앞뒤 기울임 = 상승·하강: 손끝을 내 쪽으로 젖히면 상승, 카메라 쪽으로 숙이면 하강
    (방향이 반대로 느껴지면 --invert-pitch)

화면 안내 문구는 영어다 (OpenCV 기본 글꼴은 한글을 그리지 못함).
화면 시점 (기본은 3D 추적 화면)
    마우스 왼쪽 드래그 / 트랙패드 두 손가락 스크롤 / 방향키 / 4 6 8 2   카메라 회전
    + - / Ctrl(또는 Option) + 스크롤                                   확대 / 축소
    1 뒤 / 3 옆 / 7 위 / 9 앞             미리 정한 시점,  5 또는 0 = 기본 시점
    V                                     3D <-> 2D 지도 전환
    N 또는 화면 위 DAY/NIGHT 버튼 클릭     낮 <-> 밤 전환 (--time night 으로 밤에서 시작)
    W 또는 그 옆 날씨 버튼 클릭           날씨 전환 (--weather rain 처럼 시작 날씨 지정)

종료: 창의 닫기 버튼(macOS는 창 왼쪽 위 빨간 버튼), q 또는 ESC, 터미널에서 Ctrl+C.
중단하면 진행 중이던 시도는 outcome=aborted로 기록된다.
"""

import argparse
import csv
import random
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from flight_sim import (DEFAULT_WIND_FROM, ENVIRONMENTS, SCENARIO_ORDER, SCENARIOS,
                        Environment, FreeFlight)
from autopilot import Autopilot
from cockpit import FlightEffects
from hand_tracking import (FlightController, draw_hand, ensure_model, forbid_frame_saving)
from view3d import WEATHERS, View, normalize_key, render

CONSENT_TEXT = """
==================================================================
                      실험 참여 동의 안내
==================================================================
본 실험은 손 제스처로 가상 비행기를 조종하는 방식의 사용성을
평가합니다. 이륙, 직선 비행, 원형 선회, 착륙 과제를 수행합니다.

[수집하는 정보]
  - 익명 참가자 코드 ({participant})
  - 과제 수행 기록: 시작 시각, 반응 시간, 목표 도달 시간,
    경로 오차, 착륙 성공 여부, 손 인식 실패 비율

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
]
# 나중에 추가한 열과, 그 열이 없던 예전 기록에 채울 값 (예전 기록은 단순 모델·낮 화면이었다)
ADDED_FIELDS = {"environment": "off", "wind_speed_ms": 0.0, "wind_from_deg": "", "lighting": "day",
                "weather": "clear"}

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

    def read(self):
        """(미리보기 프레임, 손 인식 여부) 반환. 카메라 오류 시 None."""
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
            self.ctrl.update(lms, side, w, h, world)
            draw_hand(frame, lms, side)
            return frame, True
        self.ctrl.reset()
        return frame, False

    def close(self):
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
    fx = FlightEffects(ac, ctrl, env)   # 이륙·착륙 연출 (표시 전용)

    phase = "ready"
    ready_until = time.monotonic() + rng.uniform(*READY_DELAY)
    go_time = start_iso = base_roll = base_pitch = base_throttle = reaction = result_until = None
    err_sum = run_time = 0.0
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
                err_sum += sc.path_error(ac) * dt
                run_time += dt
                frames += 1
                lost += not hand
                if np.hypot(ac.x - trail[-1][0], ac.y - trail[-1][1]) > 5:
                    trail.append((ac.x, ac.y, ac.alt))
                if not sc.done and t >= sc.timeout:
                    sc.finish("timeout", "time limit")
                if sc.done:
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
    }
    return row, quit_requested


# ---------------------------------------------------------------- 자유 비행

FREE_NOTICE_SECONDS = 5.0   # 접지 결과를 보여주는 시간
FREE_TRAIL_POINTS = 2000    # 궤적은 최근 것만 남긴다 (약 10 km)


def run_free(inp, info, rng):
    """자유 비행: 과제 없이 창을 닫을 때까지 난다. 아무것도 기록하지 않는다."""
    def start():
        sc = FreeFlight()
        ac = sc.initial_aircraft()
        env = None
        if ENVIRONMENTS[info["env"]] is not None:
            env = Environment(info["env"], info["wind_from"], seed=rng.random())
        return sc, ac, env, FlightEffects(ac, inp.ctrl, env), [(ac.x, ac.y, ac.alt)]

    ctrl = inp.ctrl
    sc, ac, env, fx, trail = start()
    t = 0.0
    notice_until = 0.0
    landings = 0   # 다시 시작(R)해도 이어서 센다
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

            phase = "result" if now < notice_until else "run"   # 접지 직후에만 결과를 띄운다
            cv2.imshow(WINDOW_NAME,
                       render(_view, sc, ac, trail, ctrl, cam, hand, phase, t, info, fx))
            if user_wants_quit():
                break
            while _keys:
                if _keys.pop() in (ord("r"), ord("R")):   # 활주로 출발 위치에서 다시 시작
                    sc, ac, env, fx, trail = start()
                    t, notice_until = 0.0, 0.0
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
    "auto": "자동 조종         - 컴퓨터가 스스로 비행, 지켜보기 (카메라·기록 없음)",
}
VIEW_HELP = ("화면 시점: 마우스 드래그·트랙패드 두 손가락 스크롤·방향키(또는 4 6 8 2) = 회전,\n"
             "           +/- 또는 Ctrl+스크롤 = 확대/축소,\n"
             "           1 3 7 9 = 미리 정한 시점, 5 = 기본 시점, V = 3D/2D 전환,\n"
             "낮/밤 전환: N 키 또는 화면 위 가운데의 DAY/NIGHT 버튼 클릭\n"
             "날씨 전환: W 키 또는 그 옆의 날씨 버튼 클릭\n"
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


def main_staged(args, parser):
    """단계별 시뮬레이션: 참가자 코드와 동의를 받고 과제를 차례로 수행해 기록한다."""
    if args.participant is None:
        args.participant = ask_participant()
    elif not PARTICIPANT_RE.match(args.participant):
        parser.error(PARTICIPANT_HELP)

    name = args.scenario or "full"
    scenarios = (SCENARIO_ORDER if name == "full" else [name]) * (args.repeats or 1)
    rng = random.Random(args.seed)
    info = {
        "participant": args.participant.upper(),
        "mode": args.mode,
        "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "env": args.env,
        "wind_from": args.wind_from,
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
    try:
        open_window()
        for name in scenarios:
            info["trial_no"] = trial_no
            row, quit_requested = run_trial(inp, name, info, rng)
            if row is not None:
                append_row(args.output, row)
                saved += 1
                trial_no += 1
                print(f"[{row['scenario']}] {row['outcome']}  RT={row['reaction_time_s']}s  "
                      f"time={row['target_time_s']}s  err={row['path_error_m']}m")
            if quit_requested:
                break
    finally:
        inp.close()
        close_window()
    print(f"{saved}개 시도를 {args.output}에 저장했습니다.")


def main_free(args):
    """자유 비행: 손으로 조종하지만 과제도 기록도 없다."""
    print("자유 비행 모드입니다. 과제나 판정 없이 자유롭게 비행합니다.")
    print("웹캠 영상은 손 인식을 위해 메모리에서만 처리되며 저장되지 않습니다. 비행 기록도 남기지 않습니다.")
    print("R 키 = 활주로 출발 위치에서 다시 시작\n")
    print(VIEW_HELP)
    print_environment(args)
    info = {"participant": "FREE FLIGHT", "mode": args.mode, "subtitle": "NOT RECORDED",
            "trial_no": None, "env": args.env, "wind_from": args.wind_from, "hint": "R: RESTART"}
    forbid_frame_saving()
    inp = MODES[args.mode](args)
    try:
        open_window()
        landings = run_free(inp, info, random.Random(args.seed))
    finally:
        inp.close()
        close_window()
    print(f"자유 비행을 마쳤습니다 (착륙 {landings}회). 기록은 저장하지 않았습니다.")


def main_auto(args):
    """자동 조종: 컴퓨터가 과제를 수행하는 모습을 보여준다. 카메라도 기록도 쓰지 않는다."""
    name = args.scenario or "circuit"
    scenarios = SCENARIO_ORDER if name == "full" else [name]
    print("자동 조종 모드입니다. 컴퓨터가 스스로 비행합니다. 카메라는 켜지 않고 기록도 남기지 않습니다.")
    print(f"과제: {' → '.join(scenarios)}" + ("" if args.repeats else "  (창을 닫을 때까지 반복)") + "\n")
    print(VIEW_HELP)
    print_environment(args)
    info = {"participant": "AUTOPILOT", "mode": "auto", "subtitle": "NOT RECORDED",
            "session_id": "", "trial_label": "RUN", "autopilot": True,
            "env": args.env, "wind_from": args.wind_from}
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
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["full"], default=None,
                        help="full = 이륙 -> 직선 비행 -> 원형 선회 -> 착륙 순서로 진행. "
                             "circuit = 이륙부터 착륙까지 한 번에 이어서 비행. "
                             "기본값: 단계별은 full, 자동 조종은 circuit (자유 비행은 과제 없음)")
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
    parser.add_argument("--view", choices=["3d", "2d"], default="3d",
                        help="시작 화면: 3d = 3D 추적 화면, 2d = 위에서 본 지도 (실행 중 V 키로 전환)")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--deadzone", type=float, default=5.0)
    parser.add_argument("--max-roll", type=float, default=45.0)
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
    if flight == "staged":
        main_staged(args, parser)
    elif flight == "free":
        main_free(args)
    else:
        main_auto(args)


if __name__ == "__main__":
    main()
