"""단순화한 비행 모델과 실험 시나리오 (이륙 / 직선 비행 / 원형 선회 / 착륙 / 장주 비행).

위에서 내려다본 2D 위치(x: 동쪽, y: 북쪽, 단위 m) + 고도(alt) 모델.
조종 입력은 roll(-1~1), 스로틀(0~100%), pitch(-1~1) 세 가지다.
고도는 속도와 pitch로 결정된다: V_LEVEL보다 빠르면 상승, 느리면 하강하고,
pitch를 당기면(+) 그만큼 더 상승, 밀면(-) 더 하강한다. pitch가 0이면 속도만으로 정해진다.
cv2에 의존하지 않으므로 카메라 없이 테스트할 수 있다.

환경 모델 (Environment)
    Aircraft.step()에 env를 주지 않으면 처음의 단순 모델 그대로 움직인다 (기존 기록과 같은 조건).
    env를 주면 아래 물리 효과와 바람·난기류가 더해진다.
    - 뱅크와 선회: 조종 입력은 뱅크 각을 정하고(반응 지연 있음), 선회율은 g·tan(뱅크)/속도.
      빠를수록 같은 뱅크에서 선회 반경이 커진다.
    - 선회 중 고도 손실: 기울이면 양력의 수직 성분이 줄어 수평 유지에 필요한 속도가 올라간다.
    - 에너지 교환: 상승하면 속도가 줄고 하강하면 속도가 붙는다.
      (pitch를 당겨 상승하면 속도가 줄어들므로 스로틀을 올려야 계속 올라갈 수 있다.)
    - 추력 한계: 가속·감속에 한계가 있어 이륙 활주에 실제처럼 시간이 걸린다.
    - 실속: 너무 느리면 양력을 잃고 급하게 떨어진다.
    - 지면 효과: 지면 가까이에서는 하강이 완만해진다.
    - 바람: 대기속도와 대지속도가 달라지고 옆바람에 밀린다. 지상에서는 기수가 바람 쪽으로 돈다.
    - 난기류: 시간에 따라 이어지는 무작위 돌풍(앞뒤·좌우·상하)이 속도·위치·상승률을 흔든다.
"""

import math
import random
from dataclasses import dataclass

# --- 비행 모델 상수 ---
V_MAX = 70.0             # 스로틀 100%일 때 목표 속도 (m/s)
V_LEVEL = 35.0           # 수평 비행 속도 = 이륙 속도 (스로틀 50%)
SPEED_TAU = 3.0          # 속도가 목표 속도를 따라가는 시간 상수 (s)
CLIMB_GAIN = 0.25        # (속도 - V_LEVEL) 1 m/s당 상승률 (m/s)
MAX_VS = 10.0            # 최대 상승/하강률 (m/s)
PITCH_CLIMB = 6.0        # pitch ±1일 때 더해지는 상승률 (m/s)
PITCH_ROTATE = 5.0       # 이륙 활주 중 pitch를 끝까지 당기면 이만큼 낮은 속도에서 뜬다 (m/s)
AIR_TURN_RATE = 15.0     # roll ±1일 때 공중 선회율 (deg/s)
GROUND_TURN_RATE = 5.0   # roll ±1일 때 지상 조향률 (deg/s)

# --- 환경 모델(env 사용 시)의 물리 상수 ---
G = 9.81                 # 중력 가속도 (m/s^2)
MAX_BANK_DEG = 40.0      # roll ±1일 때 뱅크 각 (도)
BANK_TAU = 0.5           # 뱅크가 입력을 따라가는 시간 상수 (s)
MAX_TURN_RATE = 25.0     # 선회율 상한 (deg/s, 아주 느릴 때 발산 방지)
MAX_ACCEL = 3.5          # 추력으로 낼 수 있는 최대 가속 (m/s^2)
MAX_DECEL = 3.0          # 항력·제동으로 낼 수 있는 최대 감속 (m/s^2)
VS_TAU = 0.6             # 상승률이 바뀌는 데 걸리는 시간 상수 (s)
V_STALL = 20.0           # 실속 속도 (m/s)
STALL_SINK_GAIN = 1.5    # 실속 속도보다 1 m/s 느릴 때마다 더해지는 하강률 (m/s)
MAX_SINK = 20.0          # 최대 하강률 (m/s, 실속 포함)
GROUND_EFFECT_ALT = 8.0  # 이 고도 아래에서 지면 효과 (m)
GROUND_EFFECT_MIN = 0.7  # 지면에서 하강률에 곱해지는 값
WEATHERVANE_GAIN = 0.15  # 지상 활주 중 옆바람 1 m/s당 기수가 바람 쪽으로 도는 비율 (deg/s)
GUST_TAU = 2.5           # 돌풍이 이어지는 시간 척도 (s)
ROTATE_MARGIN = 2.0      # 이륙은 V_LEVEL보다 이만큼 빨라야 한다 (돌풍에 다시 내려앉지 않도록, m/s)
UPDRAFT_FULL_ALT = 20.0  # 상하 돌풍이 온전히 작용하는 고도 (지면에서는 0: 공기가 땅을 뚫지 못한다)

# 환경 단계: (평균 풍속 m/s, 돌풍 표준편차 m/s). "off"는 환경 모델을 쓰지 않는 처음의 단순 모델.
ENVIRONMENTS = {"off": None, "calm": (0.0, 0.0), "normal": (4.0, 0.7), "rough": (8.0, 1.5)}
DEFAULT_WIND_FROM = 310.0   # 바람이 불어오는 방위 (도). 활주로 36 기준 왼쪽 앞에서 부는 바람


class Environment:
    """바람과 난기류. 좌표는 x 동쪽, y 북쪽, z 위쪽이고 값은 공기가 움직이는 속도(m/s)."""

    def __init__(self, level: str, wind_from: float = DEFAULT_WIND_FROM, seed=None):
        self.level = level
        self.wind_speed, self.gust_sigma = ENVIRONMENTS[level]
        self.wind_from = wind_from % 360.0
        a = math.radians(self.wind_from)
        # wind_from 에서 불어오므로 공기는 반대 방향으로 움직인다
        self._steady = (-self.wind_speed * math.sin(a), -self.wind_speed * math.cos(a))
        self._rng = random.Random(seed)
        self.gust = [0.0, 0.0, 0.0]   # 돌풍 성분 (동, 북, 위)
        self._scale = 1.0

    def step(self, alt: float, dt: float):
        """돌풍을 dt만큼 진행. 시간적으로 이어지는 잡음(Ornstein-Uhlenbeck 과정)."""
        self._scale = 0.6 + 0.4 * min(max(alt, 0.0) / 100.0, 1.0)   # 지면 가까이에서는 바람이 약하다
        if self.gust_sigma <= 0.0:
            return
        k = min(dt / GUST_TAU, 1.0)
        for i, sigma in enumerate((self.gust_sigma, self.gust_sigma, 0.6 * self.gust_sigma)):
            self.gust[i] += -self.gust[i] * k + sigma * math.sqrt(2.0 * k) * self._rng.gauss(0.0, 1.0)

    @property
    def wind(self):
        """현재 고도의 수평 바람 (동, 북): 평균 바람 + 돌풍."""
        return (self._steady[0] * self._scale + self.gust[0] * self._scale,
                self._steady[1] * self._scale + self.gust[1] * self._scale)

    def updraft(self, alt: float) -> float:
        return self.gust[2] * self._scale * min(max(alt, 0.0) / UPDRAFT_FULL_ALT, 1.0)

    def along(self, heading_deg: float, gust_only: bool = False) -> float:
        """기수 방향 성분 (+ = 뒷바람)."""
        wx, wy = (self.gust[0] * self._scale, self.gust[1] * self._scale) if gust_only else self.wind
        h = math.radians(heading_deg)
        return wx * math.sin(h) + wy * math.cos(h)

    def cross(self, heading_deg: float) -> float:
        """기수 기준 옆 성분 (+ = 왼쪽에서 불어 오른쪽으로 민다)."""
        wx, wy = self.wind
        h = math.radians(heading_deg)
        return wx * math.cos(h) - wy * math.sin(h)


# --- 활주로: x=0 중심선, y=0 ~ RUNWAY_LENGTH ---
RUNWAY_LENGTH = 1000.0
RUNWAY_HALF_WIDTH = 22.5

# --- 착륙 성공 기준 ---
LAND_MAX_SINK = 3.0      # 접지 시 최대 하강률 (m/s)
LAND_MAX_SPEED = 45.0    # 접지 시 최대 속도 (m/s)
LAND_MAX_OFFSET = 10.0   # 활주로 중심선에서 최대 이탈 (m)


@dataclass
class Aircraft:
    x: float
    y: float
    heading: float   # 도, 0 = 북쪽, 시계 방향(오른쪽)이 +
    speed: float
    alt: float
    airborne: bool
    vs: float = 0.0  # 수직 속도 (m/s, 상승 +)
    # 아래는 환경 모델(env)을 쓸 때만 의미가 있다
    bank: float = 0.0          # 뱅크 각 (도, 오른쪽 +)
    ground_speed: float = 0.0  # 대지속도 (m/s). speed는 대기속도
    stalled: bool = False
    physical: bool = False     # 환경 모델로 움직이는 중인지 (화면 표시용)
    _gust_along: float = 0.0

    def step(self, roll: float, throttle: float, dt: float, env: "Environment" = None,
             pitch: float = 0.0):
        """dt초 진행. 'liftoff' / 'touchdown' 이벤트가 생기면 반환.

        env가 None이면 처음의 단순 모델, 있으면 물리 효과·바람·난기류를 포함한 모델.
        pitch(-1~1)는 상승(+)·하강(-) 조종. 0이면 pitch 조종이 없던 때와 똑같이 움직인다.
        """
        pitch = max(-1.0, min(1.0, pitch))
        if env is not None:
            return self._step_physical(roll, throttle, dt, env, pitch)
        target_v = V_MAX * throttle / 100.0
        self.speed += (target_v - self.speed) * min(dt / SPEED_TAU, 1.0)

        rate = AIR_TURN_RATE if self.airborne else GROUND_TURN_RATE
        if self.airborne or self.speed > 1.0:
            self.heading += rate * roll * dt

        h = math.radians(self.heading)
        self.x += self.speed * math.sin(h) * dt
        self.y += self.speed * math.cos(h) * dt

        vs = max(-MAX_VS, min(MAX_VS, CLIMB_GAIN * (self.speed - V_LEVEL) + PITCH_CLIMB * pitch))
        if self.airborne:
            self.vs = vs
            self.alt += vs * dt
            if self.alt <= 0.0:
                self.alt = 0.0
                self.airborne = False
                return "touchdown"   # self.vs에 접지 순간 하강률이 남아 있음
        elif vs > 0.0:
            self.airborne = True
            self.vs = vs
            self.alt += vs * dt
            return "liftoff"
        return None

    def _step_physical(self, roll, throttle, dt, env, pitch=0.0):
        self.physical = True
        env.step(self.alt, dt)
        target_v = V_MAX * throttle / 100.0
        thrust = max(-MAX_DECEL, min(MAX_ACCEL, (target_v - self.speed) / SPEED_TAU))

        if not self.airborne:
            # --- 지상: 바퀴가 대지속도를 정하고, 대기속도 = 대지속도 + 맞바람 ---
            headwind = -env.along(self.heading)
            gs = max(self.ground_speed + thrust * dt, 0.0)
            self.ground_speed = gs
            self.speed = max(gs + headwind, 0.0)
            self.bank += (0.0 - self.bank) * min(dt / BANK_TAU, 1.0)
            if gs > 1.0:
                steer = GROUND_TURN_RATE * roll * min(gs / 10.0, 1.0)
                vane = -WEATHERVANE_GAIN * env.cross(self.heading) * min(gs / 10.0, 1.0)
                self.heading += (steer + vane) * dt
            h = math.radians(self.heading)
            self.x += gs * math.sin(h) * dt
            self.y += gs * math.cos(h) * dt
            self.vs = 0.0
            self.stalled = False
            # 양력이 무게를 충분히 넘으면 뜬다. pitch를 당기면(기수 들기) 조금 낮은 속도에서도 뜬다
            if self.speed > V_LEVEL + ROTATE_MARGIN - PITCH_ROTATE * max(pitch, 0.0):
                self.airborne = True
                self.vs = max(CLIMB_GAIN * (self.speed - V_LEVEL) + PITCH_CLIMB * pitch, 0.3)
                self.alt += self.vs * dt
                self._gust_along = env.along(self.heading, gust_only=True)
                return "liftoff"
            return None

        # --- 공중 ---
        self.bank += (roll * MAX_BANK_DEG - self.bank) * min(dt / BANK_TAU, 1.0)
        bank = math.radians(self.bank)
        # 정상 선회: 양력의 수평 성분이 구심력 -> 선회율 = g·tan(뱅크) / 속도
        rate = math.degrees(G * math.tan(bank) / max(self.speed, 10.0))
        self.heading += max(-MAX_TURN_RATE, min(MAX_TURN_RATE, rate)) * dt

        # 속도: 추력·항력 + 에너지 교환(상승하면 느려지고 하강하면 빨라진다) + 앞뒤 돌풍
        gust = env.along(self.heading, gust_only=True)
        self.speed += (thrust - G * self.vs / max(self.speed, 10.0)) * dt - (gust - self._gust_along)
        self._gust_along = gust
        self.speed = max(self.speed, 0.0)

        # 상승률: 기울일수록 수평 비행에 필요한 속도가 올라간다 (하중배수 1/cos(뱅크))
        v_level = V_LEVEL / math.sqrt(max(math.cos(bank), 0.3))
        vs_target = CLIMB_GAIN * (self.speed - v_level) + PITCH_CLIMB * pitch
        self.stalled = self.speed < V_STALL
        if self.stalled:
            vs_target -= STALL_SINK_GAIN * (V_STALL - self.speed)
        if vs_target < 0.0 and self.alt < GROUND_EFFECT_ALT:   # 지면 효과
            vs_target *= GROUND_EFFECT_MIN + (1.0 - GROUND_EFFECT_MIN) * self.alt / GROUND_EFFECT_ALT
        vs_target = max(-MAX_SINK, min(MAX_VS, vs_target))
        self.vs += (vs_target - self.vs) * min(dt / VS_TAU, 1.0)
        climb = self.vs + env.updraft(self.alt)

        # 위치: 대기속도 벡터 + 바람
        wx, wy = env.wind
        h = math.radians(self.heading)
        vx, vy = self.speed * math.sin(h) + wx, self.speed * math.cos(h) + wy
        self.ground_speed = math.hypot(vx, vy)
        self.x += vx * dt
        self.y += vy * dt
        self.alt += climb * dt
        if self.alt <= 0.0:
            self.alt = 0.0
            self.airborne = False
            self.vs = min(climb, 0.0)   # 접지 순간 하강률 (돌풍 포함)
            return "touchdown"
        return None


def on_runway(x: float, y: float) -> bool:
    return abs(x) <= RUNWAY_HALF_WIDTH and 0.0 <= y <= RUNWAY_LENGTH


class Scenario:
    name = ""
    instruction = ""
    error_label = "PATH ERR"   # 화면에 표시하는 경로 오차 이름
    timeout = 90.0     # 초
    view_scale = 0.3   # 화면 픽셀 / m

    def __init__(self):
        self.done = False
        self.outcome = None        # success / fail / timeout / aborted
        self.reason = ""
        self.reached_time = None   # 목표 도달 시각 (GO 기준 초)
        self.touchdown = None      # 착륙 시나리오: 접지 정보

    def initial_aircraft(self) -> Aircraft:
        raise NotImplementedError

    def path_error(self, ac: Aircraft) -> float:
        """목표 경로와의 수평 편차 (m)."""
        raise NotImplementedError

    def targets(self):
        """화면에 그릴 목표 도형 (월드 좌표): ('line', p1, p2) / ('circle', c, r) / ('point', p)."""
        return []

    def update(self, ac: Aircraft, event, t: float):
        raise NotImplementedError

    def finish(self, outcome: str, reason: str = "", t: float = None):
        if self.done:
            return
        self.done, self.outcome, self.reason = True, outcome, reason
        if outcome == "success" and t is not None:
            self.reached_time = t

    @property
    def landing_success(self):
        return None   # 착륙 시나리오가 아니면 기록하지 않음


class Takeoff(Scenario):
    name = "takeoff"
    instruction = "TAKEOFF: keep the centerline, climb to 100 m"
    timeout = 60.0
    view_scale = 0.35
    TARGET_ALT = 100.0

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=50.0, heading=0.0, speed=0.0, alt=0.0, airborne=False)

    def path_error(self, ac):
        return abs(ac.x)

    def targets(self):
        return [("line", (0.0, 0.0), (0.0, 3000.0))]

    def update(self, ac, event, t):
        if event == "touchdown":
            self.finish("fail", "ground contact after liftoff")
        elif not ac.airborne and abs(ac.x) > RUNWAY_HALF_WIDTH:
            self.finish("fail", "runway excursion")
        elif not ac.airborne and ac.y > RUNWAY_LENGTH:
            self.finish("fail", "runway overrun")
        elif ac.alt >= self.TARGET_ALT:
            self.finish("success", t=t)


class Straight(Scenario):
    name = "straight"
    instruction = "STRAIGHT: follow the line to the goal (2000 m)"
    timeout = 90.0
    view_scale = 0.25
    GOAL_Y = 2000.0

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=0.0, heading=0.0, speed=V_LEVEL, alt=150.0, airborne=True)

    def path_error(self, ac):
        return abs(ac.x)

    def targets(self):
        return [("line", (0.0, 0.0), (0.0, self.GOAL_Y)), ("point", (0.0, self.GOAL_Y))]

    def update(self, ac, event, t):
        if event == "touchdown":
            self.finish("fail", "ground impact")
        elif ac.y >= self.GOAL_Y:
            self.finish("success", t=t)


class Turn(Scenario):
    name = "turn"
    instruction = "TURN: fly one full right circle (R = 200 m)"
    timeout = 120.0
    view_scale = 0.7
    RADIUS = 200.0

    def __init__(self):
        super().__init__()
        self.center = (self.RADIUS, 0.0)   # 시작점 (0,0)에서 북쪽을 보면 오른쪽에 원의 중심
        self.prev_angle = None
        self.swept = 0.0                   # 시계 방향으로 돈 누적 각도 (도)

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=0.0, heading=0.0, speed=V_LEVEL, alt=150.0, airborne=True)

    def path_error(self, ac):
        return abs(math.hypot(ac.x - self.center[0], ac.y - self.center[1]) - self.RADIUS)

    def targets(self):
        return [("circle", self.center, self.RADIUS), ("point", (0.0, 0.0))]

    def update(self, ac, event, t):
        a = math.atan2(ac.y - self.center[1], ac.x - self.center[0])
        if self.prev_angle is not None:
            d = (a - self.prev_angle + math.pi) % (2 * math.pi) - math.pi
            self.swept -= math.degrees(d)   # 시계 방향이 +
        self.prev_angle = a

        if event == "touchdown":
            self.finish("fail", "ground impact")
        elif self.swept >= 360.0:
            self.finish("success", t=t)


def landing_faults(ac: Aircraft, sink: float) -> list:
    """접지 순간이 착륙 성공 기준을 벗어난 항목들. 빈 목록이면 성공."""
    reasons = []
    if not on_runway(ac.x, ac.y):
        reasons.append("off runway")
    if sink > LAND_MAX_SINK:
        reasons.append(f"hard touchdown {sink:.1f} m/s")
    if ac.speed > LAND_MAX_SPEED:
        reasons.append(f"too fast {ac.speed:.0f} m/s")
    if abs(ac.x) > LAND_MAX_OFFSET:
        reasons.append(f"off centerline {ac.x:+.1f} m")
    return reasons


class Landing(Scenario):
    name = "landing"
    instruction = "LANDING: touch down softly on the centerline"
    timeout = 120.0
    view_scale = 0.35

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=-1500.0, heading=0.0, speed=V_LEVEL, alt=80.0, airborne=True)

    def path_error(self, ac):
        return abs(ac.x)

    def targets(self):
        return [("line", (0.0, -1500.0), (0.0, RUNWAY_LENGTH))]

    def update(self, ac, event, t):
        if event != "touchdown":
            return
        sink = -ac.vs
        self.touchdown = {"sink": sink, "speed": ac.speed, "offset": ac.x, "y": ac.y}
        self.reached_time = t   # 착륙은 성공 여부와 무관하게 접지 시각을 도달 시간으로 기록

        reasons = landing_faults(ac, sink)
        if reasons:
            self.finish("fail", ", ".join(reasons))
        else:
            self.finish("success", t=t)

    @property
    def landing_success(self):
        return self.outcome == "success"


def _resample(pts, step):
    """꺾은선을 따라 step(m) 간격으로 다시 찍은 점들."""
    out = [pts[0]]
    carry = 0.0   # 직전 점에서 이미 지나온 거리
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = math.hypot(x2 - x1, y2 - y1)
        d = step - carry
        while d <= seg:
            out.append((x1 + (x2 - x1) * d / seg, y1 + (y2 - y1) * d / seg))
            d += step
        carry = (carry + seg) % step
    return out


class Circuit(Scenario):
    """장주 비행: 이륙 -> 상승 -> 오른쪽 선회 -> 배풍 구간 -> 오른쪽 선회 -> 최종 접근 -> 착륙.

    한 번의 시도로 이륙부터 착륙까지 이어서 비행한다. 경로는 활주로(x=0)를 한 변으로 하는
    경주로 모양이다: 북쪽으로 이륙해 반원으로 돌아 x=2R에서 남쪽으로 내려온 뒤,
    다시 반원으로 돌아 활주로 연장선에 정렬해 착륙한다.
    성공 기준은 착륙 시나리오와 같고, 경로를 다 돌기 전에 접지하면 실패다.
    """
    name = "circuit"
    instruction = "CIRCUIT: take off, fly the right-hand pattern, land on the runway"
    timeout = 300.0
    view_scale = 0.3
    RADIUS = 250.0      # 선회 반경 (roll 약 0.5, 수평 비행 속도 기준)
    TOP_Y = 1100.0      # 첫 선회를 시작하는 지점
    BOTTOM_Y = -800.0   # 두 번째 선회가 끝나 최종 접근이 시작되는 지점
    STEP = 10.0         # 경로 오차 계산용 점 간격 (m)

    def __init__(self):
        super().__init__()
        self.outline = self._racetrack(6.0)                     # 화면에 그릴 꺾은선
        self.path = _resample(self._racetrack(2.0), self.STEP)  # 오차·진행도 계산용
        # 최종 접근 직선이 시작되는 점의 번호: 여기까지 와야 장주를 다 돈 것
        r = self.RADIUS
        final_start = self.TOP_Y + 2 * math.pi * r + (self.TOP_Y - self.BOTTOM_Y)
        self.final_idx = int(final_start / self.STEP)
        self.idx = 5          # 지금까지 도달한 경로 위 점 번호 (시작 위치 y=50)
        self.lifted = False

    def _racetrack(self, arc_step_deg):
        r, top, bottom = self.RADIUS, self.TOP_Y, self.BOTTOM_Y
        n = int(round(180 / arc_step_deg))
        pts = [(0.0, 0.0), (0.0, top)]
        # 위쪽 반원: 서쪽 끝(180도)에서 북쪽을 지나 동쪽 끝(0도)까지 시계 방향
        pts += [(r + r * math.cos(math.radians(180 - 180 * i / n)),
                 top + r * math.sin(math.radians(180 - 180 * i / n))) for i in range(1, n + 1)]
        pts.append((2 * r, bottom))
        # 아래쪽 반원: 동쪽 끝에서 남쪽을 지나 서쪽 끝까지 시계 방향
        pts += [(r + r * math.cos(math.radians(-180 * i / n)),
                 bottom + r * math.sin(math.radians(-180 * i / n))) for i in range(1, n + 1)]
        pts.append((0.0, RUNWAY_LENGTH))
        return pts

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=50.0, heading=0.0, speed=0.0, alt=0.0, airborne=False)

    def _nearest(self, ac):
        """진행 위치 근처(뒤 100 m ~ 앞 400 m)에서 가장 가까운 경로 점 (번호, 거리).

        범위를 제한해야 상승 구간과 최종 접근 구간(둘 다 x=0)을 혼동하지 않는다.
        """
        lo = max(self.idx - 10, 0)
        hi = min(self.idx + 40, len(self.path) - 1)
        best = min(range(lo, hi + 1),
                   key=lambda i: (self.path[i][0] - ac.x) ** 2 + (self.path[i][1] - ac.y) ** 2)
        return best, math.hypot(self.path[best][0] - ac.x, self.path[best][1] - ac.y)

    def path_error(self, ac):
        return self._nearest(ac)[1]

    def targets(self):
        return [("path", self.outline)]

    @property
    def progress(self) -> float:
        """장주 진행률 0~1 (최종 접근 시작점 = 1)."""
        return min(self.idx / self.final_idx, 1.0)

    def update(self, ac, event, t):
        self.idx = max(self.idx, self._nearest(ac)[0])
        if event == "liftoff":
            self.lifted = True
        if event == "touchdown":
            sink = -ac.vs
            self.touchdown = {"sink": sink, "speed": ac.speed, "offset": ac.x, "y": ac.y}
            self.reached_time = t   # 착륙 시나리오와 같이 접지 시각을 도달 시간으로 기록
            if self.idx < self.final_idx:
                self.finish("fail", "landed before completing the circuit")
                return
            reasons = landing_faults(ac, sink)
            if reasons:
                self.finish("fail", ", ".join(reasons))
            else:
                self.finish("success", t=t)
        elif not self.lifted and not ac.airborne:   # 이륙 활주 중
            if abs(ac.x) > RUNWAY_HALF_WIDTH:
                self.finish("fail", "runway excursion")
            elif ac.y > RUNWAY_LENGTH:
                self.finish("fail", "runway overrun")

    @property
    def landing_success(self):
        return self.outcome == "success"


class FreeFlight(Scenario):
    """자유 비행: 과제·제한 시간·성공 판정 없이 마음대로 난다. 실험 기록에는 쓰지 않는다.

    활주로에서 출발하고, 접지할 때마다 착륙 기준에 비춰 본 결과를 알려 줄 뿐 끝나지 않는다.
    """
    name = "free flight"
    instruction = "FREE FLIGHT: fly anywhere, land anywhere"
    timeout = None             # 제한 시간 없음
    view_scale = 0.3
    error_label = "RWY OFFSET"

    def __init__(self):
        super().__init__()
        self.result_title = ""   # 접지 결과 제목 (화면 표시용)
        self.landings = 0

    def initial_aircraft(self):
        return Aircraft(x=0.0, y=50.0, heading=0.0, speed=0.0, alt=0.0, airborne=False)

    def path_error(self, ac):
        return abs(ac.x)   # 활주로 중심선(연장선)에서 벗어난 거리

    def update(self, ac, event, t):
        if event != "touchdown":
            return
        sink = -ac.vs
        self.landings += 1
        self.touchdown = {"sink": sink, "speed": ac.speed, "offset": ac.x, "y": ac.y}
        reasons = landing_faults(ac, sink)
        self.outcome = "fail" if reasons else "success"
        self.reason = ", ".join(reasons)
        self.result_title = "ROUGH LANDING" if reasons else "GOOD LANDING"


SCENARIOS = {cls.name: cls for cls in (Takeoff, Straight, Turn, Landing, Circuit)}
# "full" 묶음에 들어가는 기본 과제 순서. circuit은 --scenario circuit 으로 따로 고른다.
SCENARIO_ORDER = ["takeoff", "straight", "turn", "landing"]
