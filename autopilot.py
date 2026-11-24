"""자동 조종: 컴퓨터가 사람과 같은 두 가지 입력(roll -1~1, 스로틀 0/25/50/75/100%)만으로 비행한다.

방향: 경로 위의 조금 앞 지점을 향해 간다. 바람에 밀리는 것을 감안하려고 기수 방위가 아니라
      실제로 움직인 방향(항적)을 기준으로 삼는다. 그러면 옆바람에서 기수를 바람 쪽으로 튼 채
      (크랩) 경로를 따라가게 된다.
고도: 이 비행 모델에는 피치 조종이 없어 속도(스로틀)로 고도를 정한다. 목표 고도보다 낮으면
      스로틀을 올리고 높으면 내린다.
착륙: 활주로 시단 조금 안쪽을 향해 일정한 각도로 내려오다가, 지면 가까이에서는 속도를 보며
      스로틀을 조절해 접지 하강률을 줄인다.

cv2에 의존하지 않으므로 카메라와 화면 없이 테스트할 수 있다.
"""

import math

LOOKAHEAD = 80.0        # 앞을 내다보는 거리 (m)
FULL_ROLL_ERROR = 12.0  # 방향 오차가 이만큼이면 roll을 끝까지 준다 (도)
CRUISE_ALT = 150.0      # 직선·선회 과제의 유지 고도 (m)
PATTERN_ALT = 120.0     # 장주 비행의 유지 고도 (m)
AIM_Y = 150.0           # 착륙 때 겨냥하는 지점 (활주로 시단에서 안쪽으로, m)
GLIDE_SLOPE = 0.08      # 접근 강하 기울기 (고도 / 남은 거리)
PITCH_FULL_ERROR = 30.0  # 고도 오차가 이만큼이면 피치를 끝까지 준다 (m)
FLARE_ALT = 8.0         # 이 고도 아래에서는 접지 준비
FLARE_SPEED = 31.0      # 접지 준비 중 이 속도보다 빠르면 스로틀을 줄인다 (m/s)


def wrap(deg: float) -> float:
    """각도 차이를 -180 ~ 180도로."""
    return (deg + 180.0) % 360.0 - 180.0


class Autopilot:
    def __init__(self):
        self._prev = None   # 직전 위치 (항적 계산용)

    def control(self, sc, ac):
        """현재 과제(sc)와 기체 상태(ac)를 보고 (roll, 스로틀)을 정한다."""
        return self._roll(sc, ac), self._throttle(sc, ac)

    @staticmethod
    def pitch(sc, ac):
        """목표 고도가 변하는 과제(route)에서만 피치로 고도를 맞춘다. 그 밖에는 0 (스로틀로만 조절)."""
        if not hasattr(sc, "target_alt"):
            return 0.0
        return max(-1.0, min(1.0, (sc.target_alt - ac.alt) / PITCH_FULL_ERROR))

    # ------------------------------------------------------------ 조종 안내 (사람이 조종할 때)

    @staticmethod
    def _guide_alt(sc, ac):
        """사람에게 안내할 목표 고도 (m). 안내할 고도가 없으면 None."""
        if hasattr(sc, "target_alt"):
            return sc.target_alt
        if hasattr(sc, "guide_alt"):   # 항로 비행의 출발·도착 구간
            return sc.guide_alt(ac)
        if sc.name == "takeoff":
            return sc.TARGET_ALT + 20.0 if ac.airborne else None
        final = sc.name == "landing" or (sc.name == "circuit" and sc.idx >= sc.final_idx - 100)
        if final:   # 접근 강하 경로
            return max(GLIDE_SLOPE * (AIM_Y - ac.y), 0.0)
        if sc.name == "circuit":
            return PATTERN_ALT if sc.lifted else None
        if sc.name in ("straight", "turn"):
            return CRUISE_ALT
        return None

    def guidance(self, sc, ac):
        """경로와 목표 고도를 지키려면 넣어야 할 (roll, pitch) 입력. 화면의 조종 안내 십자선용."""
        alt = self._guide_alt(sc, ac)
        pitch = 0.0 if alt is None else max(-1.0, min(1.0, (alt - ac.alt) / PITCH_FULL_ERROR))
        return self._roll(sc, ac), pitch

    # ------------------------------------------------------------ 방향

    def _target(self, sc, ac):
        """경로 위에서 LOOKAHEAD만큼 앞에 있는 지점."""
        if hasattr(sc, "path"):   # 장주 비행: 경로 점 목록
            i, _ = sc._nearest(ac)
            return sc.path[min(i + int(LOOKAHEAD / sc.STEP), len(sc.path) - 1)]
        if hasattr(sc, "center"):   # 원형 선회: 원을 따라 시계 방향으로 앞쪽
            cx, cy = sc.center
            a = math.atan2(ac.y - cy, ac.x - cx) - LOOKAHEAD / sc.RADIUS
            return cx + sc.RADIUS * math.cos(a), cy + sc.RADIUS * math.sin(a)
        return 0.0, ac.y + LOOKAHEAD   # 그 밖에는 활주로 중심선(x = 0)을 따라 북쪽으로

    def _track(self, ac):
        """실제로 움직이고 있는 방향 (도). 지상이거나 거의 안 움직였으면 기수 방위."""
        prev, self._prev = self._prev, (ac.x, ac.y)
        if prev is None or not ac.airborne:
            return ac.heading
        dx, dy = ac.x - prev[0], ac.y - prev[1]
        if math.hypot(dx, dy) < 0.05:
            return ac.heading
        return math.degrees(math.atan2(dx, dy))

    def _roll(self, sc, ac):
        tx, ty = self._target(sc, ac)
        course = math.degrees(math.atan2(tx - ac.x, ty - ac.y))
        error = wrap(course - self._track(ac))
        return max(-1.0, min(1.0, error / FULL_ROLL_ERROR))

    # ------------------------------------------------------------ 고도 (스로틀)

    @staticmethod
    def _hold(ac, alt):
        """목표 고도 유지: 낮으면 상승(75%), 높으면 하강(25%), 근처면 수평(50%)."""
        if ac.alt < alt - 10.0:
            return 75
        if ac.alt > alt + 15.0:
            return 25
        return 50

    @staticmethod
    def _approach(ac):
        if ac.alt <= FLARE_ALT:   # 접지 준비: 너무 빠르면 줄이고 아니면 수평 추력으로 하강을 줄인다
            return 25 if ac.speed > FLARE_SPEED else 50
        too_high = ac.alt / max(AIM_Y - ac.y, 1.0) > GLIDE_SLOPE
        return 25 if (too_high or ac.y > AIM_Y - 50.0) else 50

    def _throttle(self, sc, ac):
        if hasattr(sc, "throttle_plan"):   # 항로 비행의 출발·도착 구간은 과제가 추력 계획을 정한다
            return sc.throttle_plan(self, ac)
        if sc.name == "takeoff":
            return 100
        if sc.name == "landing":
            return self._approach(ac)
        if sc.name == "circuit":
            if not sc.lifted:
                return 100
            if sc.idx < sc.final_idx - 100:   # 최종 접근 1 km 전까지는 장주 고도 유지
                return self._hold(ac, PATTERN_ALT)
            return self._approach(ac)
        if hasattr(sc, "target_alt"):   # 피치로 고도를 맞추되, 많이 낮으면 추력을 더한다
            return 75 if ac.alt < sc.target_alt - 25.0 else 50
        return self._hold(ac, CRUISE_ALT)
