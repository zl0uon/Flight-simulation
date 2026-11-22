"""실험 화면 그리기: 이동 지도(왼쪽) + 글래스 콕핏 스타일 계기판(오른쪽).

표시만 담당한다. 비행 모델(flight_sim)과 측정값에는 영향을 주지 않는다.
- 지도: 기체를 가운데 둔 북쪽 고정(north-up) 추적 화면. 활주로 표지, 유도로, 계류장,
  마젠타색 목표 경로, 기체 실루엣과 고도에 따라 멀어지는 그림자.
- 계기판: 자세계(인공 수평선) + 속도/고도 테이프, 방위 지시계, 스로틀, 승강계,
  조종 입력, 손 인식 상태, 카메라 미리보기.

자세계의 뱅크·피치 각도는 보여주기용 근사값이다 (모델에는 뱅크/피치 상태가 없음).
화면 문구는 영어다 (OpenCV 기본 글꼴은 한글을 그리지 못함).
"""

import math
import time

import cv2
import numpy as np

import scenery

from flight_sim import LAND_MAX_SINK, MAX_VS, RUNWAY_HALF_WIDTH, RUNWAY_LENGTH, V_LEVEL

CANVAS_W, CANVAS_H = 1000, 700
MAP_W = 700                      # 지도 영역 너비, 나머지는 계기판
MAP_C = (MAP_W // 2, CANVAS_H // 2)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# 색 (BGR)
WHITE = (245, 245, 245)
GRAY = (160, 160, 160)
DIM = (95, 95, 95)
PANEL = (30, 27, 25)
INSTR = (46, 42, 40)
GREEN = (90, 230, 90)
AMBER = (0, 190, 255)
RED = (60, 60, 235)
CYAN = (255, 230, 80)
MAGENTA = (255, 60, 255)
YELLOW = (0, 230, 255)
SKY = (214, 140, 52)
EARTH = (36, 78, 128)
ASPHALT = (54, 52, 52)
TAXIWAY = (84, 82, 80)

MAX_BANK_DEG = 30.0      # roll 입력 ±1일 때 자세계에 표시하는 뱅크 각
ROTATE_DEG = 6.0         # 이륙 속도에서의 기수 들림 각 (느릴수록 더 든다)
PITCH_PX_PER_DEG = 4.0


# ---------------------------------------------------------------- 공통 도구

def text(img, s, org, size=0.5, color=WHITE, thick=1, anchor="l"):
    """anchor: l(왼쪽) / c(가운데) / r(오른쪽) 기준으로 글자를 그린다."""
    (tw, _), _ = cv2.getTextSize(s, FONT, size, thick)
    x, y = org
    if anchor == "c":
        x -= tw / 2
    elif anchor == "r":
        x -= tw
    cv2.putText(img, s, (int(x), int(y)), FONT, size, color, thick, AA)


def shade(img, x1, y1, x2, y2, color=(0, 0, 0), alpha=0.55):
    """반투명 사각형."""
    roi = img[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    roi[:] = cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0)


def fit_text(img, s, cx, y, size, color, thick, max_w):
    """가운데 정렬. 너비를 넘으면 글자 크기를 줄인다."""
    while size > 0.4 and cv2.getTextSize(s, FONT, size, thick)[0][0] > max_w:
        size -= 0.05
    x = int(cx - cv2.getTextSize(s, FONT, size, thick)[0][0] / 2)
    cv2.putText(img, s, (x, y), FONT, size, (0, 0, 0), thick + 3, AA)   # 외곽선
    cv2.putText(img, s, (x, y), FONT, size, color, thick, AA)


# ---------------------------------------------------------------- 기체 실루엣

# 기체 좌표: x = 오른쪽, y = 앞쪽, 기수 +1 ~ 꼬리 -1
_HALF = [(0, 1.0), (0.06, 0.92), (0.09, 0.75), (0.09, 0.26), (0.95, -0.20), (0.95, -0.32),
         (0.09, -0.10), (0.08, -0.66), (0.36, -0.90), (0.36, -0.99), (0.05, -0.88), (0, -1.0)]
_OUTLINE = np.array(_HALF + [(-x, y) for x, y in reversed(_HALF[1:-1])], np.float32)
_ENGINE = np.array([(0.29, 0.21), (0.41, 0.21), (0.41, 0.0), (0.29, 0.0)], np.float32)
_SHIFT = 4   # 1/16 픽셀 정밀도로 그려 회전할 때 떨림을 줄인다


def _body(pts, center, heading_deg, size, mirror=False):
    h = math.radians(heading_deg)
    fwd = np.array([math.sin(h), -math.cos(h)])
    right = np.array([math.cos(h), math.sin(h)]) * (-1 if mirror else 1)
    p = np.asarray(center, np.float64) + size * (pts[:, :1] * right + pts[:, 1:] * fwd)
    return np.round(p * (1 << _SHIFT)).astype(np.int32)


def draw_aircraft(img, center, heading_deg, size, fill=WHITE, outline=(70, 70, 70), lights=True):
    for mirror in (False, True):
        cv2.fillPoly(img, [_body(_ENGINE, center, heading_deg, size, mirror)],
                     (175, 175, 180), AA, _SHIFT)
    body = _body(_OUTLINE, center, heading_deg, size)
    cv2.fillPoly(img, [body], fill, AA, _SHIFT)
    cv2.polylines(img, [body], True, outline, 1, AA, _SHIFT)
    if lights:   # 조종석 창, 항법등 (왼쪽 빨강 / 오른쪽 초록)
        pts = _body(np.array([(-0.045, 0.80), (0.045, 0.80), (-0.93, -0.25), (0.93, -0.25)],
                             np.float32), center, heading_deg, size)
        cv2.line(img, tuple(pts[0]), tuple(pts[1]), (90, 70, 50), 2, AA, _SHIFT)
        cv2.circle(img, tuple(pts[2]), 2 << _SHIFT, RED, -1, AA, _SHIFT)
        cv2.circle(img, tuple(pts[3]), 2 << _SHIFT, GREEN, -1, AA, _SHIFT)


# ---------------------------------------------------------------- 지도

_TILE = 512


def _make_ground(seed=7):
    """이어 붙여도 경계가 보이지 않는 잔디 질감. 한 번만 만든다."""
    rng = np.random.default_rng(seed)

    def layer(sigma):
        n = rng.standard_normal((_TILE, _TILE)).astype(np.float32)
        n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), sigma)[_TILE:2 * _TILE, _TILE:2 * _TILE]
        return n / (n.std() + 1e-6)

    n = 0.7 * layer(40) + 0.4 * layer(8) + 0.15 * layer(1.5)
    base = np.array([50, 92, 58], np.float32)
    tint = np.array([6, 14, 10], np.float32)
    tile = np.clip(base + n[..., None] * tint, 0, 255).astype(np.uint8)
    return np.tile(tile, (3, 3, 1))


_GROUND = _make_ground()


def _stamp(img, s, center, size, color, flip=False):
    """작은 글자를 찍는다. flip이면 180도 돌려서 (반대편 활주로 번호)."""
    thick = max(1, int(round(size * 2.2)))
    (tw, th), base = cv2.getTextSize(s, FONT, size, thick)
    patch = np.zeros((th + base + 2, tw + 2), np.uint8)
    cv2.putText(patch, s, (1, th + 1), FONT, size, 255, thick, AA)
    if flip:
        patch = cv2.rotate(patch, cv2.ROTATE_180)
    ph, pw = patch.shape
    x0, y0 = center[0] - pw // 2, center[1] - ph // 2
    if x0 < 0 or y0 < 0 or x0 + pw > img.shape[1] or y0 + ph > img.shape[0]:
        return
    roi = img[y0:y0 + ph, x0:x0 + pw]
    a = patch[..., None].astype(np.float32) / 255.0
    roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)


def draw_airport(m, to_s, s):
    hw, length = RUNWAY_HALF_WIDTH, RUNWAY_LENGTH

    def rect(x1, y1, x2, y2, color):
        cv2.rectangle(m, to_s((x1, y2)), to_s((x2, y1)), color, -1)

    def line(p1, p2, color, thick=1):
        cv2.line(m, to_s(p1), to_s(p2), color, thick, AA)

    # 계류장과 건물
    rect(-300, 380, -150, 620, TAXIWAY)
    rect(-292, 440, -252, 560, (128, 124, 118))
    rect(-286, 446, -258, 554, (150, 146, 140))
    rect(-292, 584, -256, 612, (112, 108, 104))
    for y in (470, 500, 530):   # 주기장 표시선
        line((-240, y), (-200, y), (0, 190, 230))

    # 유도로
    rect(-140, 0, -118, length, TAXIWAY)
    for y in (20, 500, 980):
        rect(-140, y - 11, -hw, y + 11, TAXIWAY)
        line((-129, y), (-hw, y), (0, 190, 230))
    line((-129, 0), (-129, length), (0, 190, 230))
    rect(-150, 489, -140, 511, TAXIWAY)

    # 활주로 본체 (양 끝에 과주로)
    rect(-hw, -60, hw, 0, (66, 64, 62))
    rect(-hw, length, hw, length + 60, (66, 64, 62))
    rect(-hw, 0, hw, length, ASPHALT)
    for x in (-hw + 1.5, hw - 1.5):
        line((x, 0), (x, length), WHITE)

    # 중심선 점선
    thick = max(1, int(round(s * 1.2)))
    for y in np.arange(100, length - 100, 60):
        line((0, y), (0, y + 30), WHITE, thick)

    # 시단 표지, 접지대 표지, 목표점 표지 (양쪽 끝)
    for end, sign in ((0, 1), (length, -1)):
        at = lambda d: end + sign * d
        for i in range(8):
            x = -hw + 4 + i * (2 * hw - 8 - 3) / 7
            rect(x, min(at(6), at(36)), x + 3, max(at(6), at(36)), WHITE)
        for d in (150, 225, 450):
            for x1 in (-15, 9):
                rect(x1, min(at(d), at(d + 22)), x1 + 6, max(at(d), at(d + 22)), WHITE)
        for x1 in (-15, 6):
            rect(x1, min(at(300), at(345)), x1 + 9, max(at(300), at(345)), WHITE)

    # 활주로 번호: 북쪽(360도)으로 이륙하므로 남쪽 끝이 36, 반대편이 18
    size = float(np.clip(s * 0.95, 0.3, 3.0))
    _stamp(m, "36", to_s((0, 68)), size, WHITE)
    _stamp(m, "18", to_s((0, length - 68)), size, WHITE, flip=True)


def draw_targets(m, sc, to_s, s):
    """목표 경로: 실제 항법 화면처럼 마젠타색."""
    for shape in sc.targets():
        if shape[0] == "line":
            a, b = to_s(shape[1]), to_s(shape[2])
            cv2.line(m, a, b, (110, 0, 110), 5, AA)
            cv2.line(m, a, b, MAGENTA, 2, AA)
        elif shape[0] == "circle":
            c, r = to_s(shape[1]), int(round(shape[2] * s))
            cv2.circle(m, c, r, (110, 0, 110), 5, AA)
            cv2.circle(m, c, r, MAGENTA, 2, AA)
        elif shape[0] == "path":    # 꺾은선 경로 (장주 비행)
            pts = np.array([to_s(p) for p in shape[1]], np.int32)
            cv2.polylines(m, [pts], False, (110, 0, 110), 5, AA)
            cv2.polylines(m, [pts], False, MAGENTA, 2, AA)
        elif shape[0] == "point":   # 웨이포인트 (네 갈래 별)
            x, y = to_s(shape[1])
            star = np.array([(x, y - 13), (x + 4, y - 4), (x + 13, y), (x + 4, y + 4),
                             (x, y + 13), (x - 4, y + 4), (x - 13, y), (x - 4, y - 4)], np.int32)
            cv2.fillPoly(m, [star], MAGENTA, AA)
            cv2.polylines(m, [star], True, WHITE, 1, AA)


# ---------------------------------------------------------------- 밤

# 밤의 지면 밝기: 채널별(B, G, R)로 어둡게, 약간 푸르게
_NIGHT_GAIN = (0.30, 0.22, 0.17)
_NIGHT_LUT = np.stack([np.clip(np.arange(256) * g, 0, 255) for g in _NIGHT_GAIN], axis=1) \
    .astype(np.uint8).reshape(256, 1, 3)


def darken(img):
    """낮 그림을 밤 밝기로 바꾼다."""
    return cv2.LUT(img, _NIGHT_LUT)


def _make_airport_lights():
    """공항 등화: (x, y, 색 BGR, 크기 m). 밤에만 켠다."""
    hw, length = RUNWAY_HALF_WIDTH, RUNWAY_LENGTH
    white, warm = (235, 245, 255), (150, 215, 255)
    lights = []
    for y in range(0, int(length) + 1, 50):          # 활주로등 (양옆, 끝 300 m는 노란색)
        color = warm if y > length - 300 else white
        lights += [(-hw, y, color, 0.8), (hw, y, color, 0.8)]
    for y in range(25, int(length), 50):             # 중심선등
        lights.append((0.0, y, white, 0.5))
    for x in range(-20, 21, 5):                      # 시단등(초록)과 종단등(빨강)
        lights.append((x, 0.0, (90, 255, 90), 0.8))
        lights.append((x, length, (70, 70, 255), 0.8))
    for y in range(-30, -301, -30):                  # 진입등 (시단 앞 300 m)
        lights += [(x, y, white, 0.9) for x in (-3.0, 0.0, 3.0)]
    lights += [(x, -150.0, white, 0.9) for x in range(-18, 19, 6)]
    for y in range(0, int(length) + 1, 100):         # 유도로등 (파랑)
        lights += [(-140.0, y, (255, 120, 40), 0.5), (-118.0, y, (255, 120, 40), 0.5)]
    for y in (400, 460, 520, 580):                   # 계류장 조명
        lights.append((-150.0, y, warm, 1.0))
    return lights


AIRPORT_LIGHTS = _make_airport_lights()


def blend_circle(img, center, radius, color, alpha):
    """반투명 원 (연기용). 화면 밖이면 그리지 않는다."""
    x, y, r = int(center[0]), int(center[1]), int(max(radius, 1))
    x1, y1, x2, y2 = max(x - r - 1, 0), max(y - r - 1, 0), min(x + r + 2, img.shape[1]), min(y + r + 2, img.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    roi = img[y1:y2, x1:x2]
    layer = roi.copy()
    cv2.circle(layer, (x - x1, y - y1), r, color, -1, AA)
    roi[:] = cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0)


def draw_map(sc, ac, trail, decorations=True, fx=None, night=False):
    """위에서 내려다본 2D 지도. decorations=False면 북쪽 표시·축척 막대를 뺀다 (미니맵용)."""
    s = sc.view_scale
    cx, cy = MAP_C
    to_s = lambda p: (int(round(cx + (p[0] - ac.x) * s)), int(round(cy - (p[1] - ac.y) * s)))

    # 지면: 질감을 기체 위치만큼 밀어서 움직임이 느껴지게 한다
    ox = int(math.floor(ac.x * s - cx)) % _TILE
    oy = int(math.floor(-ac.y * s - cy)) % _TILE
    m = _GROUND[oy:oy + CANVAS_H, ox:ox + MAP_W].copy()

    # 250 m 격자
    step = 250.0
    gx = math.floor((ac.x - cx / s) / step) * step
    while (gx - ac.x) * s + cx < MAP_W:
        x = int(round(cx + (gx - ac.x) * s))
        cv2.line(m, (x, 0), (x, CANVAS_H), (62, 104, 70), 1)
        gx += step
    gy = math.floor((ac.y - cy / s) / step) * step
    while cy - (gy - ac.y) * s > 0:
        y = int(round(cy - (gy - ac.y) * s))
        cv2.line(m, (0, y), (MAP_W, y), (62, 104, 70), 1)
        gy += step

    # 도시: 지도 축척에 맞는 축소 단계의 그림을 옮겨 붙인다
    level = next((k for k in (0, 1, 2) if scenery.MPP * (1 << k) * s >= 0.75), 2)
    k = scenery.MPP * (1 << level) * s
    affine = np.array([[k, 0, cx + (scenery.X0 - ac.x) * s], [0, k, cy - (scenery.Y1 - ac.y) * s]])
    cv2.warpAffine(scenery.CITY["day"][level], affine, (MAP_W, CANVAS_H), dst=m,
                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    draw_airport(m, to_s, s)
    if night:   # 밤: 지면을 어둡게 하고 불빛을 켠다 (경로·기체 표시는 그대로 밝게)
        m = cv2.add(darken(m), cv2.warpAffine(scenery.CITY["emission"][level], affine,
                                              (MAP_W, CANVAS_H), flags=cv2.INTER_LINEAR))
        for lx, ly, color, size in AIRPORT_LIGHTS:
            cv2.circle(m, to_s((lx, ly)), max(1, int(round(size * s * 2.5))), color, -1, AA)
    draw_targets(m, sc, to_s, s)

    if len(trail) > 1:   # 비행 궤적
        cv2.polylines(m, [np.array([to_s(p) for p in trail], np.int32)], False, CYAN, 2, AA)

    if fx is not None:   # 이륙·접지 지점 표시와 타이어 연기
        for mark, label, color in ((fx.liftoff, "LIFTOFF", GREEN), (fx.touchdown, "TOUCHDOWN", WHITE)):
            if mark:
                px, py = to_s(mark)
                cv2.circle(m, (px, py), 6, (0, 0, 0), -1, AA)
                cv2.circle(m, (px, py), 4, color, -1, AA)
                if decorations:
                    text(m, label, (px + 10, py + 4), 0.4, color)
        for px, py, _z, radius, alpha in fx.smoke():
            blend_circle(m, to_s((px, py)), radius * s, (225, 225, 225), alpha)

    # 그림자: 고도가 높을수록 기체에서 멀어지고 옅어진다 (밤에는 없음)
    size = 24
    if not night:
        d = min(ac.alt * 0.15, 45.0)
        shadow = m.copy()
        draw_aircraft(shadow, (cx + d, cy + d), ac.heading, size, (15, 25, 15), (15, 25, 15), False)
        alpha = 0.45 - 0.2 * min(ac.alt / 200.0, 1.0)
        cv2.addWeighted(shadow, alpha, m, 1 - alpha, 0, m)
    draw_aircraft(m, (cx, cy), ac.heading, size)
    if decorations:
        _draw_map_decorations(m, s)
    return m


def _draw_map_decorations(m, s):
    # 북쪽 표시 (오른쪽 위), 축척 막대 (왼쪽 아래)
    nx, ny = MAP_W - 32, 44
    cv2.circle(m, (nx, ny), 20, (0, 0, 0), -1, AA)
    cv2.circle(m, (nx, ny), 20, GRAY, 1, AA)
    cv2.fillPoly(m, [np.array([(nx, ny - 15), (nx + 6, ny + 2), (nx - 6, ny + 2)], np.int32)],
                 WHITE, AA)
    text(m, "N", (nx, ny + 15), 0.38, WHITE, 1, "c")
    bar_m = next((v for v in (100, 200, 500, 1000) if v * s >= 60), 1000)
    bx, by, bw = 16, CANVAS_H - 18, int(bar_m * s)
    shade(m, bx - 6, by - 24, bx + bw + 6, by + 8)
    cv2.line(m, (bx, by), (bx + bw, by), WHITE, 2)
    for x in (bx, bx + bw):
        cv2.line(m, (x, by - 5), (x, by + 1), WHITE, 2)
    text(m, f"{bar_m} m", (bx + bw // 2, by - 8), 0.42, WHITE, 1, "c")


def draw_overlays(m, sc, ac, phase, t, info, fx=None):
    """과제 정보 상자, 비행 단계, 이륙·접지 알림, 단계별 안내. 2D 지도와 3D 화면이 같이 쓴다."""
    cx = MAP_C[0]
    # 과제 정보 (왼쪽 위)
    shade(m, 10, 10, 292, 96)
    cv2.rectangle(m, (10, 10), (292, 96), DIM, 1)
    text(m, sc.name.upper(), (20, 34), 0.65, WHITE, 2)
    if info.get("trial_no") is not None:
        text(m, f"{info.get('trial_label', 'TRIAL')} {info['trial_no']}", (282, 34), 0.5, GRAY, 1, "r")
    text(m, "TIME", (20, 60), 0.45, GRAY)
    if sc.timeout is None:   # 자유 비행: 제한 시간 없음
        text(m, f"{t:5.1f} s", (282, 60), 0.5, WHITE, 1, "r")
    else:
        text(m, f"{t:5.1f} / {sc.timeout:.0f} s", (282, 60), 0.5, WHITE, 1, "r")
        frac = min(t / sc.timeout, 1.0)
        cv2.rectangle(m, (20, 66), (282, 70), (70, 70, 70), -1)
        cv2.rectangle(m, (20, 66), (20 + int(262 * frac), 70), AMBER if frac > 0.8 else GREEN, -1)
    text(m, sc.error_label, (20, 88), 0.45, GRAY)
    text(m, f"{sc.path_error(ac):6.1f} m", (282, 88), 0.5, MAGENTA, 1, "r")

    if fx is not None:
        # 비행 단계 (정보 상자 아래)
        shade(m, 10, 100, 292, 126)
        cv2.rectangle(m, (10, 100), (292, 126), DIM, 1)
        text(m, "PHASE", (20, 118), 0.45, GRAY)
        text(m, fx.phase(ac), (282, 118), 0.5, GREEN, 1, "r")
        if getattr(sc, "progress", None) is not None:   # 장주 진행률
            cv2.rectangle(m, (10, 128), (292, 131), (70, 70, 70), -1)
            cv2.rectangle(m, (10, 128), (10 + int(282 * sc.progress), 131), MAGENTA, -1)
        # 실속 경고 (깜빡임), 이륙·접지 알림과 고도 호출 (잠깐 크게 표시)
        if ac.airborne and ac.stalled and phase != "ready":
            if int(fx.clock * 4) % 2 == 0:
                fit_text(m, "STALL", cx, 175, 1.5, RED, 3, MAP_W - 30)
        elif fx.callout and phase != "ready":
            word, color = fx.callout
            fit_text(m, word, cx, 175, 1.3, color, 3, MAP_W - 30)

    # 단계별 안내
    if phase == "ready":
        shade(m, 0, 205, MAP_W, 320, alpha=0.5)
        fit_text(m, "GET READY", cx, 258, 1.5, YELLOW, 3, MAP_W - 30)
        fit_text(m, sc.instruction, cx, 300, 0.7, WHITE, 1, MAP_W - 30)
    elif phase == "run" and t < 1.0 and sc.timeout is not None:   # 자유 비행에는 출발 신호가 없다
        fit_text(m, "GO!", cx, 285, 2.5, GREEN, 5, MAP_W - 30)
    elif phase == "result":
        lines = [sc.reason] if sc.reason else []
        if sc.touchdown:   # 접지 순간 수치
            td = sc.touchdown
            lines.append(f"SINK {td['sink']:.1f} m/s    SPEED {td['speed']:.0f} m/s    "
                         f"OFFSET {td['offset']:+.1f} m")
        shade(m, 0, 205, MAP_W, 320 if len(lines) < 2 else 350, alpha=0.5)
        color = GREEN if sc.outcome == "success" else RED
        title = getattr(sc, "result_title", "") or sc.outcome.upper()
        fit_text(m, title, cx, 258, 1.5, color, 3, MAP_W - 30)
        for i, line in enumerate(lines):
            fit_text(m, line, cx, 300 + i * 32, 0.7, WHITE, 1, MAP_W - 30)


# ---------------------------------------------------------------- 계기판

def draw_adi(img, x, y, w, h, bank_deg, pitch_deg):
    """자세계(인공 수평선). 오른쪽으로 기울이면 수평선은 반시계 방향으로 돈다."""
    size = int(math.hypot(w, h)) + 40
    c = size // 2
    ppd = PITCH_PX_PER_DEG
    hy = int(round(c + pitch_deg * ppd))
    patch = np.empty((size, size, 3), np.uint8)
    patch[:] = SKY
    patch[max(hy, 0):] = EARTH
    cv2.line(patch, (0, hy), (size, hy), WHITE, 2, AA)
    for deg in range(-30, 31, 5):   # 피치 눈금
        if deg == 0:
            continue
        yy = int(round(hy - deg * ppd))
        if abs(yy - c) > h // 2 - 36:   # 뱅크 눈금과 겹치는 바깥쪽은 그리지 않는다
            continue
        half = 24 if deg % 10 == 0 else 11
        cv2.line(patch, (c - half, yy), (c + half, yy), WHITE, 1, AA)
        if deg % 10 == 0:
            text(patch, str(abs(deg)), (c - half - 6, yy + 4), 0.35, WHITE, 1, "r")
            text(patch, str(abs(deg)), (c + half + 6, yy + 4), 0.35, WHITE, 1)
    r = h // 2 - 8   # 뱅크 눈금 (수평선과 함께 회전)
    for a in (-45, -30, -20, -10, 0, 10, 20, 30, 45):
        sa, ca = math.sin(math.radians(a)), math.cos(math.radians(a))
        n = 12 if a % 30 == 0 else 7
        cv2.line(patch, (int(c + r * sa), int(c - r * ca)),
                 (int(c + (r - n) * sa), int(c - (r - n) * ca)), WHITE, 2 if a == 0 else 1, AA)

    rot = cv2.getRotationMatrix2D((c, c), bank_deg, 1.0)
    patch = cv2.warpAffine(patch, rot, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    img[y:y + h, x:x + w] = patch[c - h // 2:c - h // 2 + h, c - w // 2:c - w // 2 + w]

    cx, cy = x + w // 2, y + h // 2
    # 고정 뱅크 지침 (위쪽 삼각형)
    top = cy - r + 14
    cv2.fillPoly(img, [np.array([(cx, top), (cx - 6, top + 10), (cx + 6, top + 10)], np.int32)],
                 YELLOW, AA)
    # 고정 기체 기호
    for sign in (-1, 1):
        wing = np.array([(cx + sign * 52, cy), (cx + sign * 20, cy), (cx + sign * 20, cy + 8)],
                        np.int32)
        cv2.polylines(img, [wing], False, (0, 0, 0), 6, AA)
        cv2.polylines(img, [wing], False, YELLOW, 3, AA)
    cv2.circle(img, (cx, cy), 4, (0, 0, 0), -1, AA)
    cv2.circle(img, (cx, cy), 3, YELLOW, -1, AA)
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), DIM, 1)


def draw_tape(img, x, y, w, h, value, px_per_unit, major, minor, ticks_right, bug=None,
              ground=False):
    """세로로 흐르는 속도/고도 테이프. 가운데 상자가 현재 값."""
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), INSTR, -1)
    cy = y + h // 2
    to_y = lambda v: int(round(cy - (v - value) * px_per_unit))
    if ground and to_y(0) < y + h:   # 0 m 아래는 지면
        cv2.rectangle(img, (x, max(to_y(0), y)), (x + w - 1, y + h - 1), EARTH, -1)

    edge = x + w - 1 if ticks_right else x
    sign = -1 if ticks_right else 1
    v = math.floor((value - h / 2 / px_per_unit) / minor) * minor
    while v <= value + h / 2 / px_per_unit:
        yy = to_y(v)
        if v >= 0 and y + 2 <= yy <= y + h - 3:
            is_major = int(round(v)) % major == 0
            cv2.line(img, (edge, yy), (edge + sign * (10 if is_major else 5), yy), WHITE, 1)
            if is_major and y + 9 <= yy <= y + h - 6:
                text(img, f"{v:.0f}", (edge + sign * 14, yy + 4), 0.42, WHITE, 1,
                     "r" if ticks_right else "l")
        v += minor

    if bug is not None:   # 목표값 표시 (마젠타)
        by = int(np.clip(to_y(bug), y + 5, y + h - 6))
        tri = np.array([(edge, by), (edge + sign * 9, by - 6), (edge + sign * 9, by + 6)], np.int32)
        cv2.fillPoly(img, [tri], MAGENTA, AA)

    cv2.rectangle(img, (x + 2, cy - 14), (x + w - 3, cy + 14), (0, 0, 0), -1)
    cv2.rectangle(img, (x + 2, cy - 14), (x + w - 3, cy + 14), WHITE, 1)
    text(img, f"{value:.0f}", (x + w // 2, cy + 7), 0.6, GREEN, 2, "c")
    cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), DIM, 1)


def draw_hsi(img, c, r, heading, wind_from=None):
    """방위 지시계: 눈금판이 돌고 위쪽 지침이 현재 기수 방위."""
    cx, cy = c
    cv2.circle(img, c, r + 6, INSTR, -1, AA)
    cv2.circle(img, c, r + 6, DIM, 1, AA)
    for deg in range(0, 360, 10):
        a = math.radians(deg - heading)
        sa, ca = math.sin(a), math.cos(a)
        n = 9 if deg % 30 == 0 else 4
        cv2.line(img, (int(round(cx + r * sa)), int(round(cy - r * ca))),
                 (int(round(cx + (r - n) * sa)), int(round(cy - (r - n) * ca))), WHITE, 1, AA)
        if deg % 30 == 0:
            cardinal = {0: "N", 90: "E", 180: "S", 270: "W"}.get(deg)
            text(img, cardinal or str(deg // 10),
                 (cx + (r - 20) * sa, cy - (r - 20) * ca + 5),
                 0.45 if cardinal else 0.38, YELLOW if cardinal else WHITE, 1, "c")
    draw_aircraft(img, c, 0, 15, (0, 0, 0), WHITE, False)
    if wind_from is not None:   # 바람 화살표: 불어오는 쪽 가장자리에서 중심을 향한다
        a = math.radians(wind_from - heading)
        sa, ca = math.sin(a), math.cos(a)
        tail = (int(round(cx + (r - 6) * sa)), int(round(cy - (r - 6) * ca)))
        head = (int(round(cx + (r - 30) * sa)), int(round(cy - (r - 30) * ca)))
        cv2.arrowedLine(img, tail, head, CYAN, 2, AA, tipLength=0.45)
    top = cy - r - 6
    cv2.fillPoly(img, [np.array([(cx, top + 9), (cx - 6, top - 2), (cx + 6, top - 2)], np.int32)],
                 YELLOW, AA)
    cv2.rectangle(img, (cx - 24, top - 24), (cx + 24, top - 4), (0, 0, 0), -1)
    cv2.rectangle(img, (cx - 24, top - 24), (cx + 24, top - 4), WHITE, 1)
    text(img, f"{heading % 360:03.0f}", (cx, top - 9), 0.5, GREEN, 1, "c")


def draw_throttle(img, x, y, h, throttle):
    text(img, "THR", (x, y - 8), 0.42, GRAY)
    w = 20
    cv2.rectangle(img, (x, y), (x + w, y + h), INSTR, -1)
    fill = int(round(h * throttle / 100.0))
    cv2.rectangle(img, (x + 1, y + h - fill), (x + w - 1, y + h), GREEN, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    for pct in (0, 25, 50, 75, 100):
        yy = y + h - int(round(h * pct / 100.0))
        cv2.line(img, (x + w, yy), (x + w + 5, yy), WHITE, 1)
        text(img, str(pct), (x + w + 8, yy + 4), 0.35, GRAY)
    ly = y + h - fill   # 레버 손잡이
    cv2.rectangle(img, (x - 4, ly - 4), (x + w + 4, ly + 4), WHITE, -1)
    cv2.rectangle(img, (x - 4, ly - 4), (x + w + 4, ly + 4), (60, 60, 60), 1)
    text(img, f"{throttle:.0f}%", (x - 4, y + h + 22), 0.5, GREEN)


def draw_vsi(img, x, y, h, vs):
    """승강계. 빨간 띠는 착륙 접지 한계 하강률을 넘는 구간."""
    text(img, "VS", (x - 2, y - 8), 0.42, GRAY)
    w = 12
    cy = y + h // 2
    ppu = (h / 2) / MAX_VS
    cv2.rectangle(img, (x, y), (x + w, y + h), INSTR, -1)
    cv2.rectangle(img, (x - 4, int(cy + LAND_MAX_SINK * ppu)), (x - 2, y + h), RED, -1)
    vy = int(round(cy - np.clip(vs, -MAX_VS, MAX_VS) * ppu))
    color = GREEN if vs >= 0 else (RED if -vs > LAND_MAX_SINK else AMBER)
    cv2.rectangle(img, (x + 1, min(cy, vy)), (x + w - 1, max(cy, vy)), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    for v in (-10, -5, 0, 5, 10):
        yy = int(round(cy - v * ppu))
        cv2.line(img, (x + w, yy), (x + w + 4, yy), WHITE, 1)
        text(img, f"{v:+d}" if v else "0", (x + w + 7, yy + 4), 0.35, GRAY)
    text(img, f"{vs:+.1f}", (x - 6, y + h + 22), 0.5, color)


def draw_axis_input(img, x, y, w, label, value, deadzone, ends, hand):
    """조종 입력 한 축을 가로 막대로 표시. deadzone = 중립 범위(0~1), ends = 양 끝 글자."""
    color = CYAN if hand else DIM
    text(img, label, (x, y - 7), 0.42, GRAY)
    text(img, f"{value:+.2f}", (x + w, y - 7), 0.45, color, 1, "r")
    cx, half, h = x + w // 2, w // 2, 14
    cv2.rectangle(img, (x, y), (x + w, y + h), INSTR, -1)
    dz = int(half * deadzone)
    cv2.rectangle(img, (cx - dz, y + 1), (cx + dz, y + h - 1), (70, 66, 62), -1)
    px = int(cx + value * half)
    cv2.rectangle(img, (min(cx, px), y + 2), (max(cx, px), y + h - 2), color, -1)
    cv2.line(img, (cx, y - 3), (cx, y + h + 3), WHITE, 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    text(img, ends[0], (x - 1, y + h + 14), 0.35, GRAY)
    text(img, ends[1], (x + w, y + h + 14), 0.35, GRAY, 1, "r")


def draw_control_inputs(img, x, y, w, ctrl, hand):
    """손 입력 표시: 왼쪽 = 좌우 기울기(roll), 오른쪽 = 앞뒤 기울기(pitch, 오른쪽이 상승)."""
    half = (w - 16) // 2
    draw_axis_input(img, x, y, half, "ROLL", ctrl.roll, ctrl.deadzone / ctrl.max_roll, ("L", "R"), hand)
    draw_axis_input(img, x + half + 16, y, half, "PITCH", ctrl.pitch,
                    ctrl.pitch_deadzone / ctrl.max_pitch, ("DN", "UP"), hand)


def attitude(ac, ctrl):
    """표시용 (뱅크, 피치) 목표 각도.

    피치 = 비행 경로각 + 받음각. 받음각은 느릴수록 커서 이륙 직후와 착륙 직전에 기수가 들린다.
    지상에서는 이륙 속도에 가까워질 때 기수를 들고(로테이션), 접지 후 감속하면 내린다.
    """
    if not ac.airborne:
        return 0.0, ROTATE_DEG * float(np.clip((ac.speed / V_LEVEL - 0.8) / 0.2, 0.0, 1.0))
    aoa = min(ROTATE_DEG * (V_LEVEL / max(ac.speed, 1.0)) ** 2, 10.0)
    # 환경 모델에서는 기체가 실제 뱅크 각을 갖는다. 단순 모델에서는 입력에 비례한 표시용 각.
    bank = ac.bank if ac.physical else ctrl.roll * MAX_BANK_DEG
    # pitch 조종을 당기고 밀 때 기수가 바로 움직여 보이도록 조금 더한다
    return bank, (math.degrees(math.atan2(ac.vs, max(ac.speed, 1.0))) + aoa
                  + 4.0 * getattr(ctrl, "pitch", 0.0))


class FlightEffects:
    """이륙·착륙 연출용 상태. 표시 전용이며 기록되는 측정값에는 쓰지 않는다.

    - 비행 단계 이름 (TAKEOFF ROLL, ROTATE, LIFTOFF, CLIMB, ... TOUCHDOWN, ROLLOUT)
    - 이륙·접지 지점, 접지 순간의 타이어 연기
    - 이륙·접지 알림과 착륙 접근 중 고도 호출 (50, 30, 20, 10)
    - 부드럽게 이어지는 자세 각도 (bank, pitch)
    """
    CALLOUT_ALTS = (50, 30, 20, 10)
    SMOKE_SECONDS = 2.2

    def __init__(self, ac, ctrl, env=None):
        self.env = env           # 바람·난기류 (없으면 None). 풍향 표시와 흔들림에 쓴다
        self.clock = 0.0
        self.liftoff = None      # (x, y)
        self.touchdown = None    # (x, y)
        self._liftoff_t = self._touchdown_t = -1e9
        self._puffs = []         # (x, y, 생성 시각)
        self._callout = None     # (문구, 색, 끝나는 시각)
        self._prev_alt = ac.alt
        self.bank, self.pitch = attitude(ac, ctrl)

    def update(self, ac, ctrl, event, dt):
        self.clock += dt
        if event == "liftoff":
            self.liftoff, self._liftoff_t = (ac.x, ac.y), self.clock
            self._say("LIFTOFF", GREEN, 1.8)
        elif event == "touchdown":
            self.touchdown, self._touchdown_t = (ac.x, ac.y), self.clock
            self._say("TOUCHDOWN", WHITE, 1.8)
            h = math.radians(ac.heading)   # 주 착륙 장치 두 바퀴 위치에 연기
            for side in (-2.8, 2.8):
                self._puffs.append((ac.x + side * math.cos(h) - 1.5 * math.sin(h),
                                    ac.y - side * math.sin(h) - 1.5 * math.cos(h), self.clock))
        elif ac.airborne and ac.vs < 0:
            for alt in self.CALLOUT_ALTS:
                if self._prev_alt > alt >= ac.alt:
                    self._say(str(alt), AMBER, 0.9)
        self._prev_alt = ac.alt
        self._puffs = [p for p in self._puffs if self.clock - p[2] < self.SMOKE_SECONDS]

        bank, pitch = attitude(ac, ctrl)   # 자세가 뚝 끊기지 않게 서서히 따라간다
        if self.env is not None and ac.airborne:   # 난기류에 기체가 흔들리는 모습
            h = math.radians(ac.heading)
            bank += 2.0 * (self.env.gust[0] * math.cos(h) - self.env.gust[1] * math.sin(h))
            pitch += 1.5 * self.env.updraft(ac.alt)
        self.bank += (bank - self.bank) * min(dt / 0.15, 1.0)
        self.pitch += (pitch - self.pitch) * min(dt / 0.45, 1.0)

    def _say(self, word, color, seconds):
        self._callout = (word, color, self.clock + seconds)

    @property
    def callout(self):
        if self._callout and self.clock < self._callout[2]:
            return self._callout[:2]
        return None

    def smoke(self):
        """연기 덩어리들: (x, y, 높이 m, 반지름 m, 불투명도). 시간이 지나면 커지며 옅어진다."""
        out = []
        for x, y, born in self._puffs:
            age = (self.clock - born) / self.SMOKE_SECONDS
            out.append((x, y, 1.0 + 4.0 * age, 2.5 + 9.0 * age, 0.6 * (1.0 - age)))
        return out

    def phase(self, ac) -> str:
        if ac.airborne and ac.stalled:
            return "STALL"
        if ac.airborne:
            if self.clock - self._liftoff_t < 2.0:
                return "LIFTOFF"
            if ac.vs > 1.0:
                return "CLIMB"
            if ac.vs < -1.0:
                return "SHORT FINAL" if ac.alt < 20 else ("FINAL APPROACH" if ac.alt < 80 else "DESCENT")
            return "CRUISE"
        if self.touchdown:
            if self.clock - self._touchdown_t < 1.5:
                return "TOUCHDOWN"
            return "ROLLOUT" if ac.speed > 2.0 else "STOPPED"
        if ac.speed < 1.0:
            return "HOLDING"
        return "ROTATE" if ac.speed >= 0.8 * V_LEVEL else "TAKEOFF ROLL"


def draw_panel(img, sc, ac, ctrl, cam, hand, info, fx=None):
    x0 = MAP_W
    cv2.rectangle(img, (x0, 0), (CANVAS_W, CANVAS_H), PANEL, -1)
    cv2.line(img, (x0, 0), (x0, CANVAS_H), (80, 80, 80), 2)

    text(img, str(info["participant"]), (x0 + 12, 26), 0.6, WHITE, 2)
    text(img, str(info.get("subtitle") or f"{str(info['mode']).upper()} CONTROL"), (x0 + 12, 44),
         0.38, GRAY)
    env = fx.env if fx is not None else None
    if env is not None:   # 바람과 대지속도 (환경 모델 사용 시)
        wind = (f"WIND {env.wind_from:03.0f} / {env.wind_speed:.0f} m/s" if env.wind_speed > 0
                else "WIND CALM")
        text(img, wind, (CANVAS_W - 12, 24), 0.45, CYAN, 1, "r")
        text(img, f"GS {ac.ground_speed:.0f} m/s   {env.level.upper()}", (CANVAS_W - 12, 44), 0.38,
             GRAY, 1, "r")

    # 주 비행 표시: 속도 테이프 | 자세계 | 고도 테이프
    py, ph = 56, 184
    bank, pitch = (fx.bank, fx.pitch) if fx is not None else attitude(ac, ctrl)
    draw_tape(img, x0 + 10, py, 54, ph, ac.speed, 4.0, 10, 5, True, bug=V_LEVEL)
    draw_adi(img, x0 + 70, py, 156, ph, bank, pitch)
    draw_tape(img, x0 + 232, py, 58, ph, ac.alt, 2.0, 20, 10, False,
              bug=getattr(sc, "TARGET_ALT", None), ground=True)
    text(img, "SPD m/s", (x0 + 10, py + ph + 15), 0.38, GRAY)
    text(img, "ALT m", (CANVAS_W - 10, py + ph + 15), 0.38, GRAY, 1, "r")

    draw_hsi(img, (x0 + 86, 356), 58, ac.heading,
             env.wind_from if env is not None and env.wind_speed > 0 else None)
    draw_throttle(img, x0 + 184, 290, 104, ctrl.throttle)
    draw_vsi(img, x0 + 248, 290, 104, ac.vs)

    draw_control_inputs(img, x0 + 14, 446, 272, ctrl, hand)

    # 손 인식 상태등: 놓치면 빨갛게 깜빡인다
    ay = 482
    if info.get("autopilot"):   # 자동 조종: 손 인식 대신 자동 조종 표시
        cv2.rectangle(img, (x0 + 12, ay), (CANVAS_W - 12, ay + 22), (90, 60, 20), -1)
        text(img, "AUTOPILOT ENGAGED", (x0 + 150, ay + 16), 0.45, CYAN, 1, "c")
        text(img, "NO CAMERA IN AUTOPILOT MODE", (x0 + 150, 606), 0.4, DIM, 1, "c")
    elif hand:
        cv2.rectangle(img, (x0 + 12, ay), (CANVAS_W - 12, ay + 22), (30, 70, 30), -1)
        text(img, "HAND TRACKING", (x0 + 150, ay + 16), 0.45, GREEN, 1, "c")
    else:
        on = int(time.monotonic() * 3) % 2 == 0
        cv2.rectangle(img, (x0 + 12, ay), (CANVAS_W - 12, ay + 22), RED if on else (30, 30, 90), -1)
        text(img, "NO HAND", (x0 + 150, ay + 16), 0.45, WHITE, 1, "c")

    # 카메라 미리보기 (화면에만 표시, 저장하지 않음)
    if cam is not None:
        pw, phh = 240, 180
        cx0, cy0 = x0 + (CANVAS_W - x0 - pw) // 2, 512
        img[cy0:cy0 + phh, cx0:cx0 + pw] = cv2.resize(cam, (pw, phh), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img, (cx0 - 1, cy0 - 1), (cx0 + pw, cy0 + phh), DIM, 1)


def compose(scene, sc, ac, ctrl, cam, hand, info, fx=None):
    """왼쪽 화면(scene: 2D 지도 또는 3D 화면)에 계기판을 붙인다."""
    img = np.empty((CANVAS_H, CANVAS_W, 3), np.uint8)
    img[:, :MAP_W] = scene
    draw_panel(img, sc, ac, ctrl, cam, hand, info, fx)
    return img
