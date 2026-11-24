"""3D 추적 화면과 시점 조작.

기체 뒤에서 따라가는 카메라로 지면·활주로·건물·기체를 원근 투영해 그린다.
별도 3D 라이브러리 없이 OpenCV + NumPy만 쓴다.
- 지면과 공항: 위에서 본 질감을 호모그래피로 지면에 깐다 (cv2.warpPerspective).
- 기체와 건물: 다각형을 먼 것부터 칠한다 (화가 알고리즘) + 단순 음영.
- 목표 경로는 지면 위 마젠타색 띠, 기체 아래에는 그림자와 수직선(지면 위치 표시).

시점 조작 (View)
    N 또는 화면 위 DAY/NIGHT 버튼 클릭   낮 <-> 밤 전환
    W 또는 그 옆 날씨 버튼 클릭         날씨 전환 (맑음 -> 흐림 -> 비 -> 뇌우 -> 안개 -> 눈)
    C                       조종석 1인칭 시점 <-> 뒤에서 따라가는 시점
                            (조종석에서는 드래그·방향키로 고개를 돌린다)
    마우스 왼쪽 드래그      카메라 회전 (좌우 = 둘레, 상하 = 높이)
    트랙패드 두 손가락 스크롤 (또는 마우스 휠)   카메라 회전
    Ctrl 또는 Option + 스크롤, + / -            확대 / 축소
    방향키 또는 4 6 8 2     카메라 회전
    1 뒤 / 3 옆 / 7 위 / 9 앞   미리 정한 시점
    5 또는 0, 오른쪽 클릭   기본 시점으로
    V                       3D <-> 2D 지도 전환

표시만 담당한다. 비행 모델과 기록되는 측정값에는 영향을 주지 않는다.
"""

import math
import sys

import cv2
import numpy as np

import cockpit as ck
import satellite
import scenery
from cockpit import AA, CANVAS_H, CYAN, MAGENTA, MAP_W, WHITE

W, H = MAP_W, CANVAS_H
FOCAL = (W / 2) / math.tan(math.radians(30))   # 수평 화각 60도
CX, CY = W / 2, H * 0.56                        # 기체가 화면 중앙보다 조금 아래에 오도록
NEAR = 4.0                                      # 이보다 가까운(또는 뒤쪽) 부분은 잘라낸다 (m)
GEAR_HEIGHT = 4.2                               # 바퀴가 지면에 닿을 때 기체 중심 높이 (m)

HAZE = np.array([228, 208, 184], np.float32)    # 지평선 근처 하늘색 (BGR)
ZENITH = np.array([190, 112, 44], np.float32)   # 높은 하늘색
SUN = np.array([0.35, 0.25, 0.9]) / np.linalg.norm([0.35, 0.25, 0.9])   # 빛이 오는 방향


# ---------------------------------------------------------------- 시점 상태

TIME_BUTTON = (296, 10, 420, 42)   # 화면 위 가운데의 낮/밤 전환 버튼 (x1, y1, x2, y2)
WEATHER_BUTTON = (428, 10, 566, 42)   # 그 옆의 날씨 전환 버튼
CAM_BUTTON = (574, 10, 692, 42)   # 그 옆의 시점 전환 버튼 (조종사 시점 <-> 외부 시점)

# 날씨별 화면 설정 (표시 전용. 바람·난기류는 --env 로 따로 정한다)
#   fog    멀리 흐려지는 거리 척도 (m): 작을수록 시야가 나쁘다
#   dim    지면·기체·건물의 밝기 배율 (흐리면 어둡다)
#   cover  구름 양 0~1,  cloud = 구름 색,  base = 구름 밑면 높이 (m)
#   sky    낮 하늘의 (지평선 색, 높은 하늘 색). None이면 맑은 하늘
#   sun    해·달·별이 보이는지
#   rain / snow  빗줄기·눈송이 수,  lightning  번개
WEATHERS = {
    "clear":  dict(label="CLEAR", fog=4500.0, dim=1.0, cover=0.22, cloud=(252, 250, 246), base=520.0,
                   sky=None, sun=True, rain=0, snow=0, lightning=False),
    "cloudy": dict(label="CLOUDY", fog=3600.0, dim=0.84, cover=0.75, cloud=(226, 224, 220), base=460.0,
                   sky=((214, 204, 192), (196, 160, 118)), sun=True, rain=0, snow=0, lightning=False),
    "rain":   dict(label="RAIN", fog=1900.0, dim=0.66, cover=1.0, cloud=(150, 148, 146), base=380.0,
                   sky=((168, 164, 158), (128, 124, 118)), sun=False, rain=260, snow=0, lightning=False),
    "storm":  dict(label="STORM", fog=1300.0, dim=0.5, cover=1.0, cloud=(92, 90, 90), base=340.0,
                   sky=((118, 114, 110), (70, 68, 66)), sun=False, rain=520, snow=0, lightning=True),
    "fog":    dict(label="FOG", fog=520.0, dim=0.9, cover=0.0, cloud=(210, 210, 210), base=400.0,
                   sky=((206, 206, 204), (190, 190, 188)), sun=False, rain=0, snow=0, lightning=False),
    "snow":   dict(label="SNOW", fog=1500.0, dim=0.95, cover=1.0, cloud=(200, 200, 202), base=400.0,
                   sky=((212, 212, 212), (176, 176, 178)), sun=False, rain=0, snow=420, lightning=False),
}
_wx = WEATHERS["clear"]   # 지금 그리고 있는 프레임의 날씨 (draw_scene이 정한다)

# 한글 입력 상태에서 누른 키: 같은 자리의 영문 키로 바꿔 준다 (n = ㅜ, v = ㅍ, r = ㄱ, q = ㅂ, w = ㅈ ...)
HANGUL_KEYS = {0x315C: ord("n"), 0x314D: ord("v"), 0x3131: ord("r"), 0x3142: ord("q"),
               0x3148: ord("w"), 0x314A: ord("c"), 0x3139: ord("f"), 0x314E: ord("g"),
               0x3154: ord("p"), 0x314C: ord("x"), 0x3145: ord("t"), 0x3141: ord("a"), 0x3157: ord("h"),
               0x3163: ord("l"), 0x314B: ord("z"), 0x3137: ord("e")}


def normalize_key(key: int) -> int:
    return HANGUL_KEYS.get(key, key)


class View:
    """카메라 시점. az = 기체 진행 방향 기준 둘레 각(0 = 바로 뒤), el = 올려다본 높이 각."""

    DEFAULT = (0.0, 18.0, 110.0)   # az(도), el(도), dist(m)
    SCROLL_DEG = 0.15              # 스크롤 1단위당 회전 각(도)
    PRESETS = {ord("1"): (0.0, 18.0), ord("3"): (90.0, 12.0),
               ord("7"): (0.0, 88.0), ord("9"): (180.0, 14.0)}
    # 방향키 코드는 운영체제마다 다르다 (macOS / Linux / Windows)
    LEFT = {63234, 65361, 2424832, ord("4")}
    RIGHT = {63235, 65363, 2555904, ord("6")}
    UP = {63232, 65362, 2490368, ord("8")}
    DOWN = {63233, 65364, 2621440, ord("2")}

    def __init__(self, mode="3d"):
        self.mode = mode
        self.night = False   # 밤 화면 여부 (N 키로 전환)
        self.weather = "clear"   # 날씨 (W 키로 다음 날씨)
        self.satellite = True    # 위성 사진 지면 (E 키로 모형 풍경과 전환). 사진이 없으면 모형 풍경을 그린다
        self.cam = "chase"       # chase = 뒤에서 따라가는 시점, cockpit = 조종석 1인칭 (C 키로 전환)
        self.look_az = self.look_el = 0.0   # 조종석에서 고개를 돌린 각도 (도)
        self.az, self.el, self.dist = self.DEFAULT
        self._drag = None

    def reset(self):
        self.az, self.el, self.dist = self.DEFAULT
        self.look_az = self.look_el = 0.0

    def orbit(self, d_az, d_el):
        if self.cam == "cockpit":   # 조종석에서는 고개를 돌린다 (좌우 ±110도, 상하 ±35도)
            self.look_az = float(np.clip(self.look_az - d_az * 0.6, -110.0, 110.0))
            self.look_el = float(np.clip(self.look_el + d_el * 0.6, -35.0, 35.0))
            return
        self.az = (self.az + d_az + 180.0) % 360.0 - 180.0
        self.el = float(np.clip(self.el + d_el, 3.0, 88.0))

    def zoom(self, factor):
        self.dist = float(np.clip(self.dist * factor, 40.0, 1500.0))

    def on_key(self, key) -> bool:
        """시점 조작 키면 처리하고 True."""
        if key in (ord("v"), ord("V")):
            self.mode = "2d" if self.mode == "3d" else "3d"
        elif key in (ord("n"), ord("N")):
            self.night = not self.night
        elif key in (ord("e"), ord("E")):
            self.satellite = not self.satellite
        elif key in (ord("w"), ord("W")):
            self.next_weather()
        elif key in (ord("a"), ord("A"), ord("c"), ord("C")):   # 조종사 시점 <-> 외부 시점
            self.toggle_cam()
        elif self.mode != "3d":
            return False
        elif key in self.LEFT:
            self.orbit(6.0, 0.0)
        elif key in self.RIGHT:
            self.orbit(-6.0, 0.0)
        elif key in self.UP:
            self.orbit(0.0, 4.0)
        elif key in self.DOWN:
            self.orbit(0.0, -4.0)
        elif key in (ord("+"), ord("=")):
            self.zoom(1 / 1.15)
        elif key in (ord("-"), ord("_")):
            self.zoom(1.15)
        elif key in (ord("5"), ord("0")):
            self.reset()
        elif key in self.PRESETS:
            self.az, self.el = self.PRESETS[key]
        else:
            return False
        return True

    def next_weather(self):
        names = list(WEATHERS)
        self.weather = names[(names.index(self.weather) + 1) % len(names)]

    def toggle_cam(self):
        """조종사 시점 <-> 외부 시점. 2D 지도를 보고 있었다면 3D 화면으로 돌아온다."""
        if self.mode != "3d":
            self.mode = "3d"
        else:
            self.cam = "cockpit" if self.cam == "chase" else "chase"

    def on_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:   # 낮/밤 버튼과 날씨 버튼 (2D·3D 공통)
            x1, y1, x2, y2 = TIME_BUTTON
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.night = not self.night
                return
            x1, y1, x2, y2 = WEATHER_BUTTON
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.next_weather()
                return
            x1, y1, x2, y2 = CAM_BUTTON
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.toggle_cam()
                return
        if self.mode != "3d":
            return
        if event == cv2.EVENT_LBUTTONDOWN and x < W:
            self._drag = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag and flags & cv2.EVENT_FLAG_LBUTTON:
            dx, dy = x - self._drag[0], y - self._drag[1]
            self.orbit(dx * 0.4, dy * 0.3)
            self._drag = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None
        elif event == cv2.EVENT_RBUTTONDOWN and x < W:
            self.reset()
        elif event in (cv2.EVENT_MOUSEWHEEL, cv2.EVENT_MOUSEHWHEEL):
            # 트랙패드 두 손가락 스크롤(또는 마우스 휠)
            dx, dy = self._scroll_delta(event, x, y, flags)
            if flags & (cv2.EVENT_FLAG_CTRLKEY | cv2.EVENT_FLAG_ALTKEY):   # Ctrl/Option + 스크롤 = 확대/축소
                self.zoom(math.exp(-float(np.clip(dy, -40, 40)) * 0.006))
            else:   # 좌우 스크롤 = 둘레 회전, 상하 스크롤 = 높이
                self.orbit(float(np.clip(dx * self.SCROLL_DEG, -12.0, 12.0)),
                           float(np.clip(dy * self.SCROLL_DEG, -12.0, 12.0)))

    @staticmethod
    def _scroll_delta(event, x, y, flags):
        """스크롤 양 (가로, 세로). OpenCV가 운영체제마다 다른 자리에 넣어 준다."""
        if sys.platform == "darwin":
            # macOS(Cocoa)에서는 스크롤 이벤트의 x, y 인자가 마우스 위치가 아니라 스크롤 양이다
            return x, y
        delta = (flags >> 16) & 0xFFFF   # 그 밖에는 flags 상위 16비트(부호 있음)
        if delta >= 0x8000:
            delta -= 0x10000
        return (delta, 0) if event == cv2.EVENT_MOUSEHWHEEL else (0, delta)

class Camera:
    """한 프레임의 카메라: 월드 좌표(x 동, y 북, z 위) -> 카메라 좌표 -> 화면."""

    def __init__(self, view, ac, pitch_deg=0.0):
        self.target = np.array([ac.x, ac.y, ac.alt + GEAR_HEIGHT])
        self.cockpit = view.cam == "cockpit"
        if self.cockpit:
            # 조종석: 기수 쪽 조종사 눈높이에서 기수 방향(+ 고개 돌린 각)을 본다.
            # 좌우 기울기(뱅크)는 여기서 넣지 않고, 다 그린 그림을 돌려서 표현한다.
            h = math.radians(ac.heading)
            self.eye = self.target + np.array([math.sin(h) * 16.0, math.cos(h) * 16.0, 1.6])
            a = math.radians(ac.heading + view.look_az)
            # 기수가 들려도 조종사는 계기판 너머로 앞 지면을 내려다본다: 시선은 기수의 절반만 따라간다.
            # (그대로 따라가면 상승 중에는 하늘만 보여 경로를 놓친다)
            e = math.radians(float(np.clip(COCKPIT_PITCH * pitch_deg + view.look_el, -60.0, 60.0)))
            f = np.array([math.sin(a) * math.cos(e), math.cos(a) * math.cos(e), math.sin(e)])
        else:
            a = math.radians(ac.heading + view.az)
            e = math.radians(view.el)
            back = np.array([-math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
            self.eye = self.target + back * view.dist
            self.eye[2] = max(self.eye[2], 1.5)
            f = self.target - self.eye
            f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        r /= np.linalg.norm(r)
        self.rot = np.array([r, np.cross(r, f), f])   # 행: 오른쪽, 위, 앞
        self.forward = f

    def to_cam(self, pts):
        return (np.asarray(pts, np.float64) - self.eye) @ self.rot.T

    @staticmethod
    def project(pc):
        """카메라 좌표 (N,3) -> 화면 좌표 (N,2). z > 0 이어야 한다."""
        return np.stack([CX + FOCAL * pc[:, 0] / pc[:, 2], CY - FOCAL * pc[:, 1] / pc[:, 2]], 1)

    def plane(self):
        """지면 좌표 (X, Y, 1) -> 화면 (3x3 행렬)."""
        k = np.array([[FOCAL, 0, CX], [0, -FOCAL, CY], [0, 0, 1.0]])
        plane = np.array([[1, 0, -self.eye[0]], [0, 1, -self.eye[1]], [0, 0, -self.eye[2]]])
        return k @ self.rot @ plane

    def ground_homography(self, ax, bx, ay, by):
        """질감 픽셀 (tx, ty) -> 지면 (X = ax*tx + bx, Y = ay*ty + by, 0) -> 화면."""
        k = np.array([[FOCAL, 0, CX], [0, -FOCAL, CY], [0, 0, 1.0]])
        plane = np.array([[1, 0, -self.eye[0]], [0, 1, -self.eye[1]], [0, 0, -self.eye[2]]])
        return k @ self.rot @ plane @ np.array([[ax, 0, bx], [0, ay, by], [0, 0, 1.0]])

    @property
    def horizon_y(self):
        fz = self.forward[2]
        return CY + FOCAL * fz / math.sqrt(max(1.0 - fz * fz, 1e-9))


def clip_near(pc):
    """카메라 좌표 다각형을 z >= NEAR 로 자른다 (Sutherland-Hodgman)."""
    out = []
    n = len(pc)
    for i in range(n):
        a, b = pc[i], pc[(i + 1) % n]
        a_in, b_in = a[2] >= NEAR, b[2] >= NEAR
        if a_in:
            out.append(a)
        if a_in != b_in:
            out.append(a + (b - a) * (NEAR - a[2]) / (b[2] - a[2]))
    return np.array(out) if len(out) >= 3 else None


def to_px(uv):
    """화면 좌표를 OpenCV가 받을 수 있는 정수로 (화면 밖으로 아주 멀리 나간 값은 제한)."""
    return np.round(np.clip(uv, -30000, 30000)).astype(np.int32)


def fill_world(img, cam, pts, color, aa=True):
    """월드 좌표 다각형 하나를 칠한다."""
    pc = clip_near(cam.to_cam(pts))
    if pc is not None:
        cv2.fillPoly(img, [to_px(cam.project(pc))], color, AA if aa else cv2.LINE_8)


def line_world(img, cam, p1, p2, color, thick=1):
    pc = cam.to_cam([p1, p2])
    if pc[0, 2] < NEAR and pc[1, 2] < NEAR:
        return
    for i, j in ((0, 1), (1, 0)):
        if pc[i, 2] < NEAR:
            pc[i] = pc[i] + (pc[j] - pc[i]) * (NEAR - pc[i, 2]) / (pc[j, 2] - pc[i, 2])
    a, b = to_px(cam.project(pc))
    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thick, AA)


# ---------------------------------------------------------------- 지면 질감

GRASS_MPP = 2.0   # 잔디 질감 1픽셀 = 2 m, 500픽셀(1 km)마다 반복


AP_PPM = 3.0                                   # 공항 질감 해상도 (픽셀/m)
AP_X0, AP_X1, AP_Y0, AP_Y1 = -540.0, 140.0, -140.0, 1140.0


def _make_airport():
    """활주로·유도로·계류장을 위에서 본 그림(BGR)과 덮을 영역(알파)."""
    w, h = int((AP_X1 - AP_X0) * AP_PPM), int((AP_Y1 - AP_Y0) * AP_PPM)
    key = np.array([50, 92, 58], np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = key
    to_s = lambda p: (int(round((p[0] - AP_X0) * AP_PPM)), int(round((AP_Y1 - p[1]) * AP_PPM)))
    ck.draw_airport(img, to_s, AP_PPM, planes=False)
    alpha = (np.abs(img.astype(np.int16) - key).sum(2) > 0).astype(np.uint8) * 255
    return np.dstack([img, alpha])


_GRASS = scenery.GRASS_TILE   # 도시 그림의 바탕과 같은 무늬라야 경계가 보이지 않는다
_AIRPORT = _make_airport()

# 건물 상자: (x1, y1, x2, y2, 높이, 벽 색, 지붕 색)
BUILDINGS = ck.AIRPORT_BUILDINGS
PARKED_MAX_DISTANCE = 2500.0   # 이보다 멀면 주기 중인 여객기를 입체로 그리지 않는다


# 도시 그림의 단계별 사용 거리: (축소 단계, 이 거리 안쪽에서만 사용 m). None = 전체
CITY_LEVELS = ((2, None), (1, 2100.0), (0, 1000.0))


def _warp_city(img, cam, levels, transparent):
    """도시 지면 그림을 원근에 맞게 깐다.

    먼 곳은 축소본, 가까운 곳(화면 아래쪽 행)은 원본을 써서 먼 곳이 자글거리지 않게 한다.
    transparent=True면 그림 바깥은 img를 그대로 두고, False면 검정(불빛 그림용).
    """
    hy = cam.horizon_y
    first = _first_ground_row(cam)
    for level, max_dist in CITY_LEVELS:
        y0 = first if max_dist is None else max(first, int(np.clip(math.ceil(hy + cam.eye[2] * FOCAL / max_dist), 0, H)))
        if y0 >= H:
            continue
        mpp = scenery.MPP * (1 << level)
        hom = cam.ground_homography(mpp, scenery.X0, -mpp, scenery.Y1)
        hom = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1.0]]) @ hom   # 아래쪽 행들만 그린다
        cv2.warpPerspective(levels[level], hom, (W, H - y0), dst=img[y0:], flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_TRANSPARENT if transparent else cv2.BORDER_CONSTANT)


def _first_ground_row(cam):
    """지면이 보이기 시작하는 화면 행 (지평선 바로 아래).

    지평선과 그 위의 화면 점은 지면의 무한히 먼 곳(또는 카메라 뒤)에 대응한다. 그런 점까지 지면
    그림을 깔려고 하면 좌표가 무한대가 되어 OpenCV가 비정상 종료하므로, 지평선 아래 행만 그린다.
    """
    return int(np.clip(math.floor(cam.horizon_y) + 2, 0, H))


def _make_ocean(seed=5, size=500):
    """이어 붙여도 경계가 보이지 않는 바다 질감: 짙은 파랑에 너울과 잔물결."""
    rng = np.random.default_rng(seed)

    def layer(sx, sy):
        n = rng.standard_normal((size, size)).astype(np.float32)
        n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), sigmaX=sx, sigmaY=sy)[size:2 * size, size:2 * size]
        return n / (n.std() + 1e-6)

    n = 0.8 * layer(60, 45) + 0.5 * layer(14, 5) + 0.25 * layer(3, 1.2)
    tile = np.array([112, 70, 28], np.float32) + n[..., None] * np.array([12, 9, 5], np.float32)
    return np.clip(tile, 0, 255).astype(np.uint8)


_OCEAN = _make_ocean()
_real_ac = None   # 순항 구간에서 계기에 표시할 실제 기체 상태 (화면은 축소한 고도로 그린다)
_guide = []       # 순항 구간의 길 안내선 (화면 좌표 점들)
_terrain = None   # 항로 비행의 순항 구간이면 발아래 육지의 비율(0 = 바다 ~ 1 = 육지), 아니면 None (공항 주변)


def _draw_enroute_ground(img, cam, y0, shift):
    """순항 구간의 지면: 높은 고도에서 내려다본 바다 또는 육지 (도시·공항 없음). 해안에서는 둘을 섞는다."""
    hom = shift @ cam.ground_homography(GRASS_MPP, 0, -GRASS_MPP, 0)
    warp = lambda tile: cv2.warpPerspective(tile, hom, (W, H - y0), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_WRAP)
    if _terrain <= 0.01:
        img[y0:] = warp(_OCEAN)
    elif _terrain >= 0.99:
        img[y0:] = warp(_GRASS)
    else:
        img[y0:] = cv2.addWeighted(warp(_GRASS), _terrain, warp(_OCEAN), 1.0 - _terrain, 0)


GUIDE_HALF_WIDTH = 45.0   # 길 안내 차선의 반폭 (화면 기준 m)


def draw_guide_lane(img, cam):
    """순항 구간의 길 안내: 앞으로 갈 항로를 따라 두 줄로 뻗은 차선과 일정 간격의 가로줄.

    한 줄로 그리면 조종사 시점에서 가운데 창틀에 가려지므로 양옆으로 벌려 그린다.
    """
    if len(_guide) < 2:
        return
    pts = np.array(_guide, float)
    d = np.diff(pts[:, :2], axis=0)
    d = np.vstack([d, d[-1]])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    side = np.column_stack([d[:, 1], -d[:, 0], np.zeros(len(d))]) * GUIDE_HALF_WIDTH
    left, right = pts - side, pts + side
    for rail in (left, right):
        for a, b in zip(rail, rail[1:]):
            line_world(img, cam, a, b, (60, 0, 60), 5)
            line_world(img, cam, a, b, MAGENTA, 2)
    for a, b in zip(left[1::2], right[1::2]):   # 1 km마다 가로줄
        line_world(img, cam, a, b, MAGENTA, 1)


def _make_runway():
    """활주로와 표지만 있는 그림 (BGRA). 위성 사진 위에 겹쳐, 낮은 고도에서도 활주로가 또렷하게 한다."""
    w, h = int(80 * AP_PPM), int((RUNWAY_Y1 - RUNWAY_Y0) * AP_PPM)
    key = np.array([1, 255, 2], np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = key
    to_s = lambda p: (int(round((p[0] + 40.0) * AP_PPM)), int(round((RUNWAY_Y1 - p[1]) * AP_PPM)))
    ck.draw_airport(img, to_s, AP_PPM, runway_only=True)
    alpha = (np.abs(img.astype(np.int16) - key).sum(2) > 0).astype(np.uint8) * 255
    return np.dstack([img, alpha])


def _make_detail(seed=12, size=512):
    """가까운 지면에 곱하는 잔무늬 (평균 128 = 변화 없음). 이어 붙여도 경계가 보이지 않는다."""
    rng = np.random.default_rng(seed)

    def layer(sigma):
        n = rng.standard_normal((size, size)).astype(np.float32)
        n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), sigma)[size:2 * size, size:2 * size]
        return n / (n.std() + 1e-6)

    n = 128.0 + 9.0 * layer(1.0) + 8.0 * layer(4.0) + 7.0 * layer(14.0)
    return cv2.merge([np.clip(n, 60, 200).astype(np.uint8)] * 3)


DETAIL_MPP = 0.6        # 잔무늬 1픽셀 = 0.6 m
DETAIL_RANGE = 900.0    # 이 거리 안쪽의 지면에만 잔무늬를 준다 (m)
_DETAIL = _make_detail()
RUNWAY_Y0, RUNWAY_Y1 = -140.0, 1140.0
_RUNWAY = _make_runway()
SAT_FOG = 3.5        # 위성 사진 지면에서는 시야를 이만큼 넓힌다 (멀리 실제 지형이 보이게)
_sat_buildings = None   # 위성 사진 지면의 실제 건물 (공항 주변에서만, 자료를 받은 뒤)
_sat_layers = []     # 이번 프레임에 깔 위성 사진 층들. 비어 있으면 모형 풍경(도시·공항)을 그린다


def _overlay(img, y0, texture, hom):
    """BGRA 질감을 원근 변환해 지면 위에 겹친다 (투명한 곳은 그대로 둔다)."""
    layer = cv2.warpPerspective(texture, hom, (W, H - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    cv2.copyTo(layer[..., :3], cv2.compare(layer[..., 3], 127, cv2.CMP_GT), img[y0:])


def _draw_satellite_ground(img, cam, y0, shift):
    """실제 위성 사진을 지면에 깐다: 넓은 사진부터 자세한 사진 순서로 겹친다."""
    plane = shift @ cam.plane()
    for texture, to_world in _sat_layers:
        _overlay(img, y0, texture, plane @ to_world)
    # 위성 사진은 해상도가 약 10 m라 지면 가까이에서는 뭉개져 보인다: 가까운 곳에는 잔무늬를 곱해 질감을 준다
    near = int(max(y0, math.floor(cam.horizon_y + cam.eye[2] * FOCAL / DETAIL_RANGE)))
    if near < H and _terrain is None:
        move = np.array([[1, 0, 0], [0, 1, -near], [0, 0, 1.0]])
        detail = cv2.warpPerspective(_DETAIL, move @ cam.ground_homography(DETAIL_MPP, 0, -DETAIL_MPP, 0),
                                     (W, H - near), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        img[near:] = cv2.multiply(img[near:], detail, scale=1 / 128.0)
    if _terrain is None:   # 공항 주변: 시뮬레이션의 활주로를 실제 활주로 위에 또렷하게 겹친다
        _overlay(img, y0, _RUNWAY, shift @ cam.ground_homography(1 / AP_PPM, -40.0, -1 / AP_PPM, RUNWAY_Y1))


def draw_ground(cam, night=False):
    y0 = _first_ground_row(cam)
    img = np.empty((H, W, 3), np.uint8)
    img[:] = _sky_colors(night)[0].astype(np.uint8)   # 지면이 없는 곳은 일단 지평선 색 (하늘은 나중에 칠한다)
    if y0 >= H:   # 위를 올려다봐서 지면이 화면에 없다
        return img
    shift = np.array([[1, 0, 0], [0, 1, -y0], [0, 0, 1.0]])   # 지평선 아래 행들만 그린다
    if _sat_layers:
        _draw_satellite_ground(img, cam, y0, shift)
        if _wx["snow"]:
            img = cv2.addWeighted(img, 0.5, np.full_like(img, 236), 0.5, 0)
        if _wx["dim"] < 1.0:
            img = cv2.convertScaleAbs(img, alpha=_wx["dim"])
        return ck.darken(img) if night else img
    if _terrain is not None:
        _draw_enroute_ground(img, cam, y0, shift)
        if _wx["dim"] < 1.0:
            img = cv2.convertScaleAbs(img, alpha=_wx["dim"])
        return ck.darken(img) if night else img
    cv2.warpPerspective(_GRASS, shift @ cam.ground_homography(GRASS_MPP, 0, -GRASS_MPP, 0),
                        (W, H - y0), dst=img[y0:], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    _warp_city(img, cam, scenery.CITY["day"], True)
    ap = np.zeros((H, W, 4), np.uint8)
    cv2.warpPerspective(_AIRPORT, shift @ cam.ground_homography(1 / AP_PPM, AP_X0, -1 / AP_PPM, AP_Y1),
                        (W, H - y0), dst=ap[y0:], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0, 0))
    cv2.copyTo(ap[..., :3], cv2.compare(ap[..., 3], 127, cv2.CMP_GT), img)
    if _wx["snow"]:        # 눈: 지면이 하얗게 덮인다
        img = cv2.addWeighted(img, 0.5, np.full_like(img, 236), 0.5, 0)
    if _wx["dim"] < 1.0:   # 흐린 날은 지면이 어둡다
        img = cv2.convertScaleAbs(img, alpha=_wx["dim"])
    if not night:
        return img
    img = ck.darken(img)   # 밤 = 어둡게 한 낮 그림 + 불빛
    lights = np.zeros_like(img)
    _warp_city(lights, cam, scenery.CITY["emission"], False)
    return cv2.add(img, lights)


CITY_MAX_DISTANCE = 4200.0   # 이보다 먼 건물은 입체로 세우지 않는다 (지면 그림의 지붕만 보인다)


def city_faces(cam, night, box=None, fog_scale=1.0):
    """화면에 보이는 건물 면들: (화면 좌표 (n,4,2), 색 목록, 깊이). 먼 것부터 정렬.

    box를 주지 않으면 모형 도시의 건물, 주면 그 건물들(위성 사진 지면의 실제 건물)을 그린다.
    """
    if box is None:
        box = scenery.CITY["boxes"]
    pc = cam.to_cam(box["centers"])
    z, tall = pc[:, 2], box["heights"]
    far = min(fog_scale, 2.4)   # 시야가 넓은 위성 사진 지면에서는 더 멀리까지 세운다
    seen = (z > 8.0) & (z < CITY_MAX_DISTANCE * far) & (np.abs(pc[:, 0]) < z * 0.62 + 120.0)
    seen &= (z < 1500.0 * far) | (tall > 20.0)      # 멀수록 높은 건물만
    seen &= (z < 2700.0 * far) | (tall > 45.0)
    pick = np.flatnonzero(seen[box["owner"]])
    if not len(pick):
        return np.zeros((0, 4, 2), np.int32), [], np.zeros(0), np.zeros((0, 2))
    verts = box["verts"][pick]
    facing = np.einsum("ij,ij->i", box["normals"][pick], cam.eye - verts.mean(1)) > 0
    pick, verts = pick[facing], verts[facing]
    cam_pts = (verts.reshape(-1, 3) - cam.eye) @ cam.rot.T
    cam_pts = cam_pts.reshape(-1, 4, 3)
    front = (cam_pts[:, :, 2] >= NEAR).all(1)
    pick, cam_pts = pick[front], cam_pts[front]
    depth = cam_pts[:, :, 2].mean(1)
    order = np.argsort(-depth)
    pick, cam_pts, depth = pick[order], cam_pts[order], depth[order]
    uv = np.stack([CX + FOCAL * cam_pts[..., 0] / cam_pts[..., 2],
                   CY - FOCAL * cam_pts[..., 1] / cam_pts[..., 2]], -1)
    keep = np.exp(-depth / (_wx["fog"] * fog_scale))[:, None]   # 멀수록 지평선 색으로 흐려진다
    haze = _sky_colors(night)[0]
    lit = box["night"][pick] if night else box["day"][pick] * _wx["dim"]
    colors = lit * keep + haze * (1 - keep)
    # 벽 면의 (건물 높이, 벽 너비): 가까운 벽에 창문을 그릴 때 쓴다. 건물마다 면 5개 중 첫 번째가 지붕이다
    wall = (pick % 5) != 0
    width = np.linalg.norm(box["verts"][pick][:, 1, :2] - box["verts"][pick][:, 0, :2], axis=1)
    walls = np.column_stack([np.where(wall, box["heights"][box["owner"][pick]], 0.0), width])
    return to_px(uv), colors.astype(int).tolist(), depth, walls


WINDOW_RANGE = 900.0    # 이보다 가까운 건물 벽에는 층마다 창문 띠를 그린다 (m)
FLOOR_HEIGHT = 3.4      # 한 층 높이 (m)


def _draw_windows(img, quad, color, floors, columns):
    """벽 한 면에 창문을 그린다: 층마다 어두운 유리 띠, 그리고 세로 기둥. quad = 아래 두 점, 위 두 점 순서."""
    p0, p1, p2, p3 = quad.astype(np.float32)
    glass = tuple(int(c * 0.42 + k) for c, k in zip(color, (38, 24, 12)))   # 벽 색보다 어둡고 푸르스름하게
    lerp = lambda a, b, t: a + (b - a) * t
    for f in range(floors):
        t1, t2 = (f + 0.32) / floors, (f + 0.8) / floors
        band = np.array([lerp(p0, p3, t1), lerp(p1, p2, t1), lerp(p1, p2, t2), lerp(p0, p3, t2)])
        cv2.fillConvexPoly(img, np.round(band).astype(np.int32), glass, AA)
    for c in range(1, columns):   # 창 사이 기둥은 벽 색 그대로
        s = c / columns
        a, b = lerp(p0, p1, s), lerp(p3, p2, s)
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 1, AA)


def draw_city(img, faces, nearer_than=None, farther_than=None):
    """city_faces()의 면들을 칠한다. 기체보다 먼 것과 가까운 것을 나눠 그리려고 깊이 범위를 받는다."""
    uv, colors, depth = faces[:3]
    walls = faces[3] if len(faces) > 3 else None   # 벽이면 (건물 높이 m, 벽 너비 m), 지붕이면 높이 0
    for i in range(len(depth)):
        if nearer_than is not None and depth[i] >= nearer_than:
            continue
        if farther_than is not None and depth[i] < farther_than:
            continue
        cv2.fillConvexPoly(img, uv[i], colors[i], AA)
        if walls is not None and depth[i] < WINDOW_RANGE and walls[i][0] > 5.0:
            floors = int(min(walls[i][0] / FLOOR_HEIGHT, 30))
            if floors * 3 < abs(uv[i][3][1] - uv[i][0][1]):   # 한 층이 3픽셀은 되어야 그린다
                _draw_windows(img, uv[i], colors[i], floors, int(np.clip(walls[i][1] / 6.0, 1, 24)))


NIGHT_HAZE = np.array([46, 30, 20], np.float32)     # 밤의 지평선 색 (BGR)
NIGHT_ZENITH = np.array([20, 9, 4], np.float32)     # 밤하늘
NIGHT_AMBIENT = 0.38                                # 밤에 기체·건물에 닿는 빛의 양


def _make_stars(n=450, seed=11):
    """별: 하늘의 고정된 방향(단위 벡터)과 밝기."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.03
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d, rng.uniform(90, 255, n) ** 1.0


_STARS, _STAR_LEVEL = _make_stars()
_MOON = np.array([math.sin(math.radians(35)) * math.cos(math.radians(32)),
                  math.cos(math.radians(35)) * math.cos(math.radians(32)),
                  math.sin(math.radians(32))])   # 북동쪽 하늘, 고도 32도


_AP_LIGHT_POS = np.array([(x, y, 0.6) for x, y, _c, _s in ck.AIRPORT_LIGHTS], float)


_SUN = np.array([math.sin(math.radians(-25)) * math.cos(math.radians(38)),
                 math.cos(math.radians(-25)) * math.cos(math.radians(38)),
                 math.sin(math.radians(38))])   # 북서쪽 하늘, 고도 38도


CLOUD_RANGE = 7000.0   # 구름이 놓이는 범위 (원점 기준 ±m). 바람에 밀려 나가면 반대편에서 다시 들어온다


def _make_clouds(seed=3, clusters=90):
    """구름 덩어리들: 월드 좌표 (x, y, 밑면 위 높이), 반지름 m, 진하기, 속한 구름 번호.

    구름 하나는 여러 덩어리가 뭉친 것이다. 구름 양(cover)에 따라 앞쪽 번호의 구름만 그린다.
    """
    rng = np.random.default_rng(seed)
    pos, radius, alpha, owner = [], [], [], []
    for k in range(clusters):
        cx, cy = rng.uniform(-CLOUD_RANGE, CLOUD_RANGE, 2)
        cz = rng.uniform(0, 260)
        size = rng.uniform(0.7, 1.5)
        for _ in range(int(rng.integers(5, 10))):
            pos.append((cx + rng.normal(0, 230) * size, cy + rng.normal(0, 150) * size,
                        cz + abs(rng.normal(0, 40))))
            radius.append(rng.uniform(140, 300) * size)
            alpha.append(rng.uniform(0.55, 0.95))
            owner.append(k)
    return (np.array(pos), np.array(radius), np.array(alpha), np.array(owner), clusters)


_CLOUDS = _make_clouds()
_CLOUD_SCALE = 2   # 구름은 절반 크기로 그려서 흐리게 한 뒤 키운다 (빠르고 부드럽다)


def draw_clouds(img, cam, night, clock, wind=(3.0, -2.5)):
    """구름: 실제 높이에 떠 있어서 날아가면 지나가고, 바람을 따라 흘러간다."""
    if _wx["cover"] <= 0.0:
        return
    pos, radius, alpha, owner, clusters = _CLOUDS
    pick = owner < _wx["cover"] * clusters
    p = pos[pick].copy()
    p[:, 0] = (p[:, 0] + wind[0] * clock * 4 + CLOUD_RANGE) % (2 * CLOUD_RANGE) - CLOUD_RANGE
    p[:, 1] = (p[:, 1] + wind[1] * clock * 4 + CLOUD_RANGE) % (2 * CLOUD_RANGE) - CLOUD_RANGE
    p[:, 2] += _wx["base"]
    pc = cam.to_cam(p)
    z = pc[:, 2]
    r_px = FOCAL * radius[pick] / np.maximum(z, 1.0)
    u = CX + FOCAL * pc[:, 0] / np.maximum(z, 1.0)
    v = CY - FOCAL * pc[:, 1] / np.maximum(z, 1.0)
    ok = (z > 150.0) & (u > -r_px * 1.6) & (u < W + r_px * 1.6) & (v > -r_px) & (v < H + r_px)
    if not ok.any():
        return
    k = _CLOUD_SCALE
    mask = np.zeros((H // k, W // k), np.uint8)
    fade = np.exp(-z / (_wx["fog"] * 3.0))   # 먼 구름은 옅다
    for i in np.flatnonzero(ok)[np.argsort(-z[ok])]:
        cv2.ellipse(mask, (int(u[i] / k), int(v[i] / k)),
                    (max(int(r_px[i] * 1.5 / k), 1), max(int(r_px[i] * 0.55 / k), 1)), 0, 0, 360,
                    int(255 * alpha[pick][i] * fade[i]), -1, AA)
    mask = cv2.resize(cv2.GaussianBlur(mask, (0, 0), 5), (W, H), interpolation=cv2.INTER_LINEAR)
    cover = mask.astype(np.float32) * (0.92 / 255.0)
    color = np.array(_wx["cloud"], np.float32) * (0.2 if night else 1.0)
    layer = np.empty_like(img)
    layer[:] = color
    img[:] = cv2.blendLinear(img, layer, 1.0 - cover, cover)


def draw_day_sky(img, cam):
    """낮 하늘의 해."""
    hy = cam.horizon_y
    s = cam.rot @ _SUN
    if s[2] > 0.05:
        su, sv = CX + FOCAL * s[0] / s[2], CY - FOCAL * s[1] / s[2]
        if -80 < su < W + 80 and -80 < sv < hy:
            for radius, alpha in ((70, 0.18), (44, 0.3), (28, 0.5)):
                ck.blend_circle(img, (su, sv), radius, (215, 245, 255), alpha)
            cv2.circle(img, (int(su), int(sv)), 17, (235, 252, 255), -1, AA)


def _sky_colors(night):
    """지금 날씨의 (지평선 색, 높은 하늘 색)."""
    if not night:
        return (HAZE, ZENITH) if _wx["sky"] is None else tuple(np.array(c, np.float32) for c in _wx["sky"])
    if _wx["sky"] is None:
        return NIGHT_HAZE, NIGHT_ZENITH
    gray = np.array([34, 30, 28], np.float32) * (0.6 + 0.4 * _wx["dim"])   # 흐린 밤하늘
    return gray, gray * 0.6


# 비·눈 입자: 화면 위의 고정된 씨앗 위치에서 시간에 따라 흘러내린다 (프레임마다 이어지게)
_PARTICLES = np.random.default_rng(8).random((600, 4))


def draw_precipitation(img, clock):
    """빗줄기와 눈송이 (화면 전체에 겹쳐 그린다)."""
    h, w = img.shape[:2]
    if _wx["rain"]:
        seed = _PARTICLES[:_wx["rain"]]
        speed = 900.0 + 500.0 * seed[:, 2]                      # 픽셀/초
        y = (seed[:, 1] * (h + 60) + speed * clock) % (h + 60) - 30
        x = (seed[:, 0] * (w + 200) - 0.28 * speed * clock) % (w + 200) - 100   # 바람에 비스듬히
        length = 14 + 16 * seed[:, 3]
        layer = np.zeros_like(img)
        for x1, y1, n in zip(x, y, length):
            cv2.line(layer, (int(x1), int(y1)), (int(x1 - 0.28 * n), int(y1 + n)), (190, 185, 180), 1, AA)
        cv2.addWeighted(img, 1.0, layer, 0.5, 0, img)
    if _wx["snow"]:
        seed = _PARTICLES[:_wx["snow"]]
        speed = 60.0 + 90.0 * seed[:, 2]
        y = ((seed[:, 1] * h + speed * clock) % h).astype(np.int32)
        x = ((seed[:, 0] * w + 26.0 * np.sin(clock * 1.3 + seed[:, 3] * 6.28) - 18.0 * clock) % w).astype(np.int32)
        big = seed[:, 3] > 0.6
        img[y, x] = 245
        img[np.minimum(y[big] + 1, h - 1), x[big]] = 245
        img[y[big], np.minimum(x[big] + 1, w - 1)] = 245
        img[np.minimum(y[big] + 1, h - 1), np.minimum(x[big] + 1, w - 1)] = 235


def lightning_strength(clock):
    """뇌우의 번개 밝기 0~1. 몇 초에 한 번, 짧게 두 번 번쩍인다."""
    if not _wx["lightning"]:
        return 0.0, 0
    window = int(clock // 5.0)
    rng = np.random.default_rng(1000 + window)
    start = rng.uniform(0.3, 4.2)
    dt = clock - window * 5.0 - start
    if 0.0 <= dt < 0.12:
        return 1.0, window
    if 0.22 <= dt < 0.32:
        return 0.6, window
    return 0.0, window


def draw_lightning(img, clock, horizon_y=None):
    """번개: 화면이 번쩍이고 하늘에 벼락 줄기가 보인다."""
    strength, window = lightning_strength(clock)
    if strength <= 0.0:
        return
    cv2.convertScaleAbs(img, dst=img, alpha=1.0 + 0.5 * strength, beta=55 * strength)
    if horizon_y is None or horizon_y < 60:
        return
    rng = np.random.default_rng(2000 + window)   # 같은 번개 동안 같은 모양
    x, y = rng.uniform(80, img.shape[1] - 80), 0.0
    pts = [(int(x), 0)]
    while y < horizon_y - 8:
        y += rng.uniform(18, 46)
        x += rng.uniform(-34, 34)
        pts.append((int(x), int(min(y, horizon_y))))
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 235, 220), 3, AA)
    cv2.polylines(img, [np.array(pts, np.int32)], False, (255, 255, 255), 1, AA)


_WEATHER_ICON_COLOR = (235, 235, 235)


def draw_weather_button(img, weather):
    """날씨 전환 버튼: 지금 날씨를 보여주고, 누르면 다음 날씨로 바뀐다."""
    x1, y1, x2, y2 = WEATHER_BUTTON
    ck.shade(img, x1, y1, x2, y2, (90, 80, 70), 0.75)
    cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cx, cy = x1 + 20, (y1 + y2) // 2
    wx = WEATHERS[weather]
    if weather == "clear":
        cv2.circle(img, (cx, cy), 7, (120, 235, 255), -1, AA)
    else:   # 구름 모양 + 비/눈/번개/안개 표시
        for dx, dy, r in ((-6, 0, 5), (0, -3, 7), (7, 0, 5)):
            cv2.circle(img, (cx + dx, cy - 3 + dy), r, _WEATHER_ICON_COLOR, -1, AA)
        if wx["rain"]:
            for dx in (-6, 0, 6):
                cv2.line(img, (cx + dx, cy + 5), (cx + dx - 2, cy + 11), (255, 200, 120), 1, AA)
        if wx["snow"]:
            for dx in (-6, 0, 6):
                cv2.circle(img, (cx + dx, cy + 9), 1, (255, 255, 255), -1, AA)
        if wx["lightning"]:
            cv2.polylines(img, [np.array([(cx + 2, cy + 2), (cx - 2, cy + 8), (cx + 2, cy + 8), (cx - 1, cy + 13)],
                                         np.int32)], False, (90, 230, 255), 2, AA)
        if weather == "fog":
            for dy in (5, 9):
                cv2.line(img, (cx - 10, cy + dy), (cx + 10, cy + dy), _WEATHER_ICON_COLOR, 1, AA)
    ck.text(img, wx["label"], (x1 + 40, cy + 6), 0.5, WHITE, 2)


def draw_cam_button(img, cam):
    """시점 전환 버튼: 지금 시점(조종사 / 외부)을 보여주고, 누르면 바뀐다."""
    x1, y1, x2, y2 = CAM_BUTTON
    pilot = cam == "cockpit"
    ck.shade(img, x1, y1, x2, y2, (40, 90, 40) if pilot else (110, 70, 50), 0.75)
    cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cx, cy = x1 + 18, (y1 + y2) // 2
    if pilot:   # 눈 모양
        cv2.ellipse(img, (cx, cy), (10, 6), 0, 0, 360, WHITE, 1, AA)
        cv2.circle(img, (cx, cy), 3, WHITE, -1, AA)
    else:       # 위에서 본 비행기 모양
        cv2.line(img, (cx, cy - 9), (cx, cy + 8), WHITE, 2, AA)
        cv2.line(img, (cx - 10, cy), (cx + 10, cy), WHITE, 2, AA)
        cv2.line(img, (cx - 4, cy + 8), (cx + 4, cy + 8), WHITE, 2, AA)
    ck.text(img, "PILOT" if pilot else "OUTSIDE", (x1 + 36, cy + 6), 0.5, WHITE, 2)


def draw_time_button(img, night):
    """낮/밤 전환 버튼: 지금 상태를 보여주고, 누르면 바뀐다."""
    x1, y1, x2, y2 = TIME_BUTTON
    ck.shade(img, x1, y1, x2, y2, (60, 30, 10) if night else (150, 110, 40), 0.75)
    cv2.rectangle(img, (x1, y1), (x2, y2), (230, 230, 230), 1)
    cy = (y1 + y2) // 2
    if night:   # 초승달
        cv2.circle(img, (x1 + 18, cy), 9, (225, 235, 240), -1, AA)
        cv2.circle(img, (x1 + 23, cy - 3), 8, (60, 34, 16), -1, AA)
    else:       # 해
        cv2.circle(img, (x1 + 18, cy), 7, (120, 235, 255), -1, AA)
        for k in range(8):
            a = math.pi * k / 4
            cv2.line(img, (int(x1 + 18 + 9 * math.cos(a)), int(cy + 9 * math.sin(a))),
                     (int(x1 + 18 + 12 * math.cos(a)), int(cy + 12 * math.sin(a))), (120, 235, 255), 1, AA)
    ck.text(img, "NIGHT" if night else "DAY", (x1 + 38, cy + 6), 0.55, WHITE, 2)


def draw_sky_objects(img, cam):
    """밤하늘의 별과 달. 아주 멀리 있으므로 카메라가 돌 때만 움직인다."""
    hy = cam.horizon_y
    pc = _STARS @ cam.rot.T
    ok = pc[:, 2] > 0.05
    u = (CX + FOCAL * pc[ok, 0] / pc[ok, 2]).astype(np.int32)
    v = (CY - FOCAL * pc[ok, 1] / pc[ok, 2]).astype(np.int32)
    level = _STAR_LEVEL[ok]
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < min(H, hy - 4))
    img[v[inside], u[inside]] = level[inside, None].astype(np.uint8)

    m = cam.rot @ _MOON
    if m[2] > 0.05:
        mu, mv = CX + FOCAL * m[0] / m[2], CY - FOCAL * m[1] / m[2]
        if -40 < mu < W + 40 and -40 < mv < hy:
            ck.blend_circle(img, (mu, mv), 42, (150, 140, 120), 0.12)
            ck.blend_circle(img, (mu, mv), 24, (190, 180, 160), 0.2)
            cv2.circle(img, (int(mu), int(mv)), 13, (228, 236, 240), -1, AA)
            cv2.circle(img, (int(mu) - 4, int(mv) - 3), 3, (205, 214, 220), -1, AA)


def draw_point_lights(img, cam, pts, colors, sizes, glow=True):
    """지상의 점 광원들. 실제 등화처럼 작고 또렷한 점으로 그린다: 멀면 한 픽셀, 가까우면 작은 원과 옅은 번짐."""
    pc = cam.to_cam(pts)
    ok = pc[:, 2] >= NEAR
    z = np.where(ok, pc[:, 2], 1.0)
    u = CX + FOCAL * pc[:, 0] / z
    v = CY - FOCAL * pc[:, 1] / z
    ok &= (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    if not ok.any():
        return
    idx = np.flatnonzero(ok)
    z, ui, vi = z[idx], u[idx].astype(int), v[idx].astype(int)
    fade = np.exp(-z / (_wx["fog"] * 2.0))[:, None]   # 시야가 나쁘면 불빛도 가까이서만 보인다
    col = (np.asarray(colors, np.float32)[idx] * fade).astype(np.uint8)
    radius = FOCAL * np.asarray(sizes, np.float32)[idx] / z
    far = radius < 1.0
    # 먼 불빛: 한 픽셀 (아주 멀면 더 어둡게). 같은 픽셀에 여럿이 겹치면 밝은 쪽을 남긴다
    dim = np.clip(radius[far] * 2.5, 0.35, 1.0)[:, None]
    np.maximum.at(img, (vi[far], ui[far]), (col[far] * dim).astype(np.uint8))
    for k in np.flatnonzero(~far):   # 가까운 불빛: 작은 원 + 옅은 번짐
        r = float(min(radius[k], 3.0))
        color = tuple(int(c) for c in col[k])
        if glow and r >= 1.6:
            ck.blend_circle(img, (ui[k], vi[k]), r * 2.4, color, 0.16)
        cv2.circle(img, (int(ui[k]), int(vi[k])), max(1, int(round(r))), color, -1, AA)
        if r >= 2.0:   # 가운데는 하얗게 탄 것처럼
            cv2.circle(img, (int(ui[k]), int(vi[k])), 1, (255, 255, 255), -1, AA)


_RUNWAY_LIGHTS = np.array([abs(x) <= 25.0 for x, _y, _c, _s in ck.AIRPORT_LIGHTS])


LAMP_RANGE = 2500.0   # 이 거리 안의 가로등은 등 하나하나를 그린다 (m)
_AP_COLORS = np.array([c for _x, _y, c, _s in ck.AIRPORT_LIGHTS], np.float32)
_AP_SIZES = np.array([s for _x, _y, _c, s in ck.AIRPORT_LIGHTS], np.float32)
_STROBE_POS = np.array([(x, y, 1.0) for x, y in ck.APPROACH_STROBES], float)
_PAPI_POS = np.array([(x, ck.PAPI_Y, 0.8) for x in ck.PAPI_X], float)


def draw_runway_lights(img, cam, ac, clock, runway_only):
    """활주로·진입로·유도로의 등화. runway_only면 활주로 주변 것만 (위성 사진 지면: 유도로는 실제와 다르다)."""
    pick = _RUNWAY_LIGHTS if runway_only else slice(None)
    draw_point_lights(img, cam, _AP_LIGHT_POS[pick], _AP_COLORS[pick], _AP_SIZES[pick])
    # 섬광등: 0.5초에 한 바퀴, 먼 쪽부터 시단 쪽으로 하나씩 번쩍인다
    n = len(_STROBE_POS)
    k = int((clock % 0.5) / 0.5 * (n + 3))
    if k < n:
        draw_point_lights(img, cam, _STROBE_POS[k:k + 1], [(255, 255, 255)], [1.6])
    # 진입각 지시등: 기체에서 등을 내려다보는 각도에 따라 흰색 또는 빨간색
    dist = math.hypot(ac.x - _PAPI_POS[0, 0], ac.y - ck.PAPI_Y)
    angle = math.degrees(math.atan2(max(ac.alt, 0.0), max(dist, 1.0)))
    colors = [(245, 245, 255) if angle > limit else (50, 50, 255) for limit in ck.PAPI_ANGLES]
    draw_point_lights(img, cam, _PAPI_POS, colors, [0.9] * 4)


def draw_night_lights(img, cam, clock=0.0):
    """모형 도시의 불빛: 길 양옆의 가로등과, 높은 빌딩 꼭대기의 깜빡이는 빨간 등."""
    lamps = scenery.CITY["lamps"]
    if len(lamps):   # 가까운 가로등만 (멀리 있는 것은 지면 그림의 빛으로 충분하다)
        near = np.hypot(lamps[:, 0] - cam.eye[0], lamps[:, 1] - cam.eye[1]) < LAMP_RANGE
        if near.any():
            draw_point_lights(img, cam, lamps[near], scenery.CITY["lamp_colors"][near],
                              np.full(int(near.sum()), 0.4, np.float32))
    beacons = scenery.CITY["boxes"]["beacons"]
    if len(beacons) and clock % 1.6 < 0.8:
        draw_point_lights(img, cam, beacons, [(70, 70, 255)] * len(beacons), [1.6] * len(beacons),
                          glow=False)


def draw_landing_light(img, cam, ac):
    """낮은 고도에서 기체 앞 지면을 비추는 착륙등 불빛."""
    if ac.alt > 100.0:
        return
    h = math.radians(ac.heading)
    fwd, right = np.array([math.sin(h), math.cos(h)]), np.array([math.cos(h), -math.sin(h)])
    center = np.array([ac.x, ac.y]) + fwd * (45.0 + 2.0 * ac.alt)
    along, across = 30.0 + 0.6 * ac.alt, 12.0 + 0.3 * ac.alt
    ang = np.linspace(0, 2 * math.pi, 25)[:-1]
    pts = center + np.outer(np.cos(ang) * along, fwd) + np.outer(np.sin(ang) * across, right)
    pc = clip_near(cam.to_cam(np.column_stack([pts, np.zeros(len(pts))])))
    if pc is None:
        return
    layer = img.copy()
    cv2.fillPoly(layer, [to_px(cam.project(pc))], (190, 225, 250), AA)
    alpha = 0.38 * (1.0 - ac.alt / 100.0) + 0.05
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def _make_ridges(seed, octaves, n=2048):
    """지평선을 한 바퀴 두르는 먼 산등성이의 높이(0~1). 방위 0~360도에 n개, 끝이 이어진다."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    h = np.zeros(n)
    for k, amp in octaves:
        h += amp * np.sin(k * a + rng.uniform(0, 2 * math.pi))
    h = np.abs(h) ** 0.8 * np.sign(h)            # 봉우리는 뾰족하게
    h = (h - h.min()) / (h.max() - h.min())
    return h.astype(np.float32)


# (높이 배열, 가장 높은 봉우리의 올려본각(도), 가장 낮은 곳의 올려본각, 색이 지평선 색에서 벗어나는 정도)
_RIDGES = [
    (_make_ridges(5, [(3, 1.0), (7, 0.6), (13, 0.4), (29, 0.22), (61, 0.12), (127, 0.06)]), 3.4, 0.5, 0.30),
    (_make_ridges(9, [(5, 1.0), (11, 0.7), (23, 0.4), (47, 0.25), (97, 0.12), (211, 0.06)]), 1.7, 0.15, 0.55),
]
MOUNTAIN = np.array([112, 100, 92], np.float32)   # 가까이서 본 산의 색 (푸른 회색, BGR)


def draw_mountains(img, cam, haze_color, night):
    """지평선 위의 먼 산. 아주 멀어서 카메라가 돌 때만 움직이고, 멀수록 지평선 색에 가깝다."""
    if _sat_layers:   # 위성 사진 지면에서는 지어낸 산을 그리지 않는다
        return
    strength = float(np.clip((_wx["fog"] - 1200.0) / 3000.0, 0.0, 1.0))   # 시야가 나쁘면 안 보인다
    if _terrain is not None:
        strength *= _terrain   # 순항 구간: 바다 위에서는 산이 없다
    hy = cam.horizon_y
    if strength <= 0.02 or hy <= 0 or hy > H + 150:
        return
    u = np.arange(-8, W + 9, 4, dtype=np.float64)
    rays = np.stack([(u - CX) / FOCAL, np.full_like(u, (CY - hy) / FOCAL), np.ones_like(u)], 1) @ cam.rot
    az = np.arctan2(rays[:, 0], rays[:, 1]) % (2 * math.pi)   # 각 화면 열이 바라보는 방위
    base = MOUNTAIN * (0.25 if night else 1.0)
    if _wx["snow"]:
        base = base * 0.4 + 150.0
    for heights, top_deg, low_deg, tint in _RIDGES:
        n = len(heights)
        elev = np.radians(low_deg + (top_deg - low_deg) * heights[(az / (2 * math.pi) * n).astype(int) % n])
        pts = np.stack([u, hy - FOCAL * np.tan(elev)], 1)
        poly = np.vstack([pts, [(W + 8, hy + 1), (-8, hy + 1)]])
        color = haze_color * (1 - tint * strength) + base * tint * strength
        cv2.fillPoly(img, [to_px(poly)], tuple(float(c) for c in color), AA)


# 색 보정: 사진처럼 중간 밝기의 대비를 조금 높이고(S자 곡선) 화면 가장자리는 조금 어둡게 한다
_x = np.arange(256, dtype=np.float32) / 255.0
_GRADE = np.clip((_x + 0.035 * np.sin((_x - 0.5) * 2 * math.pi)) * 255.0, 0, 255).astype(np.uint8)
_vy, _vx = np.mgrid[0:H, 0:W].astype(np.float32)
_VIGNETTE = cv2.merge([1.0 - 0.22 * np.clip(((_vx - CX) / CX) ** 2 * 0.5 + ((_vy - CY) / CY) ** 2 * 0.5, 0, 1)] * 3)


def grade(img):
    return cv2.multiply(cv2.LUT(img, _GRADE), _VIGNETTE, dtype=cv2.CV_8U)


def draw_fog_and_sky(img, cam, night=False):
    """먼 지면은 지평선 색으로 흐리게, 지평선 위는 하늘."""
    haze_color, zenith = _sky_colors(night)
    hy = cam.horizon_y
    rows = np.arange(H, dtype=np.float32)
    top = int(np.clip(math.ceil(hy), 0, H))
    if top < H:   # 지면: 화면 행마다 지면까지의 거리로 안개 농도를 정한다
        dist = cam.eye[2] * FOCAL / np.maximum(rows[top:] - hy, 0.5)
        fog = _wx["fog"] * (SAT_FOG if _sat_layers else 1.0)
        keep = np.repeat(np.exp(-dist / fog).astype(np.float32)[:, None], W, axis=1)
        haze = np.empty((H - top, W, 3), np.uint8)
        haze[:] = haze_color
        img[top:] = cv2.blendLinear(img[top:], haze, keep, 1.0 - keep)
    if top > 0:   # 하늘: 지평선에서 위로 갈수록 짙은 파랑
        k = np.clip((hy - rows[:top]) / (FOCAL * 0.9), 0.0, 1.0)[:, None] ** 0.7
        img[:top] = (haze_color * (1 - k) + zenith * k).astype(np.uint8)[:, None, :]
        draw_mountains(img, cam, haze_color, night)
        if _wx["sun"]:   # 맑거나 구름이 조금 낀 날에만 해·달·별이 보인다
            if night:
                draw_sky_objects(img, cam)
            else:
                draw_day_sky(img, cam)


def _shaded(color, normal, ambient=1.0):
    """면의 법선과 빛 방향으로 밝기를 정한다. ambient는 전체 밝기 (밤에는 낮춘다)."""
    k = (0.55 + 0.45 * max(float(normal @ SUN), 0.0)) * ambient
    return tuple(int(c * k) for c in color)


def draw_buildings(img, cam, night=False, clock=0.0):
    faces = []
    for x1, y1, x2, y2, h, wall, roof in BUILDINGS:
        c = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        faces.append(([(x, y, h) for x, y in c], roof, np.array([0, 0, 1.0])))
        for i in range(4):
            (ax, ay), (bx, by) = c[i], c[(i + 1) % 4]
            n = np.array([by - ay, ax - bx, 0.0])
            n /= np.linalg.norm(n)
            if n @ (cam.eye - np.array([ax, ay, 0.0])) > 0:   # 카메라를 향한 벽만
                faces.append(([(ax, ay, 0), (bx, by, 0), (bx, by, h), (ax, ay, h)], wall, n))
    faces.sort(key=lambda f: -np.linalg.norm(np.mean(f[0], 0) - cam.eye))
    for pts, color, n in faces:
        fill_world(img, cam, pts, _shaded(color, n, NIGHT_AMBIENT * 0.8 if night else _wx["dim"]))
    if night and int(clock * 1.2) % 2 == 0:   # 관제탑 꼭대기의 깜빡이는 빨간 등
        x1, y1, x2, y2, h = BUILDINGS[2][:5]
        draw_point_lights(img, cam, np.array([[(x1 + x2) / 2, (y1 + y2) / 2, h + 1.5]]),
                          [(60, 60, 255)], [1.2])


# ---------------------------------------------------------------- 목표 경로

PATH_HALF_WIDTH = 2.5   # 지면 위 경로 띠의 반폭 (m)
GUIDE_REF = (90, 230, 255)   # 조종석 조종 안내의 기준 표시 색 (노랑)
COCKPIT_PITCH = 0.5   # 조종석 시선이 기수 피치를 따라가는 비율
AIRPATH = (255, 150, 255)   # 공중 목표 고도 선 (지면 경로보다 밝은 분홍)


def draw_targets(img, cam, sc):
    for shape in sc.targets():
        if shape[0] == "line":
            (x1, y1), (x2, y2) = shape[1], shape[2]
            d = np.array([x2 - x1, y2 - y1], float)
            d /= np.linalg.norm(d)
            px, py = -d[1] * PATH_HALF_WIDTH, d[0] * PATH_HALF_WIDTH
            fill_world(img, cam, [(x1 + px, y1 + py, 0), (x2 + px, y2 + py, 0),
                                  (x2 - px, y2 - py, 0), (x1 - px, y1 - py, 0)], MAGENTA)
            line_world(img, cam, (x1, y1, 0), (x2, y2, 0), MAGENTA, 1)   # 멀리서도 보이게
        elif shape[0] == "circle":
            (cx, cy), r = shape[1], shape[2]
            ang = np.linspace(0, 2 * math.pi, 97)
            cs, sn = np.cos(ang), np.sin(ang)
            rings = [cam.to_cam(np.column_stack([cx + rr * cs, cy + rr * sn, np.zeros(97)]))
                     for rr in (r - PATH_HALF_WIDTH, r + PATH_HALF_WIDTH)]
            if all((ring[:, 2] >= NEAR).all() for ring in rings):
                # 원 전체가 카메라 앞에 있으면 고리를 한 번에 칠한다 (안쪽 원은 구멍)
                cv2.fillPoly(img, [to_px(cam.project(ring)) for ring in rings], MAGENTA, AA)
                continue
            for i in range(96):
                quad = [(cx + rr * cs[j], cy + rr * sn[j], 0)
                        for rr, j in ((r - PATH_HALF_WIDTH, i), (r + PATH_HALF_WIDTH, i),
                                      (r + PATH_HALF_WIDTH, i + 1), (r - PATH_HALF_WIDTH, i + 1))]
                fill_world(img, cam, quad, MAGENTA, aa=False)
                line_world(img, cam, (cx + r * cs[i], cy + r * sn[i], 0),
                           (cx + r * cs[i + 1], cy + r * sn[i + 1], 0), MAGENTA, 1)
        elif shape[0] == "path":    # 꺾은선 경로 (장주 비행): 구간마다 지면 띠
            pts = np.array(shape[1], float)
            d = np.diff(pts, axis=0)
            d /= np.linalg.norm(d, axis=1, keepdims=True)
            side = np.column_stack([-d[:, 1], d[:, 0]]) * PATH_HALF_WIDTH
            for (x1, y1), (x2, y2), (px, py) in zip(pts[:-1], pts[1:], side):
                fill_world(img, cam, [(x1 + px, y1 + py, 0), (x2 + px, y2 + py, 0),
                                      (x2 - px, y2 - py, 0), (x1 - px, y1 - py, 0)], MAGENTA, aa=False)
                line_world(img, cam, (x1, y1, 0), (x2, y2, 0), MAGENTA, 1)
        elif shape[0] == "airpath":   # 목표 고도를 따라 공중에 그은 선 (복잡한 경로 과제)
            for a, b in zip(shape[1], shape[1][1:]):
                line_world(img, cam, a, b, AIRPATH, 2)
        elif shape[0] in ("point", "gate"):   # 웨이포인트: 지면 마름모 + 수직 기둥 + 별
            x, y = shape[1]
            top = shape[2] if shape[0] == "gate" else 70   # gate는 목표 고도까지 기둥을 세운다
            fill_world(img, cam, [(x, y - 18, 0), (x + 18, y, 0), (x, y + 18, 0), (x - 18, y, 0)],
                       MAGENTA)
            line_world(img, cam, (x, y, 0), (x, y, top), MAGENTA, 2)
            pc = cam.to_cam([(x, y, top)])
            if pc[0, 2] >= NEAR:
                u, v = (int(k) for k in to_px(cam.project(pc))[0])
                if shape[0] == "gate" and -100 < u < W + 100 and -100 < v < H + 100:
                    ck.text(img, f"{top:.0f} m", (u + 16, v + 5), 0.5, WHITE, 2)
                star = np.array([(u, v - 12), (u + 4, v - 4), (u + 12, v), (u + 4, v + 4),
                                 (u, v + 12), (u - 4, v + 4), (u - 12, v), (u - 4, v - 4)], np.int32)
                cv2.fillPoly(img, [star], MAGENTA, AA)
                cv2.polylines(img, [star], True, WHITE, 1, AA)


# ---------------------------------------------------------------- 기체 3D 모델

# 기체 좌표 (m): x = 오른쪽, y = 앞, z = 위. 길이 38 m, 날개폭 36 m.
C_BODY, C_WINDOW, C_WING = (242, 242, 242), (95, 75, 60), (215, 215, 218)
C_TAIL, C_ENGINE, C_INTAKE = (170, 95, 30), (165, 165, 170), (45, 45, 50)


def _build_aircraft():
    faces = []   # (꼭짓점 (k,3), 색, 닫힌 입체의 면인지)

    def add(pts, color, ref=None):
        """ref(입체 안쪽 점)를 주면 법선이 바깥을 향하도록 꼭짓점 순서를 맞춘다."""
        pts = np.array(pts, float)
        if ref is not None and _normal(pts) @ (pts.mean(0) - np.array(ref, float)) < 0:
            pts = pts[::-1]
        faces.append((pts, color, ref is not None))

    def tube(stations, n, x_off, color_fn, caps=None):
        ang = [2 * math.pi * j / n for j in range(n)]
        rings = [[(x_off + r * math.cos(a), y, zc + r * math.sin(a)) for a in ang]
                 for y, r, zc in stations]
        for i in range(len(rings) - 1):
            ref = (x_off, (stations[i][0] + stations[i + 1][0]) / 2,
                   (stations[i][2] + stations[i + 1][2]) / 2)
            for j in range(n):
                k = (j + 1) % n
                add([rings[i][j], rings[i][k], rings[i + 1][k], rings[i + 1][j]],
                    color_fn(i, j), ref)
        if caps:
            mid = (x_off, (stations[0][0] + stations[-1][0]) / 2, stations[0][2])
            add(rings[0], caps[0], mid)
            add(rings[-1], caps[1], mid)

    # 동체: (y, 반지름, 중심 높이). 조종석 창과 객실 창 띠는 면 색으로 표현
    fuselage = [(19, 0.2, -0.4), (17.5, 1.15, -0.25), (16.3, 1.55, -0.1), (14.5, 1.85, 0),
                (11, 1.9, 0), (-8, 1.9, 0), (-14, 1.3, 0.45), (-19, 0.3, 1.1)]

    def body_color(i, j):
        if i == 1 and 1 <= j <= 4:
            return C_WINDOW
        if i in (3, 4) and j in (0, 5):
            return C_WINDOW
        return C_BODY

    tube(fuselage, 12, 0.0, body_color)

    for s in (1, -1):   # 좌우 대칭 부품
        add([(s * 1.7, 4.0, -0.9), (s * 18, -4.0, 0.4), (s * 18, -6.5, 0.4), (s * 1.7, -3.5, -0.9)],
            C_WING)                                                     # 주날개
        add([(s * 18, -4.0, 0.4), (s * 18.3, -5.6, 2.2), (s * 18.3, -6.8, 2.2), (s * 18, -6.5, 0.4)],
            C_TAIL)                                                     # 윙렛
        add([(s * 1.0, -14.5, 0.6), (s * 6.5, -17.5, 0.9), (s * 6.5, -19, 0.9), (s * 1.0, -18, 0.6)],
            C_WING)                                                     # 수평 꼬리날개
        tube([(4.6, 1.05, -2.1), (0.4, 0.85, -2.1)], 8, s * 6.0, lambda i, j: C_ENGINE,
             caps=(C_INTAKE, C_INTAKE))                                 # 엔진
    add([(0, -12.5, 1.6), (0, -17.5, 7.8), (0, -19.4, 7.8), (0, -18.6, 1.2)], C_TAIL)   # 수직 꼬리날개

    verts = np.concatenate([f[0] for f in faces])
    ends = np.cumsum([len(f[0]) for f in faces])
    return {
        "verts": verts,
        "spans": list(zip(ends - [len(f[0]) for f in faces], ends)),
        "normals": np.array([_normal(f[0]) for f in faces]),
        "centers": np.array([f[0].mean(0) for f in faces]),
        "colors": np.array([f[1] for f in faces], float),
        "closed": np.array([f[2] for f in faces]),
    }


def _normal(pts):
    """다각형의 단위 법선 (Newell 방법: 찌그러진 사각형에도 안정적)."""
    p = pts - pts.mean(0)
    n = np.cross(p, np.roll(p, -1, axis=0)).sum(0)
    return n / (np.linalg.norm(n) + 1e-12)


_MODEL = _build_aircraft()
# 착륙 장치: (기체 좌표 위쪽 끝, 바퀴 위치)
_GEAR = [((0, 14, -1.7), (0, 14, -GEAR_HEIGHT + 0.5)),
         ((2.8, -1.5, -1.2), (2.8, -1.5, -GEAR_HEIGHT + 0.5)),
         ((-2.8, -1.5, -1.2), (-2.8, -1.5, -GEAR_HEIGHT + 0.5))]
_SHADOW = np.array(ck._OUTLINE, float) * 19.0   # 위에서 본 윤곽 (그림자용)


def body_rotation(heading_deg, bank_deg, pitch_deg):
    """기체 좌표 -> 월드 좌표 회전. 뱅크(+ = 오른쪽 날개 아래) -> 피치(+ = 기수 위) -> 방위."""
    h, b, p = (math.radians(v) for v in (heading_deg, bank_deg, pitch_deg))
    rb = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rp = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    rh = np.array([[math.cos(h), math.sin(h), 0], [-math.sin(h), math.cos(h), 0], [0, 0, 1]])
    return rh @ rp @ rb


def draw_shadow(img, cam, ac):
    """해가 바로 위에 있다고 보고 기체 윤곽을 지면에 떨군다. 높을수록 옅다."""
    rot = body_rotation(ac.heading, 0, 0)[:2, :2]
    pts = _SHADOW @ rot.T + (ac.x, ac.y)
    pc = cam.to_cam(np.column_stack([pts, np.zeros(len(pts))]))
    if (pc[:, 2] < NEAR).any():
        return
    alpha = 0.5 - 0.25 * min(ac.alt / 200.0, 1.0)
    layer = img.copy()
    cv2.fillPoly(layer, [to_px(cam.project(pc))], (20, 30, 20), AA)
    cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


# 항공등: (기체 좌표, 색 BGR, 크기 m, 깜빡임). 깜빡임 None = 계속 켜짐
_NAV_LIGHTS = [
    ((-18.2, -5.4, 1.2), (60, 60, 255), 0.55, None),      # 왼쪽 날개 끝: 빨강
    ((18.2, -5.4, 1.2), (90, 255, 90), 0.55, None),       # 오른쪽 날개 끝: 초록
    ((0.0, -19.3, 1.4), (240, 240, 240), 0.45, None),     # 꼬리: 흰색
    ((0.0, 2.0, 2.2), (60, 60, 255), 0.7, "beacon"),      # 동체 위 충돌 방지등: 빨강 점멸
    ((-18.0, -6.3, 0.5), (255, 255, 255), 0.9, "strobe"),  # 날개 끝 섬광등: 흰색 섬광
    ((18.0, -6.3, 0.5), (255, 255, 255), 0.9, "strobe"),
]


def draw_aircraft_lights(img, cam, rot, pos, clock):
    lit = {None: True, "beacon": clock % 1.2 < 0.25, "strobe": clock % 1.5 < 0.08}
    on = [light for light in _NAV_LIGHTS if lit[light[3]]]
    pts = np.array([p for p, _c, _s, _b in on], float) @ rot.T + pos
    draw_point_lights(img, cam, pts, [c for _p, c, _s, _b in on], [s for _p, _c, s, _b in on])


def draw_parked(img, cam, night, clock):
    """탑승구와 격납고 앞에 서 있는 여객기 (먼 것부터)."""
    order = sorted(ck.PARKED, key=lambda p: -math.hypot(p[0] - cam.eye[0], p[1] - cam.eye[1]))
    for x, y, heading in order:
        if math.hypot(x - cam.eye[0], y - cam.eye[1]) < PARKED_MAX_DISTANCE:
            draw_aircraft(img, cam, _ParkedAircraft(heading), 0.0, 0.0, night, clock,
                          pos=np.array([x, y, GEAR_HEIGHT]), lights=False)


class _ParkedAircraft:
    gear_down = True

    def __init__(self, heading):
        self.heading = heading


def draw_aircraft(img, cam, ac, bank_deg, pitch_deg, night=False, clock=0.0, pos=None, lights=True):
    rot = body_rotation(ac.heading, bank_deg, pitch_deg)
    pos = cam.target if pos is None else pos
    pc = cam.to_cam(_MODEL["verts"] @ rot.T + pos)
    if (pc[:, 2] < NEAR).any():   # 카메라가 기체 안으로 들어간 경우
        return
    uv = to_px(cam.project(pc))

    if ac.gear_down:   # 착륙 장치가 내려와 있을 때만 그린다 (자동이면 낮은 고도에서)
        scale = FOCAL / max(float(np.linalg.norm(pos - cam.eye)), 1.0)
        for top, wheel in _GEAR:
            g = cam.to_cam(np.array([top, wheel], float) @ rot.T + pos)
            if (g[:, 2] < NEAR).any():
                continue
            a, b = to_px(cam.project(g))
            cv2.line(img, tuple(map(int, a)), tuple(map(int, b)), (70, 70, 70),
                     max(1, int(scale * 0.35)), AA)
            cv2.circle(img, tuple(map(int, b)), max(1, int(scale * 0.55)), (30, 30, 30), -1, AA)

    ambient = NIGHT_AMBIENT if night else _wx["dim"]
    normals = _MODEL["normals"] @ rot.T
    centers = _MODEL["centers"] @ rot.T + pos
    facing = np.einsum("ij,ij->i", normals, cam.eye - centers)
    depth = np.linalg.norm(centers - cam.eye, axis=1)
    for i in np.argsort(-depth):   # 먼 면부터
        n = normals[i]
        if facing[i] <= 0:
            if _MODEL["closed"][i]:
                continue   # 입체의 뒷면은 보이지 않는다
            n = -n         # 날개 같은 얇은 판은 양면
        a, b = _MODEL["spans"][i]
        cv2.fillPoly(img, [uv[a:b]], _shaded(_MODEL["colors"][i], n, ambient), AA)
    if night and lights:
        draw_aircraft_lights(img, cam, rot, pos, clock)


def draw_trail(img, cam, trail):
    if len(trail) < 2:
        return
    pts = np.array([(p[0], p[1], (p[2] if len(p) > 2 else 0.0) + GEAR_HEIGHT) for p in trail])
    pc = cam.to_cam(pts)
    ok = pc[:, 2] >= NEAR
    uv = to_px(cam.project(np.where(ok[:, None], pc, [0, 0, 1.0])))
    start = None
    for i in range(len(pts) + 1):   # 카메라 앞에 있는 구간만 이어서 그린다
        if i < len(pts) and ok[i]:
            start = i if start is None else start
        elif start is not None:
            if i - start >= 2:
                cv2.polylines(img, [uv[start:i]], False, CYAN, 2, AA)
            start = None


# ---------------------------------------------------------------- 화면 조립

HELP_3D = "DRAG/ARROWS: LOOK  + -: ZOOM  1379: VIEW  5: RESET  A: PILOT/OUTSIDE  V: 2D  N: NIGHT  W: WX  F: FLAPS  G: GEAR"
HELP_2D = "V: 3D VIEW    N: DAY/NIGHT    W: WEATHER    F: FLAPS    G: GEAR"
MINIMAP = 170
_mini = {"img": None, "n": 0}


def draw_marks(img, cam, fx):
    """이륙·접지 지점을 지면 원과 깃대로 표시."""
    for mark, label, color in ((fx.liftoff, "LIFTOFF", ck.GREEN), (fx.touchdown, "TOUCHDOWN", WHITE)):
        if not mark:
            continue
        x, y = mark
        ang = np.linspace(0, 2 * math.pi, 17)[:-1]
        fill_world(img, cam, [(x + 5 * math.cos(a), y + 5 * math.sin(a), 0) for a in ang], color)
        line_world(img, cam, (x, y, 0), (x, y, 14), color, 2)
        pc = cam.to_cam([(x, y, 14)])
        if pc[0, 2] >= NEAR:
            u, v = (int(k) for k in to_px(cam.project(pc))[0])
            ck.text(img, label, (u + 6, v + 4), 0.42, color)


def draw_smoke(img, cam, fx):
    """접지 순간의 타이어 연기."""
    for x, y, z, radius, alpha in fx.smoke():
        pc = cam.to_cam([(x, y, z)])
        if pc[0, 2] >= NEAR:
            u, v = to_px(cam.project(pc))[0]
            ck.blend_circle(img, (u, v), FOCAL * radius / pc[0, 2], (228, 228, 228), alpha)


WINDSOCKS = [(-45.0, 150.0), (-45.0, 850.0)]   # 활주로 양 끝 옆의 바람자루 위치


def draw_windsocks(img, cam, env):
    """바람자루: 바람이 가는 쪽으로 눕고, 약하면 축 처진다."""
    wx, wy = env.wind
    speed = math.hypot(wx, wy)
    ux, uy = (wx / speed, wy / speed) if speed > 0.3 else (0.0, 0.0)
    lift = min(speed / 8.0, 1.0)   # 8 m/s 이상이면 수평으로 뻗는다
    reach, drop = 7.0 * (0.25 + 0.75 * lift), 5.0 * (1.0 - lift)
    for px, py in WINDSOCKS:
        line_world(img, cam, (px, py, 0), (px, py, 10.2), (210, 210, 210), 2)
        tx, ty, tz = px + ux * reach, py + uy * reach, 9.0 - drop
        fill_world(img, cam, [(px, py, 10.2), (px, py, 7.8), (tx, ty, tz - 0.5), (tx, ty, tz + 0.5)],
                   (0, 130, 255))


def draw_scene(view, sc, ac, trail, fx):
    bank_deg, pitch_deg = fx.bank, fx.pitch
    night = view.night
    cam = Camera(view, ac, pitch_deg)
    cockpit = cam.cockpit
    img = draw_ground(cam, night)
    if _terrain is not None:   # 순항 구간: 공항·도시·경로 표시 없이 하늘과 구름, 기체만
        draw_fog_and_sky(img, cam, night)
        draw_clouds(img, cam, night, fx.clock)
        draw_guide_lane(img, cam)
        if not cockpit:
            draw_aircraft(img, cam, ac, bank_deg, pitch_deg, night, fx.clock)
        draw_lightning(img, fx.clock, cam.horizon_y)
        return apply_motion(grade(img), fx, bank_deg if cockpit else None, _real_ac, night)
    draw_targets(img, cam, sc)
    draw_guide_lane(img, cam)
    draw_marks(img, cam, fx)
    if night:
        draw_landing_light(img, cam, ac)   # 밤에는 그림자 대신 착륙등 불빛
    elif _wx["sun"] and not cockpit:
        draw_shadow(img, cam, ac)          # 해가 가려진 날에는 그림자가 없다
    draw_fog_and_sky(img, cam, night)
    wind = fx.env.wind if fx.env is not None and fx.env.wind_speed > 0 else (3.0, -2.5)
    draw_clouds(img, cam, night, fx.clock, wind)
    model = not _sat_layers   # 모형 도시·공항 건물은 위성 사진 지면에서는 그리지 않는다
    if model:
        faces = city_faces(cam, night)
    elif _sat_buildings is not None:   # 위성 사진 지면: 그 자리에 실제로 있는 건물
        faces = city_faces(cam, night, _sat_buildings, SAT_FOG)
    else:
        faces = None
    # 조종석에서는 자기 기체를 그리지 않으므로 모든 건물을 한 번에 그린다
    plane_depth = 0.0 if cockpit else float(cam.to_cam([cam.target])[0, 2])
    if faces is not None:
        draw_city(img, faces, farther_than=plane_depth)   # 기체보다 먼 건물
    if night and model:
        draw_night_lights(img, cam, fx.clock)
    if night or not _wx["sun"]:   # 등화는 밤과, 낮이라도 날이 궂어 해가 가려졌을 때 켠다
        draw_runway_lights(img, cam, ac, fx.clock, runway_only=not model)
    # 주기 중인 여객기는 터미널의 동쪽에 있다: 카메라가 그보다 서쪽이면 여객기를 먼저 그린다
    if not model:
        pass
    elif cam.eye[0] < -340.0:
        draw_parked(img, cam, night, fx.clock)
        draw_buildings(img, cam, night, fx.clock)
    else:
        draw_buildings(img, cam, night, fx.clock)
        draw_parked(img, cam, night, fx.clock)
    if fx.env is not None:
        draw_windsocks(img, cam, fx.env)
    draw_trail(img, cam, trail)
    if not cockpit:
        if ac.alt > 3.0:   # 지면 위 위치를 알 수 있게 수직선
            line_world(img, cam, (ac.x, ac.y, 0), (ac.x, ac.y, ac.alt + GEAR_HEIGHT - 2),
                       (230, 230, 230), 1)
        draw_aircraft(img, cam, ac, bank_deg, pitch_deg, night, fx.clock)
        if faces is not None:
            draw_city(img, faces, nearer_than=plane_depth)    # 기체보다 가까운 건물은 기체를 가린다
        draw_smoke(img, cam, fx)
    draw_lightning(img, fx.clock, cam.horizon_y)
    img = grade(img)
    return apply_motion(img, fx, bank_deg if cockpit else None, ac, night)


COCKPIT_ZOOM = 1.3            # 조종석 그림을 돌려도 모서리가 비지 않게 조금 키운다
COCKPIT_CENTER = (350, 262)   # 앞 유리창의 가운데 (지평선이 놓이는 곳)
C_FRAME, C_FRAME_EDGE, C_DASH = (40, 38, 36), (78, 76, 72), (30, 29, 28)


def apply_motion(img, fx, bank_deg, ac, night=False):
    """움직임 효과(화면 흔들림)를 입힌다. bank_deg를 주면 조종석 시점: 기울기만큼 바깥 풍경을 돌린다."""
    dx, dy, rot = fx.motion
    if bank_deg is None:   # 뒤에서 따라가는 시점
        if abs(dx) + abs(dy) + abs(rot) < 0.3:
            return img
        m = cv2.getRotationMatrix2D((CX, CY), rot, 1.0)
        m[:, 2] += (dx, dy)
        return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 조종석: 오른쪽으로 기울면 바깥 지평선은 반시계 방향으로 돈다. 흔들림은 더 크게 느껴진다
    m = cv2.getRotationMatrix2D((CX, CY), bank_deg + rot * 1.5, COCKPIT_ZOOM)
    m[:, 2] += (COCKPIT_CENTER[0] - CX + dx * 1.8, COCKPIT_CENTER[1] - CY + dy * 1.8)
    out = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    draw_cockpit_frame(out, ac, night, fx)
    return out


def _make_cockpit_art():
    """조종석 내부 그림 (움직이지 않는 부분): 창틀, 기둥, 계기판 덮개, 계기판과 화면 테두리.

    한 번만 그려 두고 프레임마다 바깥 풍경 위에 겹친다. (미리 곱한 색, 투명도) 두 장을 돌려준다.
    빛이 위에서 들어오는 것처럼 위쪽은 밝게, 아래쪽과 안쪽은 어둡게 칠하고 표면에는 잔결을 넣는다.
    """
    rng = np.random.default_rng(6)
    art = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.8)

    def fill(pts, top, bottom, y1, y2, rough=3.0):
        """다각형을 위(top)에서 아래(bottom)로 바뀌는 색으로 칠한다. rough = 표면 잔결의 세기."""
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255, AA)
        m = mask.astype(np.float32) / 255.0
        k = np.clip((yy - y1) / max(y2 - y1, 1), 0, 1)[..., None]
        color = np.array(top, np.float32) * (1 - k) + np.array(bottom, np.float32) * k + (grain * rough)[..., None]
        art[:] = art * (1 - m[..., None]) + color * m[..., None]
        np.maximum(alpha, m, out=alpha)

    def line(a, b, color, thick=1):
        cv2.line(art, a, b, color, thick, AA)

    # 유리창 가장자리의 옅은 그늘 (창틀 가까이는 유리가 어둡게 보인다)
    edge_shade = np.clip(1.0 - np.minimum(np.minimum(xx - 60, W - 60 - xx), yy - 30) / 45.0, 0, 1) * 0.28
    edge_shade[470:] = 0
    np.maximum(alpha, edge_shade, out=alpha)

    # 위쪽 창틀과 차양
    fill([(0, 0), (W, 0), (W, 30), (566, 42), (134, 42), (0, 30)], (58, 56, 54), (30, 29, 28), 0, 42)
    line((134, 42), (566, 42), (96, 94, 90), 1)
    # 좌우 기둥: 안쪽 모서리에 밝은 띠, 바깥쪽은 어둡게
    for sign, x0 in ((1, 0), (-1, W)):
        fill([(x0, 0), (x0 + sign * 80, 0), (x0 + sign * 48, 478), (x0, 492)], (62, 60, 58), (34, 33, 32), 0, 490)
        fill([(x0 + sign * 66, 0), (x0 + sign * 80, 0), (x0 + sign * 48, 478), (x0 + sign * 36, 484)],
             (84, 82, 78), (50, 49, 47), 0, 490, 2.0)
        line((x0 + sign * 80, 0), (x0 + sign * 48, 478), (18, 18, 18), 2)       # 유리 고무 테
    # 가운데 기둥
    fill([(341, 38), (359, 38), (364, 464), (336, 464)], (60, 58, 56), (36, 35, 34), 38, 464)
    fill([(347, 38), (353, 38), (355, 464), (345, 464)], (86, 84, 80), (54, 53, 51), 38, 464, 1.5)
    for x1, x2 in ((341, 336), (359, 364)):
        line((x1, 38), (x2, 464), (18, 18, 18), 1)

    # 계기판 덮개 (윗면이 볼록하고, 앞 가장자리에 빛이 걸린다)
    lip = [(x, int(462 + 24 * ((x - 350) / 350) ** 2)) for x in range(0, W + 1, 25)]
    fill([(0, 488)] + lip + [(W, 488), (W, 540), (0, 540)], (52, 50, 48), (26, 25, 24), 462, 540, 4.0)
    cv2.polylines(art, [np.array(lip, np.int32)], False, (110, 108, 104), 2, AA)
    cv2.polylines(art, [np.array([(x, y + 3) for x, y in lip], np.int32)], False, (20, 20, 20), 2, AA)
    # 계기판 본체
    fill([(0, 534), (W, 534), (W, H), (0, H)], (44, 43, 42), (24, 23, 23), 534, H, 2.5)
    line((0, 534), (W, 534), (14, 14, 14), 3)
    line((0, 537), (W, 537), (70, 68, 66), 1)

    def bezel(x1, y1, x2, y2, screws=True):
        """화면 테두리: 둥근 모서리의 어두운 틀과 안쪽 그림자."""
        cv2.rectangle(art, (x1 - 7, y1 - 7), (x2 + 7, y2 + 7), (22, 22, 22), -1, AA)
        cv2.rectangle(art, (x1 - 7, y1 - 7), (x2 + 7, y2 + 7), (86, 84, 80), 1, AA)
        cv2.rectangle(art, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (6, 6, 6), -1)
        if screws:
            for sx, sy in ((x1 - 4, y1 - 4), (x2 + 4, y1 - 4), (x1 - 4, y2 + 4), (x2 + 4, y2 + 4)):
                cv2.circle(art, (sx, sy), 2, (120, 118, 114), -1, AA)
                cv2.circle(art, (sx, sy), 2, (10, 10, 10), 1, AA)

    for box in COCKPIT_SCREENS.values():
        bezel(*box)
    # 자동 비행 조작판 (덮개 앞면): 숫자 창 네 개와 손잡이
    cv2.rectangle(art, (70, 494), (500, 532), (20, 20, 20), -1, AA)
    cv2.rectangle(art, (70, 494), (500, 532), (92, 90, 86), 1, AA)
    for i in range(4):
        x = 78 + i * 104
        cv2.rectangle(art, (x, 499), (x + 66, 527), (4, 4, 4), -1)
        cv2.rectangle(art, (x, 499), (x + 66, 527), (70, 68, 66), 1)
        cv2.circle(art, (x + 84, 513), 9, (58, 56, 54), -1, AA)      # 돌리는 손잡이
        cv2.circle(art, (x + 84, 513), 9, (112, 110, 106), 1, AA)
        line((x + 84, 513), (x + 84, 506), (190, 188, 184), 2)
    alpha3 = cv2.merge([alpha] * 3)
    return art * alpha3, 1.0 - alpha3


# 계기판의 화면 자리 (x1, y1, x2, y2): 주 비행 화면, 엔진 화면, 항법 화면(미니맵이 놓이는 자리)
COCKPIT_SCREENS = {"pfd": (78, 552, 238, 668), "eicas": (262, 552, 496, 668), "nd": (520, 496, 690, 666)}
_COCKPIT_ART, _COCKPIT_CLEAR = _make_cockpit_art()
C_SCREEN_TEXT, C_SCREEN_DIM = (90, 230, 90), (150, 150, 150)


def draw_cockpit_frame(img, ac, night=False, fx=None):
    """조종석 내부를 바깥 풍경 위에 겹치고, 계기판 화면에 지금 값을 그린다."""
    lit = 0.45 if night else 1.0   # 밤에는 실내가 어둡고 화면만 밝다
    blend = cv2.multiply(img, _COCKPIT_CLEAR, dtype=cv2.CV_32F)
    cv2.scaleAdd(_COCKPIT_ART, lit, blend, blend)
    img[:] = cv2.convertScaleAbs(blend)

    # 자동 비행 조작판의 숫자: 속도, 방위, 고도, 상승률
    readouts = (("SPD", f"{ac.speed:03.0f}"), ("HDG", f"{ck.shown_heading(ac):03.0f}"),
                ("ALT", f"{ac.alt:05.0f}"), ("V/S", f"{ac.vs:+05.1f}"))
    for i, (label, value) in enumerate(readouts):
        x = 78 + i * 104
        ck.text(img, label, (x + 3, 508), 0.27, C_SCREEN_DIM)
        ck.text(img, value, (x + 63, 523), 0.5, (60, 170, 255), 1, "r")

    # 주 비행 화면: 자세계와 속도·고도
    x1, y1, x2, y2 = COCKPIT_SCREENS["pfd"]
    bank, pitch = (fx.bank, fx.pitch) if fx is not None else (0.0, 0.0)
    ck.draw_adi(img, x1, y1, x2 - x1, y2 - y1, bank, pitch)
    cy = (y1 + y2) // 2
    for label, value, bx in (("SPD", f"{ac.speed:.0f}", x1 + 1), ("ALT", f"{ac.alt:.0f}", x2 - 39)):   # 양옆의 값 상자
        cv2.rectangle(img, (bx, cy - 34), (bx + 38, cy - 12), (0, 0, 0), -1)
        cv2.rectangle(img, (bx, cy - 34), (bx + 38, cy - 12), (200, 200, 200), 1)
        ck.text(img, value, (bx + 35, cy - 18), 0.36, C_SCREEN_TEXT, 1, "r")
        ck.text(img, label, (bx + 2, cy - 38), 0.26, WHITE)
    ck.text(img, f"V/S {ac.vs:+.1f}", (x2 - 3, y2 - 5), 0.3, WHITE if abs(ac.vs) < 6 else (0, 190, 255), 1, "r")

    # 엔진 화면: 좌우 엔진 추력 계기, 플랩, 착륙 장치
    x1, y1, x2, y2 = COCKPIT_SCREENS["eicas"]
    n1 = 0.0 if ac.engine_failed else min(ac.speed / 70.0, 1.0)
    needle = (60, 60, 235) if ac.engine_failed else (245, 245, 245)
    for k in range(2):
        cx, cy, r = x1 + 34 + k * 62, y1 + 40, 25
        cv2.ellipse(img, (cx, cy), (r, r), 0, 150, 390, (210, 210, 210), 1, AA)
        cv2.ellipse(img, (cx, cy), (r, r), 0, 150 + 240 * 0.9, 390, (60, 60, 235), 2, AA)   # 한계 구간
        a = math.radians(150 + 240 * n1)
        cv2.line(img, (cx, cy), (int(cx + (r - 3) * math.cos(a)), int(cy + (r - 3) * math.sin(a))), needle, 2, AA)
        ck.text(img, f"{100 * n1:4.1f}", (cx, cy + 22), 0.33, needle if ac.engine_failed else C_SCREEN_TEXT, 1, "c")
    ck.text(img, "ENG FAIL" if ac.engine_failed else "N1 %", (x1 + 65, y2 - 8), 0.3,
            (60, 60, 235) if ac.engine_failed else C_SCREEN_DIM, 1, "c")
    fx0 = x1 + 140
    ck.text(img, "FLAPS", (fx0, y1 + 16), 0.3, C_SCREEN_DIM)
    for i, name in enumerate(("UP", "1", "2", "F")):
        on = i == ac.flaps
        bx = fx0 + i * 23
        cv2.rectangle(img, (bx, y1 + 22), (bx + 20, y1 + 40), (30, 120, 30) if on else (20, 20, 20), -1)
        cv2.rectangle(img, (bx, y1 + 22), (bx + 20, y1 + 40), (90, 90, 90), 1)
        ck.text(img, name, (bx + 10, y1 + 35), 0.3, (245, 245, 245) if on else (120, 120, 120), 1, "c")
    ck.text(img, "GEAR", (fx0, y1 + 62), 0.3, C_SCREEN_DIM)
    for i in range(3):   # 세 바퀴: 내려와 있으면 초록, 올라가 있으면 꺼짐
        cv2.circle(img, (fx0 + 8 + i * 22, y1 + 78), 7, (70, 220, 70) if ac.gear_down else (30, 30, 30), -1, AA)
        cv2.circle(img, (fx0 + 8 + i * 22, y1 + 78), 7, (110, 110, 110), 1, AA)
    ck.text(img, ("DOWN" if ac.gear_down else "UP") + ("" if not getattr(ac, "gear_auto", False) else " AUTO"),
            (fx0, y2 - 8), 0.3, (70, 220, 70) if ac.gear_down else (0, 190, 255))


def render(view, sc, ac, trail, ctrl, cam_frame, hand, phase, t, info, fx):
    """한 프레임 전체(왼쪽 화면 + 계기판)를 그린다."""
    global _wx, _terrain, _real_ac, _guide, _sat_layers, _sat_buildings
    _wx = WEATHERS[view.weather]
    _real_ac = ac
    airway = info.get("airway")   # 항로 비행의 순항 구간: 지도는 세계 지도, 지면은 바다·육지
    # 순항 구간이라도 아직 출발 공항 근처면(local) 공항·도시가 있는 원래 풍경을 그린다
    _terrain = airway["land"] if airway and not airway.get("local") else None
    _guide = airway["guide"] if airway else []
    sat = info.get("sat") if view.satellite else None
    _sat_layers = sat.layers() if sat is not None else []
    _sat_buildings = sat.buildings if _sat_layers and _terrain is None else None
    if airway and view.mode != "3d":
        scene = airway["map"].draw(airway["frac"])
        help_text = HELP_2D
    elif view.mode == "3d":
        scene = draw_scene(view, sc, airway["visual"] if airway else ac, trail, fx)
        # 미니맵 (오른쪽 아래): 경로에서 벗어난 정도를 위에서 확인
        _mini["n"] += 1
        if airway:
            _mini["img"] = airway["minimap"].draw(airway["frac"])
        elif _mini["img"] is None or _mini["n"] % 3 == 0:   # 미니맵은 3프레임에 한 번만 다시 그린다
            _mini["img"] = cv2.resize(ck.draw_map(sc, ac, trail, decorations=False, fx=fx, night=view.night, sat_layers=_sat_layers),
                                      (MINIMAP, MINIMAP), interpolation=cv2.INTER_AREA)
        mini = _mini["img"]
        x0, y0 = W - MINIMAP - 10, H - MINIMAP - 34
        scene[y0:y0 + MINIMAP, x0:x0 + MINIMAP] = mini
        cv2.rectangle(scene, (x0 - 1, y0 - 1), (x0 + MINIMAP, y0 + MINIMAP), (200, 200, 200), 1)
        help_text = HELP_3D
    else:
        scene = ck.draw_map(sc, ac, trail, fx=fx, night=view.night, sat_layers=_sat_layers)
        draw_lightning(scene, fx.clock)
        help_text = HELP_2D
    if view.mode == "3d" and view.cam == "cockpit" and info.get("fd") is not None:
        # 앞 유리창의 조종 안내: 가운데 기준 표시에 마젠타 십자선을 맞추면 된다
        cx, cy = COCKPIT_CENTER
        ck.draw_fd(scene, cx, cy, info["fd"], reach=90, size=60)
        for sign in (-1, 1):
            cv2.line(scene, (cx + sign * 14, cy), (cx + sign * 40, cy), (0, 0, 0), 4, AA)
            cv2.line(scene, (cx + sign * 14, cy), (cx + sign * 40, cy), GUIDE_REF, 2, AA)
        cv2.circle(scene, (cx, cy), 5, (0, 0, 0), 3, AA)
        cv2.circle(scene, (cx, cy), 5, GUIDE_REF, 2, AA)
    draw_precipitation(scene, fx.clock)
    ck.draw_overlays(scene, sc, ac, phase, t, info, fx)
    if airway:   # 길 안내: 목적지, 남은 거리, 남은 시간
        ck.shade(scene, 296, 48, 692, 76, (60, 20, 60), 0.75)
        cv2.rectangle(scene, (296, 48), (692, 76), MAGENTA, 1)
        ck.text(scene, airway["banner"], (494, 68), 0.5, WHITE, 1, "c")
    if _sat_layers and view.mode == "3d":   # 위성 사진 출처 표기 (CC BY 조건)
        ck.text(scene, satellite.CREDIT, (8, 476 if view.cam == "cockpit" else H - 30), 0.3, (215, 215, 215))
    draw_time_button(scene, view.night)
    draw_cam_button(scene, view.cam)
    draw_weather_button(scene, view.weather)
    if info.get("hint"):   # 모드별 추가 조작 안내 (예: 자유 비행의 R 키)
        help_text += "    " + info["hint"]
    if view.mode == "3d":
        ck.shade(scene, 0, H - 24, W, H, alpha=0.6)
        ck.text(scene, help_text, (10, H - 8), 0.36, (225, 225, 225))
    else:
        ck.text(scene, help_text, (W - 12, H - 12), 0.42, WHITE, 1, "r")
    return ck.compose(scene, sc, ac, ctrl, cam_frame, hand, info, fx)
