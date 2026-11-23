"""실제 항로 비행 (자동 조종 모드): 두 공항 사이를 대권 항로로 잇고, 순항 구간을 시간 압축해 보여준다.

- 공항: data/airports.csv (전 세계, IATA 코드가 있는 공항 약 6,000곳). 출처는 OpenFlights 공항 데이터베이스
  (https://openflights.org/data, Open Database License)이고 이름의 악센트 기호는 뺐다.
- 해안선: data/land.npz. 출처는 Natural Earth 1:50m 육지 다각형 (https://www.naturalearthdata.com, 퍼블릭 도메인).
  지도 표시와 "지금 바다 위인지 육지 위인지" 판단에 쓴다.
- 경로: 출발 공항과 도착 공항을 잇는 대권(지구 위의 최단 경로). 실제 운항편은 항공로·제트기류에 맞춰
  이보다 조금 돌아가지만, 큰 모양은 같다.
- 고도·속도: 순항 고도까지 상승 -> 순항(마하 0.8대, 약 250 m/s) -> 강하. 값은 여객기의 대표적인 수치다.
- 시간 압축: 실제로는 몇 시간 걸리는 순항을 1~3분에 보여준다. 화면의 비행 시간은 실제 기준이다.
"""

import csv
import math
from pathlib import Path

import cv2
import numpy as np

EARTH_RADIUS = 6371.0   # km
AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_SIMPLEX
DATA = Path(__file__).parent / "data"

# 자주 쓰는 공항의 한글 이름 (메뉴에 먼저 보여준다). 나머지 공항은 코드나 도시 이름으로 찾는다
KOREAN_NAMES = {
    "ICN": "서울 인천", "GMP": "서울 김포", "PUS": "부산 김해", "CJU": "제주",
    "NRT": "도쿄 나리타", "HND": "도쿄 하네다", "KIX": "오사카 간사이", "FUK": "후쿠오카",
    "PEK": "베이징", "PVG": "상하이 푸둥", "HKG": "홍콩", "TPE": "타이베이",
    "BKK": "방콕", "SIN": "싱가포르", "SYD": "시드니", "DXB": "두바이",
    "LHR": "런던 히스로", "CDG": "파리 샤를 드골", "FRA": "프랑크푸르트", "FCO": "로마",
    "JFK": "뉴욕", "LAX": "로스앤젤레스", "SFO": "샌프란시스코", "SEA": "시애틀",
    "YVR": "밴쿠버", "HNL": "호놀룰루", "ORD": "시카고", "GRU": "상파울루",
    "JNB": "요하네스버그", "IST": "이스탄불",
}


def _load_airports():
    """코드 -> (화면 표시 이름(영문 대문자), 위도, 경도, 메뉴 표시 이름, 도시, 나라)."""
    airports = {}
    with (DATA / "airports.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            code = row["iata"]
            shown = KOREAN_NAMES.get(code) or f"{row['city'] or row['name']}, {row['country']}"
            airports[code] = ((row["city"] or row["name"]).upper(), float(row["lat"]), float(row["lon"]),
                              shown, row["city"], row["country"], row["name"])
    return airports


AIRPORTS = _load_airports()
MIN_DISTANCE = 150.0   # 이보다 가까운 두 공항은 항로 비행으로 다루지 않는다 (km)


def search_airports(text, limit=12):
    """도시·공항·나라 이름에 text가 들어 있는 공항 코드들 (도시 이름이 그 글자로 시작하는 것 먼저)."""
    q = text.strip().lower()
    hits = [(not a[4].lower().startswith(q), code) for code, a in AIRPORTS.items()
            if q in a[4].lower() or q in a[6].lower() or q == a[5].lower()]
    return [code for _rank, code in sorted(hits)][:limit]


CRUISE_SPEED = 250.0     # 순항 속도 (m/s, 약 900 km/h)
CLIMB_SPEED = 140.0      # 상승 초기·접근 속도 (m/s)
START_ALT = 100.0        # 순항 구간이 시작되는 고도 (이륙 과제가 끝나는 고도, m)
END_ALT = 600.0          # 순항 구간이 끝나는 고도 (그 뒤는 착륙 과제, m)
JETSTREAM = 28.0         # 동쪽으로 갈 때 받는 평균 뒷바람 (m/s). 서쪽으로 가면 맞바람


def _wrap180(lon):
    return (lon + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------- 육지 (경도 -180~180, 위도)

def _load_lands():
    data = np.load(DATA / "land.npz")
    rings = np.split(data["points"], np.cumsum(data["sizes"])[:-1])
    return [(ring, ring.min(0), ring.max(0)) for ring in rings]


LANDS = _load_lands()   # (다각형 (N,2) [경도, 위도], 최솟값, 최댓값)


def over_land(lat, lon) -> bool:
    lon = _wrap180(lon)
    for ring, lo, hi in LANDS:
        if lo[0] <= lon <= hi[0] and lo[1] <= lat <= hi[1] \
                and cv2.pointPolygonTest(ring, (float(lon), float(lat)), False) >= 0:
            return True
    return False


# ---------------------------------------------------------------- 대권 항로

def _unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


class Route:
    """출발 공항에서 도착 공항까지의 대권 항로와 비행 계획(고도·속도)."""

    def __init__(self, origin, dest):
        self.origin, self.dest = origin, dest
        self.label = f"{AIRPORTS[origin][3]} ({origin}) → {AIRPORTS[dest][3]} ({dest})"
        self._a = _unit(*AIRPORTS[origin][1:3])
        self._b = _unit(*AIRPORTS[dest][1:3])
        self._angle = math.acos(float(np.clip(self._a @ self._b, -1.0, 1.0)))
        self.distance = self._angle * EARTH_RADIUS   # km
        # 비행 계획: 거리가 짧으면 낮게 날고, 상승·강하 구간도 짧다
        self.eastbound = _wrap180(AIRPORTS[dest][2] - AIRPORTS[origin][2]) > 0.0
        self.cruise_alt = 10700.0 if self.distance > 1500 else 10000.0 if self.distance > 800 else 7300.0
        if not self.eastbound:
            self.cruise_alt -= 300.0   # 실제 관례처럼 서쪽으로 갈 때는 동쪽 방향과 다른 고도를 쓴다
        self.climb_km = min(220.0, self.distance * 0.3)
        self.descent_km = min(240.0, self.distance * 0.35)
        # 순항 구간을 화면에서 보여주는 시간 (초): 장거리 150초, 단거리 60초
        self.show_seconds = float(np.clip(self.distance / 60.0, 60.0, 150.0))
        self.points = [self.position(i / 200) for i in range(201)]   # 지도에 그릴 경로

    def position(self, frac):
        """경로의 frac(0~1) 지점의 (위도, 경도 -180~180)."""
        s = math.sin(self._angle)
        p = (math.sin((1 - frac) * self._angle) * self._a + math.sin(frac * self._angle) * self._b) / s
        return math.degrees(math.asin(float(np.clip(p[2], -1, 1)))), math.degrees(math.atan2(p[1], p[0]))

    def bearing(self, frac):
        """그 지점에서의 진행 방위 (도, 북 = 0, 시계 방향)."""
        la1, lo1 = self.position(min(frac, 0.999))
        la2, lo2 = self.position(min(frac + 0.001, 1.0))
        d = math.radians(lo2 - lo1)
        y = math.sin(d) * math.cos(math.radians(la2))
        x = (math.cos(math.radians(la1)) * math.sin(math.radians(la2))
             - math.sin(math.radians(la1)) * math.cos(math.radians(la2)) * math.cos(d))
        return math.degrees(math.atan2(y, x)) % 360.0

    def _profile(self, km):
        """상승·강하 진행 정도 0(공항 근처)~1(순항)."""
        up = km / self.climb_km
        down = (self.distance - km) / self.descent_km
        k = float(np.clip(min(up, down), 0.0, 1.0))
        return k * k * (3 - 2 * k)   # 부드럽게 시작하고 끝난다

    def climb_progress(self, km):
        """0 = 공항 근처(상승 시작·강하 끝) ~ 1 = 순항 중."""
        return self._profile(km)

    def altitude(self, km):
        base = START_ALT if km < self.distance / 2 else END_ALT
        return base + (self.cruise_alt - base) * self._profile(km)

    def airspeed(self, km):
        return CLIMB_SPEED + (CRUISE_SPEED - CLIMB_SPEED) * self._profile(km)

    def ground_speed(self, km):
        wind = JETSTREAM * self._profile(km)   # 제트기류는 높은 고도에서만 분다
        return self.airspeed(km) + (wind if self.eastbound else -wind)

    @property
    def flight_seconds(self):
        """순항 구간의 실제 소요 시간 (초)."""
        step = self.distance / 400
        return sum(step * 1000.0 / self.ground_speed((i + 0.5) * step) for i in range(400))


# ---------------------------------------------------------------- 지도

OCEAN = (92, 62, 30)
LAND = (78, 104, 82)
COAST = (120, 150, 126)
GRID = (112, 82, 48)
ROUTE_AHEAD = (255, 0, 255)
ROUTE_DONE = (255, 230, 0)
WHITE = (240, 240, 240)


def _text(img, s, org, size, color, thick=1, anchor="l"):
    (tw, _th), _ = cv2.getTextSize(s, FONT, size, thick)
    x = org[0] - (tw if anchor == "r" else tw // 2 if anchor == "c" else 0)
    cv2.putText(img, s, (int(x), int(org[1])), FONT, size, (0, 0, 0), thick + 2, AA)
    cv2.putText(img, s, (int(x), int(org[1])), FONT, size, color, thick, AA)


class RouteMap:
    """항로 전체가 들어오는 지도. 바탕(바다·육지·경위선·경로)은 한 번만 그려 둔다."""

    def __init__(self, route, width, height, labels=True):
        self.route, self.w, self.h = route, width, height
        lats = np.array([p[0] for p in route.points])
        # 경도는 출발지부터 이어 붙인다 (날짜변경선이나 본초 자오선을 넘어도 끊기지 않게)
        self._lons = np.degrees(np.unwrap(np.radians([p[1] for p in route.points])))
        self.center = (self._lons.max() + self._lons.min()) / 2
        # 경로가 화면에 들어오도록 범위를 잡는다. 위도 1도를 경도 1.4도만큼 크게 그려 중위도 모양에 맞춘다
        stretch = 1.4
        span = max((self._lons.max() - self._lons.min()) * 1.25,
                   (lats.max() - lats.min()) * 1.5 * stretch * width / height, 9.0)
        self.span = span
        self.scale = width / span                       # 픽셀 / 경도 1도
        self.lat_scale = self.scale * stretch
        self.lat1 = (lats.max() + lats.min()) / 2 + height / self.lat_scale / 2
        self._path = np.array([self._xy(lon, lat) for lon, lat in zip(self._lons, lats)], np.int32)
        self.base = self._draw_base(labels)

    def _xy(self, lon, lat):
        """이어 붙인 경도(지도 중심 기준으로 정리된 값) -> 화면."""
        return (int(round((lon - self.center) * self.scale + self.w / 2)),
                int(round((self.lat1 - lat) * self.lat_scale)))

    def to_px(self, lat, lon):
        return self._xy(self.center + _wrap180(lon - self.center), lat)

    def _draw_base(self, labels):
        img = np.empty((self.h, self.w, 3), np.uint8)
        img[:] = OCEAN
        half = self.span / 2
        for ring, lo, hi in LANDS:
            if hi[1] < self.lat1 - self.h / self.lat_scale or lo[1] > self.lat1:
                continue
            # 다각형의 경도를 지도 중심 근처로 옮기고, 화면에 걸칠 수 있는 360도 복사본을 모두 그린다
            shift = round((self.center - (lo[0] + hi[0]) / 2) / 360.0) * 360.0
            for extra in (-360.0, 0.0, 360.0):
                a, b = lo[0] + shift + extra, hi[0] + shift + extra
                if b < self.center - half or a > self.center + half:
                    continue
                pts = np.column_stack([(ring[:, 0] + shift + extra - self.center) * self.scale + self.w / 2,
                                       (self.lat1 - ring[:, 1]) * self.lat_scale])
                pts = np.round(np.clip(pts, -20000, 20000)).astype(np.int32)
                cv2.fillPoly(img, [pts], LAND, AA)
                cv2.polylines(img, [pts], True, COAST, 1, AA)
        step = 30 if self.span > 90 else 10 if self.span > 30 else 2   # 경위선 간격 (도)
        first = math.floor((self.center - half) / step) * step
        for k in range(int(self.span / step) + 2):
            x = self._xy(first + k * step, 0)[0]
            cv2.line(img, (x, 0), (x, self.h), GRID, 1)
        for lat in range(-90, 91, step):
            y = self._xy(0, lat)[1]
            cv2.line(img, (0, y), (self.w, y), GRID, 1)
        cv2.polylines(img, [self._path], False, (110, 0, 110), 4 if labels else 2, AA)
        cv2.polylines(img, [self._path], False, ROUTE_AHEAD, 2 if labels else 1, AA)
        for code, (x, y) in ((self.route.origin, self._path[0]), (self.route.dest, self._path[-1])):
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 5 if labels else 3, (0, 0, 0), -1, AA)
            cv2.circle(img, (x, y), 4 if labels else 2, WHITE, -1, AA)
            if labels:
                right = x < self.w - 90
                _text(img, code, (x + (9 if right else -9), y - 8), 0.55, WHITE, 2, "l" if right else "r")
        return img

    def _at(self, frac):
        """경로의 frac 지점의 화면 좌표 (경로 점 사이는 직선으로 잇는다)."""
        k = float(np.clip(frac, 0.0, 1.0)) * (len(self._path) - 1)
        i = min(int(k), len(self._path) - 2)
        return self._path[i] + (self._path[i + 1] - self._path[i]) * (k - i)

    def draw(self, frac):
        """지금 위치(frac)까지 지나온 경로와 기체 기호를 얹은 지도."""
        img = self.base.copy()
        x, y = self._at(frac)
        n = int(frac * (len(self._path) - 1))
        done = np.vstack([self._path[:n + 1], [(x, y)]]).astype(np.int32)
        cv2.polylines(img, [done], False, ROUTE_DONE, 2, AA)
        # 기체 기호: 화면에서의 진행 방향을 향한 삼각형
        nx, ny = self._at(frac + 0.01)
        px, py = self._at(frac - 0.01)
        a = math.atan2(ny - py, nx - px)
        size = max(7, self.w // 50)
        tri = np.array([(x + size * math.cos(a + d), y + size * math.sin(a + d)) for d in (0.0, 2.5, -2.5)], np.int32)
        cv2.fillPoly(img, [tri], (0, 0, 0), AA)
        cv2.polylines(img, [tri], True, WHITE, 2, AA)
        return img


def format_lat_lon(lat, lon) -> str:
    return f"{abs(lat):04.1f}{'N' if lat >= 0 else 'S'} {abs(lon):05.1f}{'E' if lon >= 0 else 'W'}"


def format_hm(seconds) -> str:
    minutes = int(round(seconds / 60.0))
    return f"{minutes // 60}h {minutes % 60:02d}m"
