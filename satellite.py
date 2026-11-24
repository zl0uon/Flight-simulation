"""실제 위성 사진을 지면으로 쓴다 (구글 어스처럼 실제 장소의 모습이 보이게).

- 사진: Sentinel-2 cloudless 2016 (https://s2maps.eu, EOX IT Services GmbH, CC BY 4.0,
  "Contains modified Copernicus Sentinel data 2016"). 해상도는 약 10 m여서 높은 고도에서는 선명하고
  지면 가까이에서는 흐리다. 구글 어스의 사진과 3D 건물은 구글 소유라 쓸 수 없다.
- 활주로: data/runways.csv (OurAirports, 퍼블릭 도메인). 공항마다 가장 긴 활주로의 시작점과 방향.
  시뮬레이션의 활주로(원점에서 북쪽)를 실제 활주로 위에 겹쳐 놓는 데 쓴다.
- 타일은 필요할 때 인터넷에서 받아 cache/tiles 에 저장하고 다음부터는 저장된 것을 쓴다.
  받는 중이거나 인터넷이 안 되면 그 부분은 비어 보인다 (프로그램은 계속 돈다).

cv2와 numpy만 쓴다. 지면에 까는 방법(원근 변환)은 view3d가 맡고, 여기서는 사진과 좌표 변환만 준비한다.
"""

import csv
import json
import math
import queue
import ssl
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless_3857/default/g/{z}/{y}/{x}.jpg"
CREDIT = ("Sentinel-2 cloudless by EOX (CC BY 4.0), contains modified Copernicus Sentinel data 2016 | "
          "buildings (c) OpenStreetMap contributors")
CACHE = Path(__file__).parent / "cache" / "tiles"
DATA = Path(__file__).parent / "data"
TILE = 256
WORKERS = 8
EARTH_CIRCUMFERENCE = 40075016.686   # m (적도)


def _load_runways():
    """공항 코드 -> (활주로 시작점 위도, 경도, 방향(도), 길이 m, 이름)."""
    runways = {}
    path = DATA / "runways.csv"
    if path.exists():
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                runways[row["iata"]] = (float(row["lat"]), float(row["lon"]), float(row["heading"]),
                                        float(row["length_m"]), row["name"])
    return runways


RUNWAYS = _load_runways()


def mercator_px(lat, lon, z):
    """위도·경도 -> 그 확대 단계에서의 전 세계 픽셀 좌표 (웹 메르카토르)."""
    n = TILE * 2 ** z
    lat = max(min(lat, 85.0), -85.0)
    return ((lon + 180.0) / 360.0 * n,
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)


def meters_per_px(lat, z):
    return EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE * 2 ** z)


class TileStore:
    """타일을 디스크에서 읽거나, 없으면 뒤에서 내려받는다."""

    def __init__(self):
        self._jobs = queue.Queue()
        self._asked = set()
        self._lock = threading.Lock()
        self._context = None
        self.failed = 0          # 받지 못한 타일 수 (인터넷이 안 될 때 알려 주려고)
        self.pending = 0
        for _ in range(WORKERS):
            threading.Thread(target=self._work, daemon=True).start()

    @staticmethod
    def path(z, x, y):
        return CACHE / str(z) / str(x) / f"{y}.jpg"

    def request(self, z, x, y, done):
        """타일을 준비해 done(이미지)를 부른다 (작업 스레드에서). 같은 타일을 여러 곳에서 청해도 된다."""
        with self._lock:
            self.pending += 1
        self._jobs.put((z, x, y, done))

    def _fetch(self, z, x, y):
        path = self.path(z, x, y)
        if not path.exists():
            if self._context is None:
                try:
                    import certifi
                    self._context = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    self._context = ssl.create_default_context()
            req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={"User-Agent": "flight-sim/1.0"})
            data = urllib.request.urlopen(req, timeout=20, context=self._context).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".part{threading.get_ident()}")
            tmp.write_bytes(data)
            tmp.replace(path)
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)

    def _work(self):
        while True:
            z, x, y, done = self._jobs.get()
            try:
                n = 2 ** z
                img = self._fetch(z, x % n, y) if 0 <= y < n else None
                if img is not None and img.shape[:2] == (TILE, TILE):
                    done(img)
            except Exception:   # 인터넷이 안 되거나 서버가 응답하지 않으면 그 타일은 비워 둔다
                with self._lock:
                    self.failed += 1
            finally:
                with self._lock:
                    self.pending -= 1


_store = None


def store():
    global _store
    if _store is None:
        _store = TileStore()
    return _store


class Mosaic:
    """한 지점을 중심으로 타일 n x n장을 이어 붙인 사진 (BGRA, 아직 못 받은 곳은 투명)."""

    def __init__(self, lat, lon, z, n):
        self.z, self.n = z, n
        cx, cy = mercator_px(lat, lon, z)
        self.x0 = int(round(cx / TILE - n / 2))   # 왼쪽 위 타일 번호
        self.y0 = int(round(cy / TILE - n / 2))
        self.image = np.zeros((n * TILE, n * TILE, 4), np.uint8)
        self.loaded = 0
        for j in range(n):
            for i in range(n):
                store().request(z, self.x0 + i, self.y0 + j, self._place(i, j))

    def _place(self, i, j):
        def done(tile):
            block = self.image[j * TILE:(j + 1) * TILE, i * TILE:(i + 1) * TILE]
            block[..., :3] = tile
            block[..., 3] = 255
            self.loaded += 1
        return done

    @property
    def ready(self) -> float:
        return self.loaded / (self.n * self.n)

    def covers(self, lat, lon, margin=0.25) -> bool:
        """그 지점이 사진의 가운데 부분(가장자리 margin 제외) 안에 있는지."""
        px, py = mercator_px(lat, lon, self.z)
        u, v = px / TILE - self.x0, py / TILE - self.y0
        lo, hi = self.n * margin, self.n * (1 - margin)
        return lo <= u <= hi and lo <= v <= hi

    def to_world(self, lat, lon, world_xy, rotation_deg, scale):
        """사진 픽셀 -> 시뮬레이션 지면 좌표 (3x3 행렬).

        실제 지점 (lat, lon)이 world_xy에 오고, 실제의 rotation_deg 방향(북에서 시계 방향)이
        시뮬레이션의 북쪽(+y)이 되며, 실제 1 m가 scale 만큼의 길이가 된다.
        """
        cx, cy = mercator_px(lat, lon, self.z)
        k = meters_per_px(lat, self.z) * scale
        t = math.radians(rotation_deg)
        c, s = math.cos(t), math.sin(t)
        # 픽셀 -> 실제 (동, 북) -> 시뮬레이션 (x, y)
        ex, ny = cx - self.x0 * TILE, cy - self.y0 * TILE
        to_en = np.array([[k, 0, -k * ex], [0, -k, k * ny], [0, 0, 1.0]])
        rot = np.array([[c, -s, world_xy[0]], [s, c, world_xy[1]], [0, 0, 1.0]])
        return rot @ to_en


# ---------------------------------------------------------------- 실제 건물 (OpenStreetMap)

OVERPASS_SERVERS = ("https://overpass-api.de/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter")
BUILDING_RADIUS = 4500        # 활주로 시작점에서 이 거리 안의 건물을 받는다 (m)
BUILDING_CACHE = CACHE.parent / "buildings"
MAX_BOXES = 1500              # 세우는 상자 수의 상한 (큰 건물부터)
SMALL = "house|detached|garage|garages|shed|hut|residential|semidetached_house|carport|roof"
LEVEL_HEIGHT = 3.3            # 한 층 높이 (m)
# 건물 종류별 (기본 높이 m, 벽 색 BGR, 지붕 색 BGR). 높이 태그가 없을 때 쓴다
KINDS = {
    "terminal": (24.0, (182, 158, 124), (212, 212, 206)),
    "hangar": (17.0, (176, 180, 184), (198, 202, 206)),
    "control_tower": (70.0, (190, 188, 184), (104, 120, 140)),
    "tower": (45.0, (176, 176, 172), (120, 120, 120)),
    "apartments": (48.0, (206, 210, 214), (150, 152, 156)),
    "hotel": (40.0, (190, 186, 178), (140, 140, 140)),
    "office": (32.0, (184, 170, 150), (150, 150, 154)),
    "commercial": (16.0, (176, 180, 186), (146, 148, 152)),
    "retail": (10.0, (170, 176, 186), (152, 154, 158)),
    "industrial": (11.0, (160, 166, 172), (178, 182, 186)),
    "warehouse": (11.0, (160, 166, 172), (178, 182, 186)),
    "transportation": (16.0, (180, 172, 160), (190, 192, 194)),
    "train_station": (16.0, (180, 172, 160), (190, 192, 194)),
    "parking": (14.0, (150, 150, 150), (120, 120, 122)),
}
DEFAULT_KIND = (9.0, (170, 172, 176), (140, 142, 146))


def _number(text):
    try:
        return float(str(text).split()[0].replace(",", "."))
    except (ValueError, IndexError):
        return None


def _split_footprint(poly):
    """건물 바닥 다각형 (N,2)을 직사각형 조각들로 나눈다 (각 조각 = 네 꼭짓점, 반시계 방향).

    네모난 건물은 한 조각, ㄱ자나 활처럼 휜 터미널은 격자로 잘라 여러 조각으로 만든다.
    """
    pts = poly.astype(np.float32)
    (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
    area = abs(cv2.contourArea(pts))
    if w < 3.0 or h < 3.0 or area < 30.0:
        return []
    if area / (w * h) > 0.72:
        box = cv2.boxPoints(((cx, cy), (w, h), angle))
        return [box[::-1] if cv2.contourArea(box) < 0 else box]
    # 바닥의 긴 방향에 맞춘 격자에 칠해 보고, 줄마다 이어진 칸을 직사각형으로 만든다
    t = math.radians(angle)
    rot = np.array([[math.cos(t), math.sin(t)], [-math.sin(t), math.cos(t)]], np.float32)   # 세계 -> 격자
    local = (pts - (cx, cy)) @ rot.T
    cell = float(np.clip(math.sqrt(area) / 9.0, 6.0, 30.0))
    lo = local.min(0)
    size = np.ceil((local.max(0) - lo) / cell).astype(int) + 1
    mask = np.zeros((size[1] * 4, size[0] * 4), np.uint8)   # 칸마다 4x4로 칠해 절반 넘게 찬 칸만 건물로 본다
    cv2.fillPoly(mask, [np.round((local - lo) / cell * 4).astype(np.int32)], 255)
    filled = cv2.resize(mask, (size[0], size[1]), interpolation=cv2.INTER_AREA) > 110
    pieces, open_runs = [], {}
    for row in range(size[1] + 1):
        runs, col = set(), 0
        while row < size[1] and col < size[0]:
            if filled[row, col]:
                start = col
                while col < size[0] and filled[row, col]:
                    col += 1
                runs.add((start, col))
            else:
                col += 1
        for run in list(open_runs):   # 위 줄과 같은 범위가 이어지면 세로로 합친다
            if run not in runs:
                first = open_runs.pop(run)
                x1, x2 = lo[0] + run[0] * cell, lo[0] + run[1] * cell
                y1, y2 = lo[1] + first * cell, lo[1] + row * cell
                corners = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], np.float32)
                pieces.append(corners @ rot + (cx, cy))
        for run in runs:
            open_runs.setdefault(run, row)
    return pieces


def build_prisms(elements, lat0, lon0, heading_deg):
    """OpenStreetMap 요소들 -> 시뮬레이션 좌표의 건물 상자 목록 (scenery.make_prisms에 넣을 형식)."""
    t = math.radians(heading_deg)
    c, s = math.cos(t), math.sin(t)
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0

    def to_sim(lat, lon):
        e, n = (lon - lon0) * kx, (lat - lat0) * ky
        return e * c - n * s, e * s + n * c

    found = []
    for el in elements:
        tags = el.get("tags", {})
        kind = tags.get("aeroway") if tags.get("aeroway") in KINDS else tags.get("building", "")
        if tags.get("man_made") == "tower":
            kind = "control_tower" if tags.get("tower:type") == "airport_control" else "tower"
        base, wall, roof = KINDS.get(kind, DEFAULT_KIND)
        height = _number(tags.get("height")) or (_number(tags.get("building:levels")) or 0) * LEVEL_HEIGHT or base
        height = float(np.clip(height, 3.0, 400.0))
        if el["type"] == "node":   # 관제탑이 점으로만 있는 경우: 가는 탑을 세운다
            x, y = to_sim(el["lat"], el["lon"])
            found.append((height * 144.0, [np.array([(x - 6, y - 6), (x + 6, y - 6), (x + 6, y + 6), (x - 6, y + 6)])],
                          height, wall, roof))
            continue
        geom = el.get("geometry") or []
        if len(geom) < 4:
            continue
        poly = np.array([to_sim(p["lat"], p["lon"]) for p in geom[:-1]])
        if kind in ("tower", "control_tower"):   # 탑은 바닥을 그대로 한 조각으로
            pieces = _split_footprint(poly)[:1]
        else:
            pieces = _split_footprint(poly)
        if pieces:
            found.append((abs(cv2.contourArea(poly.astype(np.float32))) * height, pieces, height, wall, roof))
    found.sort(key=lambda b: -b[0])   # 부피가 큰 건물부터
    prisms = []
    for i, (_volume, pieces, height, wall, roof) in enumerate(found):
        tint = 0.9 + 0.2 * ((i * 37) % 10) / 10.0   # 건물마다 밝기를 조금씩 다르게
        wall_c = tuple(min(255.0, v * tint) for v in wall)
        for corners in pieces:
            prisms.append(([tuple(map(float, p)) for p in corners], 0.0, height, wall_c, roof,
                           0.5 if height > 12 else 0.2))
        if len(prisms) >= MAX_BOXES:
            break
    return prisms


def fetch_buildings(key, lat, lon, heading_deg):
    """공항 주변 건물 상자 목록. 받은 자료는 cache/buildings 에 저장하고 다음부터는 그것을 쓴다."""
    path = BUILDING_CACHE / f"{key}.json"
    if path.exists():
        elements = json.loads(path.read_text(encoding="utf-8"))
    else:
        around = f"(around:{BUILDING_RADIUS},{lat},{lon})"
        query = ("[out:json][timeout:90];("
                 f'way["building"]["building"!~"^({SMALL})$"]{around};'
                 f'way["aeroway"~"terminal|hangar"]{around};'
                 f'node["aeroway"="control_tower"]{around};'
                 f'way["man_made"="tower"]{around};'
                 ");out tags geom;")
        import urllib.parse
        try:
            import certifi
            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        elements, error = None, None
        for attempt in range(4):   # 공개 서버라 바쁠 때가 있다: 잠깐 쉬고, 다른 서버로도 다시 청한다
            server = OVERPASS_SERVERS[attempt % len(OVERPASS_SERVERS)]
            try:
                req = urllib.request.Request(server, data=urllib.parse.urlencode({"data": query}).encode(),
                                             headers={"User-Agent": "flight-sim/1.0"})
                elements = json.loads(urllib.request.urlopen(req, timeout=120, context=context).read())["elements"]
                break
            except Exception as ex:
                error = ex
                time.sleep(4.0 * (attempt + 1))
        if elements is None:
            raise error
        # 필요한 것만 남겨 저장한다
        elements = [{"type": e["type"], "tags": e.get("tags", {}), "lat": e.get("lat"), "lon": e.get("lon"),
                     "geometry": e.get("geometry")} for e in elements]
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(elements), encoding="utf-8")
        tmp.replace(path)
    return build_prisms(elements, lat, lon, heading_deg)


class Scenery:
    """지금 화면에 깔 위성 사진 층들 (먼 곳용 넓은 사진 + 가까운 곳용 자세한 사진)."""

    # (확대 단계, 가로세로 타일 수): 공항 주변과 순항 중
    LOCAL_LAYERS = ((10, 6), (13, 6), (14, 8))
    CRUISE_LAYERS = ((7, 4), (10, 6))

    def __init__(self):
        self.mosaics = []
        self.frame = None        # (위도, 경도, 지면 좌표, 회전, 축척): 사진을 놓는 기준
        self.buildings = None    # 공항 주변의 실제 건물 (view3d가 그릴 면 배열). 받는 중이면 None
        self._airport = None
        self._next = None        # 미리 받아 두는 다음 지점의 사진들

    def _make(self, lat, lon, layers):
        return [Mosaic(lat, lon, z, n) for z, n in layers]

    def set_airport(self, lat, lon, heading_deg, key=None):
        """공항 주변: 활주로 시작점이 원점, 활주로 방향이 시뮬레이션의 북쪽. key를 주면 건물도 세운다."""
        self.mosaics = self._make(lat, lon, self.LOCAL_LAYERS)
        self.frame = (lat, lon, (0.0, 0.0), heading_deg, 1.0)
        self._next = None
        self.buildings = None
        self._airport = key
        if key is not None:
            threading.Thread(target=self._load_buildings, args=(key, lat, lon, heading_deg), daemon=True).start()

    def _load_buildings(self, key, lat, lon, heading_deg):
        import scenery   # 면 배열을 만드는 함수 (모형 도시와 같은 형식)
        try:
            faces = scenery.make_prisms(fetch_buildings(key, lat, lon, heading_deg))
        except Exception as ex:   # 인터넷이 안 되거나 서버가 바쁘면 건물 없이 간다
            print(f"[위성 지면] {key} 공항의 건물 자료를 받지 못했습니다 ({type(ex).__name__}). 건물 없이 진행합니다.")
            return
        if self._airport == key:   # 받는 사이에 다른 공항으로 바뀌지 않았을 때만
            self.buildings = faces

    def prefetch(self, lat, lon):
        """곧 갈 지점의 사진을 미리 받기 시작한다 (순항 중 장면이 바뀌기 전에)."""
        if self._next is None or not all(m.covers(lat, lon, 0.35) for m in self._next[2]):
            self._next = (lat, lon, self._make(lat, lon, self.CRUISE_LAYERS))

    def set_cruise(self, lat, lon, world_xy, rotation_deg, scale, recenter=False):
        """순항 중: 실제 지점 (lat, lon)이 지금 world_xy에 있다. recenter면 그 지점 사진으로 바꾼다."""
        if recenter or not self.mosaics:
            if self._next is not None and all(m.covers(lat, lon, 0.2) for m in self._next[2]):
                self.mosaics = self._next[2]
            else:
                self.mosaics = self._make(lat, lon, self.CRUISE_LAYERS)
            self._next = None
        self.frame = (lat, lon, world_xy, rotation_deg, scale)

    def layers(self):
        """그릴 층들: (사진, 사진 픽셀 -> 지면 좌표 행렬). 먼 곳용부터."""
        if self.frame is None:
            return []
        lat, lon, world_xy, rotation, scale = self.frame
        return [(m.image, m.to_world(lat, lon, world_xy, rotation, scale)) for m in self.mosaics if m.loaded]

    @property
    def ready(self) -> float:
        return min((m.ready for m in self.mosaics), default=0.0)
