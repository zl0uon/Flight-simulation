"""공항 주변 도시 풍경 (도시 건설 게임처럼 구역이 나뉜 도시). 표시 전용이다.

구성
    - 고속도로 2개(남북·동서)와 나들목, 공항 진입로, 구역을 잇는 간선도로
    - 도심(고층 빌딩) / 상업·중층 지구 / 주택가(동·남서) / 공업 지대 / 공원·호수 / 강과 다리 / 농경지
    - 활주로 주변과 접근 경로 아래는 비워 둔다 (잔디·초지)

만드는 것
    DAY       위에서 본 낮 지면 그림 (도로, 지붕, 나무, 물, 건물 그림자)
    EMISSION  밤에 빛나는 것만 그린 그림 (가로등, 창문 불빛). 밤 = 어둡게 한 DAY + EMISSION
    BOXES     입체로 세울 건물 상자들 (도심 빌딩, 아파트, 공장 등). 주택은 그림으로만 둔다

좌표: x 동쪽, y 북쪽 (m). 그림의 픽셀 (tx, ty) = ((x - X0) / MPP, (Y1 - y) / MPP).
무작위 요소는 고정된 시드로 만들어 실행할 때마다 같은 도시가 나온다.
비행 모델·판정과는 무관하다 (건물과 충돌하지 않는다).
"""

import math

import cv2
import numpy as np

MPP = 1.6                       # 그림 1픽셀 = 1.6 m
SIZE = 4000                     # 그림 한 변 (픽셀) -> 6.4 km
X0, Y1 = -3000.0, 4000.0        # 그림 왼쪽 위 모서리의 월드 좌표
X1, Y0 = X0 + SIZE * MPP, Y1 - SIZE * MPP

AA = cv2.LINE_AA

# 색 (BGR)
GRASS = (58, 104, 66)
MEADOW = (70, 122, 84)
PARK = (64, 138, 78)
LAWN = (86, 150, 104)
WATER = (196, 148, 72)
SAND = (150, 196, 208)
ROAD = (74, 72, 72)
HIGHWAY = (58, 56, 58)
PAVED = (150, 150, 150)
PLAZA = (176, 178, 182)
LOT = (96, 96, 100)
INDUSTRIAL_GROUND = (128, 140, 150)
TREE = (40, 96, 44)
TREE_LIGHT = (58, 124, 62)
SHADOW_OFFSET = (0.35, -0.75)   # 건물 높이 1 m당 그림자가 뻗는 거리 (동, 북)

HOUSE_ROOFS = [(70, 80, 170), (60, 70, 140), (90, 100, 110), (110, 90, 70), (80, 110, 160),
               (120, 120, 125), (70, 90, 120)]
TOWER_WALLS = [(205, 180, 140), (190, 160, 120), (215, 205, 190), (170, 170, 160), (200, 190, 150),
               (160, 140, 110), (220, 215, 205), (185, 200, 175)]
MIDRISE_WALLS = [(190, 200, 215), (170, 185, 205), (205, 205, 200), (160, 175, 195), (185, 180, 170),
                 (150, 165, 200)]
FACTORY_ROOFS = [(190, 190, 185), (170, 160, 150), (150, 140, 165), (120, 130, 170), (200, 195, 180)]
FIELDS = [(96, 176, 190), (84, 150, 96), (70, 128, 84), (110, 190, 170), (92, 138, 152),
          (76, 164, 120), (120, 170, 150)]

# 건물을 두지 않는 곳: 공항 부지와 접근 경로 아래 (x1, y1, x2, y2)
KEEP_CLEAR = [(-420.0, -450.0, 180.0, 1450.0), (-170.0, -1750.0, 170.0, -450.0)]


def px(x, y):
    return int(round((x - X0) / MPP)), int(round((Y1 - y) / MPP))


def length_px(meters):
    return max(1, int(round(meters / MPP)))


def _clear(x1, y1, x2, y2):
    """이 사각형이 비워 둘 곳과 겹치지 않는지."""
    return all(x2 < a or x1 > c or y2 < b or y1 > d for a, b, c, d in KEEP_CLEAR)


class Builder:
    def __init__(self, base_tile, seed=21):
        self.rng = np.random.default_rng(seed)
        # 바탕: 3D 화면의 잔디 질감과 이어지도록 같은 무늬(1 km 반복)를 깐다
        period = int(round(1000.0 / MPP))
        tile = cv2.resize(base_tile, (period, period), interpolation=cv2.INTER_AREA)
        reps = SIZE // period + 1
        self.img = np.tile(tile, (reps, reps, 1))[:SIZE, :SIZE].copy()
        self.emission = np.zeros((SIZE, SIZE, 3), np.uint8)
        self.shadow = np.zeros((SIZE, SIZE), np.uint8)
        self.roofs = []     # 그림자 다음에 그릴 지붕: (x1, y1, x2, y2, 색)
        self.boxes = []     # 입체 건물: (x1, y1, x2, y2, 바닥 높이, 꼭대기 높이, 벽 색, 지붕 색, 불 켜진 정도)
        self.lamps = []     # 가로등 위치: (x, y, 색)

    # ------------------------------------------------------------ 그리기 도구

    def rect(self, x1, y1, x2, y2, color, img=None):
        cv2.rectangle(self.img if img is None else img, px(x1, y2), px(x2, y1), color, -1)

    def tint(self, x1, y1, x2, y2, color, amount=0.75):
        """바탕 무늬를 조금 남기며 색을 입힌다 (구역 바닥)."""
        (ax, ay), (bx, by) = px(x1, y2), px(x2, y1)
        ax, ay, bx, by = max(ax, 0), max(ay, 0), min(bx, SIZE), min(by, SIZE)
        if ax < bx and ay < by:
            roi = self.img[ay:by, ax:bx]
            roi[:] = (roi * (1 - amount) + np.array(color, np.float32) * amount).astype(np.uint8)

    def road(self, pts, width, color=ROAD, line=None, lamp_color=(120, 190, 255), lamp_step=45.0):
        """꺾은선 도로. line = 중앙선 색(없으면 생략). 가로등 위치도 기록한다."""
        p = np.array([px(x, y) for x, y in pts], np.int32)
        cv2.polylines(self.img, [p], False, color, length_px(width), AA)
        if line is not None:
            cv2.polylines(self.img, [p], False, line, 1, AA)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            n = max(int(math.hypot(x2 - x1, y2 - y1) / lamp_step), 1)
            for k in range(n + 1):
                self.lamps.append((x1 + (x2 - x1) * k / n, y1 + (y2 - y1) * k / n, lamp_color))

    def tree(self, x, y, r=4.0):
        c = px(x, y)
        cv2.circle(self.img, (c[0] + 1, c[1] + 1), length_px(r), (30, 62, 34), -1, AA)
        cv2.circle(self.img, c, length_px(r), TREE, -1, AA)
        cv2.circle(self.img, (c[0] - 1, c[1] - 1), max(length_px(r) - 2, 1), TREE_LIGHT, -1, AA)

    def trees(self, x1, y1, x2, y2, count, r=(3.0, 6.0)):
        for _ in range(count):
            self.tree(self.rng.uniform(x1, x2), self.rng.uniform(y1, y2), self.rng.uniform(*r))

    def box(self, x1, y1, x2, y2, height, wall, roof=None, base=0.0, lit=None):
        """입체 건물 하나. 지면 그림에는 지붕 색 발자국과 그림자를 남긴다."""
        roof = roof or tuple(int(c * 0.72) for c in wall)
        lit = float(self.rng.uniform(0.2, 1.0)) if lit is None else lit
        self.boxes.append((x1, y1, x2, y2, base, height, wall, roof, lit))
        if base == 0.0:
            dx, dy = SHADOW_OFFSET[0] * height, SHADOW_OFFSET[1] * height
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            hull = cv2.convexHull(np.array([px(x, y) for x, y in corners]
                                           + [px(x + dx, y + dy) for x, y in corners], np.int32))
            cv2.fillConvexPoly(self.shadow, hull, 255)
            self.roofs.append((x1, y1, x2, y2, roof))

    def house(self, x, y, w, d, lit_chance=0.4):
        """주택: 입체로 세우지 않고 그림으로만 (작은 그림자 + 지붕 + 용마루 선)."""
        roof = HOUSE_ROOFS[int(self.rng.integers(len(HOUSE_ROOFS)))]
        self.rect(x + 2.0, y - 3.0, x + w + 2.0, y + d - 3.0, (40, 66, 44))
        self.rect(x, y, x + w, y + d, roof)
        (ax, ay), (bx, by) = px(x, y + d), px(x + w, y)
        light = tuple(min(int(c * 1.25), 255) for c in roof)
        if w >= d:
            cv2.line(self.img, (ax, (ay + by) // 2), (bx, (ay + by) // 2), light, 1)
        else:
            cv2.line(self.img, ((ax + bx) // 2, ay), ((ax + bx) // 2, by), light, 1)
        if self.rng.random() < lit_chance:   # 밤에 불 켜진 집
            cv2.circle(self.emission, px(x + w / 2, y + d / 2 - 2), 1, (110, 185, 250), -1)

    # ------------------------------------------------------------ 자연

    def river(self):
        knots = [(-3000, -1760), (-2300, -1900), (-1400, -1790), (-500, -1990), (500, -1880),
                 (1500, -2050), (2500, -1900), (3400, -2010)]
        xs = np.arange(X0, X1 + 1, 20.0)
        ys = np.zeros_like(xs)
        for (xa, ya), (xb, yb) in zip(knots, knots[1:]):   # 부드럽게 이어지는 굴곡
            m = (xs >= xa) & (xs <= xb)
            t = (xs[m] - xa) / (xb - xa)
            ys[m] = ya + (yb - ya) * (t * t * (3 - 2 * t))
        self.river_line = list(zip(xs.tolist(), ys.tolist()))
        p = np.array([px(x, y) for x, y in self.river_line], np.int32)
        cv2.polylines(self.img, [p], False, SAND, length_px(150), AA)
        cv2.polylines(self.img, [p], False, WATER, length_px(122), AA)
        cv2.polylines(self.img, [p], False, (206, 162, 88), length_px(60), AA)   # 가운데는 조금 밝게

    def river_y(self, x):
        xs = [p[0] for p in self.river_line]
        return float(np.interp(x, xs, [p[1] for p in self.river_line]))

    def lake_park(self):
        self.tint(1550, -1050, 2600, -330, PARK, 0.85)
        cv2.ellipse(self.img, px(2060, -680), (length_px(318), length_px(214)), 0, 0, 360, SAND, -1, AA)
        cv2.ellipse(self.img, px(2060, -680), (length_px(300), length_px(198)), 0, 0, 360, WATER, -1, AA)
        cv2.ellipse(self.img, px(2030, -660), (length_px(190), length_px(110)), 0, 0, 360,
                    (206, 162, 88), -1, AA)
        ring = [(2060 + 360 * math.cos(a), -680 + 250 * math.sin(a))
                for a in np.linspace(0, 2 * math.pi, 49)]
        self.road(ring, 5, (150, 180, 190), lamp_step=90.0)   # 호숫가 산책로
        for _ in range(260):
            x, y = self.rng.uniform(1570, 2580), self.rng.uniform(-1030, -350)
            if ((x - 2060) / 335) ** 2 + ((y + 680) / 232) ** 2 > 1.0:
                self.tree(x, y, self.rng.uniform(3, 7))

    def farmland(self, x1, y1, x2, y2, size=(180, 320)):
        """줄무늬가 있는 밭들의 조각보."""
        y = y1
        while y < y2 - 30:
            h = min(self.rng.uniform(*size) * 0.6, y2 - y)
            x = x1
            while x < x2 - 30:
                w = min(self.rng.uniform(*size), x2 - x)
                if _clear(x, y, x + w, y + h) or True:
                    color = FIELDS[int(self.rng.integers(len(FIELDS)))]
                    self.rect(x + 3, y + 3, x + w - 3, y + h - 3, color)
                    dark = tuple(int(c * 0.9) for c in color)
                    if self.rng.random() < 0.5:   # 이랑 방향
                        for yy in np.arange(y + 8, y + h - 5, 9.0):
                            cv2.line(self.img, px(x + 4, yy), px(x + w - 4, yy), dark, 1)
                    else:
                        for xx in np.arange(x + 8, x + w - 5, 9.0):
                            cv2.line(self.img, px(xx, y + 4), px(xx, y + h - 4), dark, 1)
                x += w
            y += h

    def forest(self, cx, cy, radius, count):
        for _ in range(count):
            a, r = self.rng.uniform(0, 2 * math.pi), radius * math.sqrt(self.rng.random())
            self.tree(cx + r * math.cos(a), cy + 0.7 * r * math.sin(a), self.rng.uniform(4, 8))

    # ------------------------------------------------------------ 구역

    def blocks(self, x1, y1, x2, y2, bw, bh, street):
        """격자 도로를 깔고 블록(안쪽 사각형)들을 돌려준다."""
        xs = np.arange(x1, x2 + 1, bw + street)
        ys = np.arange(y1, y2 + 1, bh + street)
        for x in xs:
            self.road([(x, y1), (x, min(ys[-1], y2))], street, lamp_step=60.0)
        for y in ys:
            self.road([(x1, y), (min(xs[-1], x2), y)], street, lamp_step=60.0)
        half = street / 2
        return [(x + half, y + half, x + bw + half, y + bh + half)
                for x in xs[:-1] for y in ys[:-1]]

    def downtown(self, x1, y1, x2, y2):
        self.tint(x1, y1, x2, y2, PAVED, 0.8)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for bx1, by1, bx2, by2 in self.blocks(x1, y1, x2, y2, 150, 150, 24):
            core = math.exp(-((bx1 + 75 - cx) ** 2 + (by1 + 75 - cy) ** 2) / (2 * 520.0 ** 2))
            roll = self.rng.random()
            if roll < 0.07:   # 도심 공원
                self.rect(bx1 + 6, by1 + 6, bx2 - 6, by2 - 6, PARK)
                self.trees(bx1 + 12, by1 + 12, bx2 - 12, by2 - 12, 22)
                continue
            if roll < 0.13:   # 광장과 주차장
                self.rect(bx1 + 6, by1 + 6, bx2 - 6, by2 - 6, PLAZA if roll < 0.1 else LOT)
                continue
            self.rect(bx1 + 4, by1 + 4, bx2 - 4, by2 - 4, PLAZA)
            for qx in (0, 1):     # 블록을 네 칸으로 나눠 건물을 세운다
                for qy in (0, 1):
                    if self.rng.random() < 0.18:
                        self.tree(bx1 + 40 + qx * 70, by1 + 40 + qy * 70, 6)
                        continue
                    w, d = self.rng.uniform(40, 58), self.rng.uniform(40, 58)
                    ox = bx1 + 10 + qx * 70 + self.rng.uniform(0, 60 - w)
                    oy = by1 + 10 + qy * 70 + self.rng.uniform(0, 60 - d)
                    tall = core > 0.45 and self.rng.random() < core
                    if tall:
                        h = (45 + 110 * core) * self.rng.uniform(0.6, 1.1)
                        wall = TOWER_WALLS[int(self.rng.integers(len(TOWER_WALLS)))]
                        self.box(ox, oy, ox + w, oy + d, h, wall)
                        if h > 75:   # 위로 갈수록 좁아지는 꼭대기 층
                            lit = self.boxes[-1][8]
                            self.box(ox + w * 0.2, oy + d * 0.2, ox + w * 0.8, oy + d * 0.8,
                                     h * self.rng.uniform(1.15, 1.3), wall, base=h, lit=lit)
                    else:
                        h = self.rng.uniform(14, 38) * (0.7 + 0.8 * core)
                        self.box(ox, oy, ox + w, oy + d, h,
                                 MIDRISE_WALLS[int(self.rng.integers(len(MIDRISE_WALLS)))])

    def midrise(self, x1, y1, x2, y2):
        """상업·중층 지구: 건물과 주차장이 섞여 있다."""
        self.tint(x1, y1, x2, y2, PAVED, 0.7)
        for bx1, by1, bx2, by2 in self.blocks(x1, y1, x2, y2, 190, 150, 18):
            if not _clear(bx1, by1, bx2, by2):
                continue
            roll = self.rng.random()
            if roll < 0.12:
                self.rect(bx1 + 6, by1 + 6, bx2 - 6, by2 - 6, PARK)
                self.trees(bx1 + 12, by1 + 12, bx2 - 12, by2 - 12, 26)
                continue
            self.rect(bx1 + 5, by1 + 5, bx2 - 5, by2 - 5, LOT)
            for k in range(8):   # 주차 구획선
                xx = bx1 + 14 + k * 22
                cv2.line(self.img, px(xx, by1 + 8), px(xx, by1 + 40), (160, 160, 160), 1)
            n = 1 if roll < 0.45 else 2
            for k in range(n):
                w, d = self.rng.uniform(60, 90) / n + 30, self.rng.uniform(50, 80)
                ox = bx1 + 12 + k * 92 + self.rng.uniform(0, 10)
                oy = by2 - 12 - d
                self.box(ox, oy, min(ox + w, bx2 - 8), oy + d, self.rng.uniform(10, 30),
                         MIDRISE_WALLS[int(self.rng.integers(len(MIDRISE_WALLS)))])

    def residential(self, x1, y1, x2, y2, apartments=0.06):
        """주택가: 잔디 바닥, 집과 나무. 가끔 아파트 블록(입체)."""
        self.tint(x1, y1, x2, y2, LAWN, 0.8)
        for bx1, by1, bx2, by2 in self.blocks(x1, y1, x2, y2, 130, 84, 12):
            if not _clear(bx1, by1, bx2, by2):
                continue
            roll = self.rng.random()
            if roll < 0.05:   # 동네 공원
                self.rect(bx1 + 3, by1 + 3, bx2 - 3, by2 - 3, PARK)
                self.trees(bx1 + 8, by1 + 8, bx2 - 8, by2 - 8, 16)
                continue
            if roll < 0.05 + apartments:   # 아파트 두 동
                self.rect(bx1 + 3, by1 + 3, bx2 - 3, by2 - 3, PLAZA)
                for k in (0, 1):
                    ox = bx1 + 10 + k * 62
                    self.box(ox, by1 + 14, ox + 50, by2 - 14, self.rng.uniform(18, 34),
                             MIDRISE_WALLS[int(self.rng.integers(len(MIDRISE_WALLS)))])
                continue
            for row, y in enumerate((by1 + 6, by2 - 6 - 13)):   # 길을 따라 두 줄
                x = bx1 + 5
                while x < bx2 - 16:
                    w, d = self.rng.uniform(10, 15), self.rng.uniform(9, 13)
                    if self.rng.random() < 0.9:
                        self.house(x, y + (0 if row else 13 - d), w, d)
                    if self.rng.random() < 0.55:
                        self.tree(x + w / 2 + self.rng.uniform(-3, 3),
                                  (by1 + by2) / 2 + self.rng.uniform(-9, 9), self.rng.uniform(2.5, 4.5))
                    x += w + self.rng.uniform(5, 9)

    def industrial(self, x1, y1, x2, y2):
        """공업 지대: 큰 창고, 굴뚝, 저장 탱크."""
        self.tint(x1, y1, x2, y2, INDUSTRIAL_GROUND, 0.8)
        for bx1, by1, bx2, by2 in self.blocks(x1, y1, x2, y2, 280, 200, 20):
            roll = self.rng.random()
            if roll < 0.2:   # 저장 탱크 단지
                for i in range(3):
                    for j in range(2):
                        c = px(bx1 + 55 + i * 85, by1 + 55 + j * 90)
                        cv2.circle(self.img, (c[0] + 3, c[1] + 4), length_px(30), (70, 78, 84), -1, AA)
                        cv2.circle(self.img, c, length_px(30), (215, 215, 210), -1, AA)
                        cv2.circle(self.img, c, length_px(30), (150, 150, 150), 1, AA)
                continue
            roof = FACTORY_ROOFS[int(self.rng.integers(len(FACTORY_ROOFS)))]
            w, d = self.rng.uniform(150, 230), self.rng.uniform(90, 140)
            wall = tuple(int(c * 0.82) for c in roof)
            self.box(bx1 + 20, by1 + 20, bx1 + 20 + w, by1 + 20 + d, self.rng.uniform(9, 16), wall, roof,
                     lit=0.25)
            for xx in np.arange(bx1 + 30, bx1 + 12 + w, 16.0):   # 지붕 골
                cv2.line(self.img, px(xx, by1 + 22), px(xx, by1 + 18 + d), wall, 1)
            if self.rng.random() < 0.45:   # 굴뚝
                self.box(bx2 - 40, by2 - 40, bx2 - 32, by2 - 32, self.rng.uniform(40, 62), (150, 150, 160),
                         lit=0.0)
            self.rect(bx1 + 20, by1 + 30 + d, bx2 - 60, by2 - 14, LOT)

    def stadium(self, cx, cy):
        cv2.ellipse(self.img, px(cx, cy), (length_px(150), length_px(118)), 0, 0, 360, LOT, -1, AA)
        cv2.ellipse(self.img, px(cx, cy), (length_px(118), length_px(92)), 0, 0, 360, (200, 200, 205), -1, AA)
        cv2.ellipse(self.img, px(cx, cy), (length_px(84), length_px(60)), 0, 0, 360, (160, 60, 60), -1, AA)
        self.rect(cx - 52, cy - 34, cx + 52, cy + 34, (70, 160, 80))
        cv2.rectangle(self.img, px(cx - 52, cy + 34), px(cx + 52, cy - 34), (235, 235, 235), 1)
        cv2.line(self.img, px(cx, cy + 34), px(cx, cy - 34), (235, 235, 235), 1)
        for a in np.linspace(0, 2 * math.pi, 17)[:-1]:   # 관중석 상자와 조명
            x, y = cx + 104 * math.cos(a), cy + 80 * math.sin(a)
            self.box(x - 13, y - 13, x + 13, y + 13, 20, (205, 205, 210), (180, 180, 186), lit=0.9)
        cv2.ellipse(self.emission, px(cx, cy), (length_px(84), length_px(60)), 0, 0, 360, (120, 150, 130),
                    -1, AA)

    # ------------------------------------------------------------ 도시 전체

    def build(self):
        self.river()
        self.farmland(X0, Y0, X1, -2130)                       # 강 남쪽
        self.farmland(190, -430, 740, 1440, size=(150, 260))   # 공항 동쪽 (장주 경로 아래)
        self.farmland(-940, -1650, -200, -700, size=(160, 260))
        self.tint(-170, -1750, 170, -450, MEADOW, 0.6)         # 접근 경로 아래 초지
        self.forest(-560, 250, 110, 70)
        self.forest(430, -1300, 190, 120)
        self.forest(-2700, 2130, 200, 110)
        self.forest(3050, 2150, 220, 120)
        self.lake_park()

        self.residential(-2990, -1560, -990, -280)             # 남서 주택가
        self.residential(780, -250, 3390, 1950)                # 동쪽 주택가
        self.residential(780, -1540, 1500, -420, apartments=0.15)
        self.residential(-2990, 2480, -1400, 3980)             # 북서 주택가
        self.midrise(-1300, 2480, 600, 3900)                   # 북쪽 상업 지구
        self.industrial(900, 2480, 3380, 3900)                 # 북동 공업 지대
        self.downtown(-2690, -130, -950, 1960)                 # 도심
        self.stadium(-1500, -820)

        # 구역을 잇는 간선도로
        yellow = (90, 200, 235)
        self.road([(-950, 1650), (780, 1650)], 16, line=yellow)
        self.road([(-990, -650), (780, -650)], 16, line=yellow)
        self.road([(780, -1540), (780, 2250)], 18, line=yellow)
        self.road([(-2700, 1960), (-2700, 2250)], 16, line=yellow)
        self.road([(-1825, 1960), (-1825, 2480)], 18, line=yellow)
        self.road([(-2000, -1560), (-2000, -280)], 16, line=yellow)
        self.road([(1500, -1540), (1500, self.river_y(1500) - 320)], 16, line=yellow)   # 강 건너는 길
        self.road([(-700, 500), (-300, 500)], 18, line=yellow)                          # 공항 진입로
        self.road([(-950, -130), (-950, 1960)], 18, line=yellow)

        # 고속도로 (남북 x = -700, 동서 y = 2250)와 나들목
        white = (235, 235, 235)
        lamp = (170, 215, 255)
        for a, b in (((-700, Y0), (-700, Y1)), ((X0, 2250), (X1, 2250))):
            self.road([a, b], 40, (120, 124, 128))                    # 갓길과 방음벽
            self.road([a, b], 34, HIGHWAY, lamp_color=lamp, lamp_step=40.0)
            self.road([a, b], 3, (120, 150, 120))                     # 중앙 분리대
            for off in (-9.5, 9.5):                                   # 차선
                shift = (off, 0) if a[0] == b[0] else (0, off)
                p = [px(a[0] + shift[0], a[1] + shift[1]), px(b[0] + shift[0], b[1] + shift[1])]
                cv2.line(self.img, p[0], p[1], white, 1, AA)
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):           # 클로버형 나들목
            loop = [(-700 + sx * (70 + 62 * math.cos(a)), 2250 + sy * (70 + 62 * math.sin(a)))
                    for a in np.linspace(0, 2 * math.pi, 33)]
            self.road(loop, 9, HIGHWAY, lamp_step=70.0)
        for bx in (-700, 1500):                                       # 다리 난간
            y = self.river_y(bx)
            for off in (-18, 18) if bx == -700 else (-9, 9):
                cv2.line(self.img, px(bx + off, y - 95), px(bx + off, y + 95), (215, 215, 215), 2, AA)

        # 그림자 -> 지붕 순서로 입혀서 그림자가 도로 위에도 지게 한다
        dark = (self.img * 0.6).astype(np.uint8)
        cv2.copyTo(dark, cv2.GaussianBlur(self.shadow, (0, 0), 1.2), self.img)
        for x1, y1, x2, y2, roof in self.roofs:
            self.rect(x1, y1, x2, y2, roof)

        # 밤 불빛: 가로등은 작은 점 + 번진 빛
        for x, y, color in self.lamps:
            if X0 < x < X1 and Y0 < y < Y1:
                cv2.circle(self.emission, px(x, y), 1, color, -1)
        glow = cv2.GaussianBlur(self.emission, (0, 0), 3.0)
        self.emission = cv2.add(self.emission, cv2.convertScaleAbs(glow, alpha=2.2))
        return self


def _levels(img):
    """멀리서 볼 때 쓰는 축소본들 (1/2, 1/4). 축소해 두어야 먼 곳이 자글거리지 않는다."""
    half = cv2.resize(img, (SIZE // 2, SIZE // 2), interpolation=cv2.INTER_AREA)
    return [img, half, cv2.resize(half, (SIZE // 4, SIZE // 4), interpolation=cv2.INTER_AREA)]


def _make_boxes(boxes):
    """입체 건물 상자들을 면 단위 배열로 바꾼다 (그릴 때 한꺼번에 계산하기 위해)."""
    sun = np.array([-0.42, 0.9])   # 햇빛이 오는 수평 방향 (북서)
    sun /= np.linalg.norm(sun)
    verts, normals, day, night, owner = [], [], [], [], []
    centers, heights, beacons = [], [], []
    for i, (x1, y1, x2, y2, z0, z1, wall, roof, lit) in enumerate(boxes):
        c = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2, (z0 + z1) / 2))
        heights.append(z1)
        glow = np.array([70, 150, 215]) * lit   # 창문 불빛 (따뜻한 색)
        faces = [([(x, y, z1) for x, y in c], (0.0, 0.0, 1.0), np.array(roof, float),
                  np.array(roof, float) * 0.16)]
        for k in range(4):
            (ax, ay), (bx, by) = c[k], c[(k + 1) % 4]
            n = np.array([by - ay, ax - bx, 0.0])
            n /= np.linalg.norm(n)
            shade = 0.62 + 0.38 * max(float(n[:2] @ sun), 0.0)
            faces.append(([(ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1)], n,
                          np.array(wall, float) * shade, np.array(wall, float) * 0.14 + glow * 0.45))
        for pts, n, col_day, col_night in faces:
            verts.append(pts)
            normals.append(n)
            day.append(col_day)
            night.append(col_night)
            owner.append(i)
        if z1 > 85 and z0 == 0.0:
            beacons.append(((x1 + x2) / 2, (y1 + y2) / 2, z1 + 2.0))
    return {
        "verts": np.array(verts, float), "normals": np.array(normals, float),
        "day": np.clip(np.array(day), 0, 255), "night": np.clip(np.array(night), 0, 255),
        "owner": np.array(owner), "centers": np.array(centers, float),
        "heights": np.array(heights, float), "beacons": np.array(beacons, float).reshape(-1, 3),
    }


GRASS_MPP = 2.0   # 잔디 질감 1픽셀 = 2 m, 500픽셀(1 km)마다 반복


def _make_grass(seed=7, size=500):
    """이어 붙여도 경계가 보이지 않는 잔디 질감 + 250 m 격자 (크기·속도 감각용)."""
    rng = np.random.default_rng(seed)

    def layer(sigma):
        n = rng.standard_normal((size, size)).astype(np.float32)
        n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), sigma)[size:2 * size, size:2 * size]
        return n / (n.std() + 1e-6)

    n = 0.7 * layer(40) + 0.4 * layer(8) + 0.15 * layer(1.5)
    tile = np.clip(np.array(GRASS, np.float32) + n[..., None] * np.array([6, 14, 10], np.float32),
                   0, 255).astype(np.uint8)
    for k in range(0, size, size // 4):
        tile[k, :] = (66, 112, 74)
        tile[:, k] = (66, 112, 74)
    return tile


def build():
    """도시를 만든다."""
    b = Builder(GRASS_TILE).build()
    return {"day": _levels(b.img), "emission": _levels(b.emission), "boxes": _make_boxes(b.boxes),
            "count": len(b.boxes)}


GRASS_TILE = _make_grass()
CITY = build()   # 프로그램을 시작할 때 한 번 만든다 (1~2초)
