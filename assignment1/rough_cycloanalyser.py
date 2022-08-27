__author__ = "Gurbaaz Singh Nandra"
__rollno__ = 190349
__filename__ = "cycloanalyzer.ipynb"
__description__ = """python notebook file cycloneanalyzer.ipynb for doing all the analysis including 
reading, manipulating images, intermittent results and final answers justifying the analysis"""

import math
from typing import Tuple, List
import numpy as np
from PIL import Image

imgs = [np.array(Image.open(f"{i+1}.jpg")) for i in range(3)]
timestamps = [41, 55]


def is_color(pixel: Tuple, color: str) -> bool:
    r, g, b = pixel

    if color == "R":
        # return (r != 0) and (g == 0) and (b == 0)
        return (r > 240) and (g < 50) and (b < 50)

    if color == "G":
        return (g > 240) and (r < 50) and (b < 50)

    if color == "B":
        return (b > 230) and (r < 50) and (g < 50)


def find_color_dots(A: np.ndarray, color: str) -> List[Tuple]:
    assert color == "R" or color == "G" or color == "B"

    color_dots = []

    m, n, _ = A.shape

    for i in range(m):
        for j in range(n):
            pixel = A[(i, j)]

            if is_color(pixel, color):
                color_dots.append((i, j))

    return color_dots


def average_sum_list(A: List[Tuple]) -> List:
    result = []

    for j in range(len(A[0])):

        temp = 0

        for i in range(len(A)):
            temp += A[i][j]

        result.append(temp // len(A))

    return result


def average_dots(A: List[Tuple]) -> List[Tuple]:
    threshold = 10

    if len(A) == 0:
        return []

    avg_dots = []

    temp_dots = []
    temp_dots.append(A[0])

    i = 1

    while i < len(A):

        x, y = temp_dots[0]

        if abs(x - A[i][0]) < threshold and abs(y - A[i][1]) < threshold:
            temp_dots.append(A[i])

        else:
            avg_dots.append(average_sum_list(temp_dots))
            temp_dots = []

            if i != len(A) - 1:
                temp_dots.append(A[i + 1])
                i += 1

        i += 1

    if len(temp_dots) != 0:
        avg_dots.append(average_sum_list(temp_dots))

    return avg_dots


def velocity(a: Tuple, b: Tuple, time: int) -> float:
    print(a, b, time)
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) / time


def circumcenter(a: Tuple, b: Tuple, c: Tuple) -> Tuple:
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    cx = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    cy = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[1] - a[1])
    ) / d

    return cx, cy


def define_circle(p1: Tuple, p2: Tuple, p3: Tuple) -> Tuple[Tuple, float]:
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0]) ** 2 + (cy - p1[1]) ** 2)
    return ((cx, cy), radius)


def draw_circle(A: np.ndarray, center: Tuple, radius: int, color: Tuple) -> np.ndarray:
    x0, y0 = center

    for i in range(x0 - radius, x0 + radius):
        for j in range(y0 - radius, y0 + radius):
            if (i - x0) ** 2 + (j - y0) ** 2 <= radius**2:
                A[(i, j)] = color

    return A


def find_tangent(c: Tuple, p: Tuple) -> Tuple:
    slope = (c[1] - p[1]) / (c[0] - p[0])
    m = -1 / slope
    c = p[1] - m * p[0]
    return (m, c)


def find_line(c: Tuple, p: Tuple) -> Tuple:
    m = (c[1] - p[1]) / (c[0] - p[0])
    c = p[1] - m * p[0]
    return (m, c)


def distance(a: Tuple, b: Tuple) -> float:
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def satisfy_circle(c: Tuple, p: Tuple, r: float) -> bool:
    d = np.sqrt((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
    # print(f"{d=}  {r=} {c=} {p=}")
    return abs(d - r) <= 1


def satisfy_line(theta: Tuple, p: Tuple) -> bool:
    return abs((theta[1] * p[0] ** 0 + theta[0] * p[0] ** 1) - p[1]) <= 1


red_dots = []

for img in imgs:
    print("=" * 20)
    red_dots.append(average_dots(find_color_dots(img, "R")))

print(red_dots)

for i in range(1, len(red_dots)):
    print("velocity")
    for j in range(len(red_dots[0])):
        print(velocity(red_dots[i][j], red_dots[i - 1][j], timestamps[i - 1]))

ccs = []
rss = []

for i in range(len(red_dots[0])):
    res = define_circle(red_dots[0][i], red_dots[1][i], red_dots[2][i])
    ccs.append(res[0])
    rss.append(res[1])
    print("cc: ", ccs[-1])

green = average_dots(find_color_dots(imgs[0], "G"))
blue = average_dots(find_color_dots(imgs[0], "B"))
print("Green: ", green)
print("Blue: ", blue)

ni = np.copy(imgs[0])

for i in red_dots:
    for j in i:
        print(tuple(j))
        ni = draw_circle(ni, tuple(j), 3, (255, 0, 0))

print(f"{rss=}")

for cc, rs in zip(ccs, rss):
    for i in range(ni.shape[0]):
        for j in range(ni.shape[1]):
            if satisfy_circle(cc, (i, j), rs):
                ni[(i, j)] = (255, 255, 0)

poi = (0, 0)


thetas = []
tangents = []

for i in range(2):
    X = []
    y = []

    for ix, rd in enumerate(red_dots):
        if i == 0 and ix == 0:
            continue
        X.append(rd[i][0])
        y.append(rd[i][1])

    print(f"{X=} {y=}")

    thetas.append(np.polyfit(X, y, 1))

tangents.append(find_tangent(ccs[0], red_dots[2][0]))
tangents.append(find_tangent(ccs[1], red_dots[2][1]))

for tangent in tangents:
    for i in range(ni.shape[0]):
        for j in range(ni.shape[1]):
            if satisfy_line(tangent, (i, j)):
                ni[(i, j)] = (255, 255, 255)

m1, c1, m2, c2 = *tangents[0], *tangents[1]

poi = int((c2 - c1) / (m1 - m2)), int((m1 * c2 - m2 * c1) / (m1 - m2))

ni = draw_circle(ni, poi, 4, (0, 255, 255))

m1, c1, m2, c2 = *find_line(red_dots[2][0], red_dots[1][0]), *find_line(
    red_dots[2][1], red_dots[1][1]
)

poi = int((c2 - c1) / (m1 - m2)), int((m1 * c2 - m2 * c1) / (m1 - m2))

for m_c in [(m1, c1), (m2, c2)]:
    for i in range(ni.shape[0]):
        for j in range(ni.shape[1]):
            if satisfy_line(m_c, (i, j)):
                ni[(i, j)] = (165, 42, 42)

ni = draw_circle(ni, poi, 4, (0, 255, 255))

Image.fromarray(ni).save("allinone.jpg")

print("distance between reunion and mauritius = 250km := ", distance(*blue))
print(
    "apojuncture will be from the capital city = ",
    distance(green[1], poi) / distance(*blue) * 250,
    " km",
)

t1 = distance(poi, red_dots[2][0]) / velocity(
    red_dots[2][0], red_dots[1][0], timestamps[1]
)

t2 = distance(poi, red_dots[2][1]) / velocity(
    red_dots[2][1], red_dots[1][1], timestamps[1]
)

print(f"{t1=}")
print(f"{t2=}")
