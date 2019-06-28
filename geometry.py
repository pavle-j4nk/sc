from math import acos
from math import sqrt
from math import pi


def line_length(w, v=(0, 0)):
    return sqrt((v[0] + w[0]) ** 2 + (v[1] + w[1]) ** 2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    cosx = dot_product(v, w) / (line_length(v) * line_length(w))
    rad = acos(cosx)  # in radians
    return rad * 180 / pi  # returns degrees


def angle_clockwise(A, B):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360 - inner
