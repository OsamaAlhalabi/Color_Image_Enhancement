import numpy as np
import math

eps = 0.00001


def phi(v):
    return 0.5 * np.log((1 + v) / ((1 - v) + eps))


def norm(v):
    return abs(phi(v))


def real_scalar_multiplication(scalar, value):
    s = (1 + value) ** scalar
    z = (1 - value) ** scalar
    res = (s - z) / (s + z + eps)
    return res


def add(v1, v2):
    return (v1 + v2) / (1 + (v1 * v2) + eps)


def sub(v1, v2):
    return (v1 - v2) / (1 - (v1 * v2) + eps)


def c(m, i):
    num = math.factorial(m)
    denom = ((math.factorial(i) * math.factorial(m - i)) + eps)
    return num / denom


def translate(img, left_min, left_max, right_min, right_max):
    # We consider the unit cube of colors (0,1)^3 'in case of colored images' and we transform it in the cube (-1,
    # 1)^3 by the simplest linear transformation, same transformation in gray scale images but on single vector.
    left_span = left_max - left_min
    right_span = right_max - right_min

    value_scaled = (img - left_min) / float(left_span)

    return right_min + (value_scaled * right_span)
