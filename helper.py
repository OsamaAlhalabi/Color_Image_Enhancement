import numpy as np
import math
import utils


class Helper:
    def __init__(self, width, height, m, n):
        self.width = width
        self.height = height
        self.m = m
        self.n = n

    def qxi(self, i, x):
        x0 = 0
        x1 = self.width - 1
        if x == self.width - 1:
            return 0
        cmi = utils.c(self.m, i)
        num = np.power(x - x0, i) * np.power(x1 - x, self.m - i)
        denom = np.power(x1 - x0, self.m) + utils.eps
        return cmi * (num / denom)

    def qyj(self, j, y):
        y0 = 0
        y1 = self.height - 1
        if y == self.height - 1:
            return 0
        cnj = utils.c(self.n, j)
        num = np.power((y - y0), j) * np.power((y1 - y), self.n - j)
        denom = np.power(y1 - y0, self.n) + utils.eps

        return cnj * (num / denom)

    def pij(self, i, j, x, y):  # fuzzy partition of the support D.
        return self.qxi(i, x) * self.qyj(j, y)

