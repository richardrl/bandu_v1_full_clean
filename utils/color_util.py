# Based on https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors

import colorsys
import itertools
from fractions import Fraction
import math
import numpy as np


def zenos_dichotomy():
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1, 2 ** k)


def getfracs():
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield 0
    for k in zenos_dichotomy():
        i = k.denominator  # [1,2,4,8,16,...]
        for j in range(1, i, 2):
            yield Fraction(j, i)


bias = lambda x: (math.sqrt(x / 3) / Fraction(2, 3) + Fraction(1, 3)) / Fraction(6, 5)


def genhsv(h):
    for s in [Fraction(6, 10)]:  # optionally use range
        for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
            yield (h, s, v)  # use bias for v here if you use range


genrgb = lambda x: colorsys.hsv_to_rgb(*x)

flatten = itertools.chain.from_iterable

gethsvs = lambda: flatten(map(genhsv,getfracs()))

getrgbs = lambda: map(genrgb, gethsvs())


def genhtml(x):
    uint8tuple = map(lambda y: y, x)
    uint8tuple = map(lambda x: round(float(x), 4), uint8tuple)
    return "{} {} {}".format(*uint8tuple)

gethtmlcolors = lambda: map(genhtml, getrgbs())

# def get_colors(n):
#     return list(itertools.islice(gethtmlcolors(), n))

def get_colors(n):
    base_colors = np.ones((30, 3))

    base_colors[15] = (230, 25, 75)
    base_colors[14] = (70, 240, 240)
    base_colors[7] = (60, 180, 75)
    base_colors[9] = (170, 110, 40)
    base_colors[16] = (60, 180, 75)
    # base_colors[5] = (0, 0, 128)
    base_colors[5] = (170, 110, 40)
    base_colors[4] = (128, 0, 0)
    base_colors[0] = (240, 50, 230)
    base_colors[11] = (0, 128, 128)
    base_colors[6] = (255, 225, 25)
    base_colors[12] = (245, 130, 48)
    base_colors[19] = (220, 190, 255)

def gen_colors(n):
    """

    :param n:
    :return: Returns list of colors as numpy arrays
    """
    def color_string_to_np(color_string):
        return np.array([float(st) for st in color_string.split(" ")])

    colors_as_strings = list(itertools.islice(gethtmlcolors(), n))
    colors_as_np = [color_string_to_np(ct) for ct in colors_as_strings]
    return colors_as_np


if __name__ == "__main__":
    bandu_logger.debug(list(itertools.islice(gethtmlcolors(), 100)))
RED = np.array([1, 0, 0])
YELLOW = np.array([1, 1, 0])
GREEN = np.array([0, 1, 0])
BLUE = np.array([0, 0, 1])
BLACK = np.array([0,0,0])
GOLD = np.array([255/255, 215/255, 0])
PURPLE = np.array([128/255, 0, 128/255])
TURQUOISE = np.array([64,224,208])/255
BABY_BLUE = np.array([52, 143, 235])/255
MURKY_GREEN = np.array([44/255, 102/255, 0])
SADDLE_BROWN = np.array([139,69,19])/255
ORANGE = np.array([255,69,0])/255