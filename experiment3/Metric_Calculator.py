import numpy as np


def calc(a, belong, mean):
    belong = int(belong)
    if belong == 0:
        belong = 1
        for i in range(1, 3):
            if (mean[i][0] - a[0]) ** 2 + (mean[i][1] - a[1]) ** 2 < (mean[belong][0] - a[0]) ** 2 + (mean[belong][1] - a[1]):
                belong = i
    return (mean[belong][0] - a[0]) * 0.75 + a[0], (mean[belong][1] - a[1]) * 0.75 + a[1], belong
