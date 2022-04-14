from math import log
import operator
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt


def createDataSet():
    file = open('lenses.data', 'r')
    files = file.readline()
    dataSet = []
    while files:
        line = list(map(str, files.split(' ')))
        list_ = []
        for i in range(1, 5):
            list_.append(int(line[i]))
        list_.append(str(line[5][0]))
        dataSet.append(list_)
        files = file.readline()
    labels = ['DD', 'A', 'E', 'R']  # 特征标签
    return dataSet, labels  # 返回数据集和分类属性
