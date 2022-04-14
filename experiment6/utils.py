from math import log
import random
import numpy as np


def calcGini(dataSet):
    labels_count = {}
    number = len(dataSet)
    for i, data in enumerate(dataSet):
        label = dataSet[i][-1]
        if label in labels_count.keys():
            labels_count[label] += 1
        else:
            labels_count[label] = 1
    Gini = 0.0
    for label, value in labels_count.items():
        pr = 1.0 * value / number * value / number
        Gini += 1 - pr
    return Gini


def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label 计数
        shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
    shannonEnt -= prob * log(prob, 2)  # 利用公式计算
    return shannonEnt


def calcError(dataSet):
    labels_count = {}
    number = len(dataSet)
    for i, data in enumerate(dataSet):
        label = dataSet[i][-1]
        if label in labels_count.keys():
            labels_count[label] += 1
        else:
            labels_count[label] = 1
    pr = max(labels_count.values())
    pr /= number
    Error = 1 - max(1 - pr, pr)
    return Error


def random(dataSet):
    return np.random.rand()
