from utils import *
from datasets import *
import operator
from draw import *


def splitDataSet(dataSet, axis, value):
    retDataSet = []  # 创建返回的数据集列表
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉 axis 特征
            reducedFeatVec.extend(featVec[axis+1:])  # 将符合条件的添加到返回的数据集
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet, mode="Shannon"):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    if mode == "Shannon":
        baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    elif mode == "Gini":
        baseEntropy = calcGini(dataSet)  # 计算数据集的基尼
    elif mode == "Random":
        baseEntropy = random(dataSet)  # 计算数据集的随机
    elif mode == "Error":
        baseEntropy = calcError(dataSet)  # 错误率
    else:
        raise Exception("no such mode")
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取 dataSet 的第 i 个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建 set 集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet 划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            if mode == "Shannon":
                newEntropy += prob * calcShannonEnt(subDataSet)  # 计算数据集的香农熵
            elif mode == "Gini":
                newEntropy += prob * calcGini(subDataSet)  # 计算数据集的基尼
            elif mode == "Random":
                newEntropy += prob * random(subDataSet)  # 计算数据集的随机
            elif mode == "Error":
                newEntropy += prob * calcError(subDataSet)  # 错误率
            else:
                raise Exception("no such mode")
            infoGain = baseEntropy - newEntropy  # 信息增益
            # print("第%d 个特征的增益为%.3f" % (i, infoGain)) #打印每个特征的信息增益
        if (infoGain > bestInfoGain):  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


def createTree(dataSet, labels, featLabels, mode="Shannon"):
    classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:  # 遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet, mode=mode)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的标签
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
    del(classList[bestFeat])  # 删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


def majorityCnt(classList):
    classCount = {}
    # 统计 classList 中每个元素出现的次数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排列
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print(myTree)
    createPlot(myTree)
