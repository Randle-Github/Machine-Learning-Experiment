from decision_tree import *
from random_forest import *

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(len(dataSet))
    TrainingSet = dataSet[:16]
    TestingSet = dataSet[16:]
    # print(dataSet)
    featLabels = []
    # myTree = createTree(dataSet, labels, featLabels)
    myTree = randomForest(TrainingSet, labels, featLabels, num_trees=5)
    print(myTree)
    # myTree.predict()
    # print(myTree)
    myTree.draw()
    # createPlot(myTree)
    plt.show()  # 显示绘制结果
