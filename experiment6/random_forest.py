from decision_tree import *
from draw import *


class randomForest():
    def __init__(self, dataSet, labels, featLabels, num_trees=3):
        self.forest = []
        self.num_trees = num_trees
        for _ in range(num_trees):
            self.forest.append(createTree(
                dataSet, labels, featLabels, mode="Random"))

    def predict(self, X):
        for i in range(self.num_trees):
            tree = self.forest[i]
            classify = -1
            count = {}
            while(classify == -1):
                for content in tree:
                    subtree = content
                    if type(subtree) == str:
                        classify = subtree
                    else:
                        tree = (content[1])[X[content[0]]]
                        continue
            if classify in count:
                count[classify] += 1
            else:
                count[classify] = 1
        return max(count, key=count.get)

    def __str__(self):
        for i in range(self.num_trees):
            print("------------------------------------")
            print(self.forest[i])
        return "------------------------------------"

    def draw(self):
        for i in range(self.num_trees):
            plt.title("tree" + str(i))
            createPlot(self.forest[i], i)
