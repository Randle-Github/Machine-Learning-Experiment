from sklearn.datasets import load_iris
import numpy as np
from sklearn.decomposition import PCA


class Iris():
    def __init__(self, cfg=None):
        self.cfg = cfg
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        if self.cfg["MODEL"]["PCA"] == True:
            pca = PCA(n_components=2, random_state=42)
            self.X = pca.fit_transform(self.X, self.y)

    def all(self):
        return [self.X, self.y]

    def train(self):
        x = []
        y = []
        for i in range(150):
            if i % 15 < 3:
                x.append(self.X[i])
                y.append(self.y[i])
        return [x, y]

    def test(self):
        x = []
        y = []
        for i in range(150):
            if i % 15 >= 3:
                x.append(self.X[i])
                y.append(self.y[i])
        return [x, y]

    def __getitem__(self, item):
        return [self.X[item], self.y[item]]
