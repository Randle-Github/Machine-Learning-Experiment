from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class sklearn_classifier():
    def __init__(self, cfg=None):
        self.n_neighbors = cfg["MODEL"]["n_neighbors"]
        self.metric = cfg["MODEL"]["n_neighbors"]
        self.p = cfg["MODEL"]["n_neighbors"]
        self.kmeans = KNeighborsClassifier(
            n_neighbors=self.n_neighbors, metric=self.metric, p=self.p)

    def fit(self, X, y):
        self.kmeans.fit(X, y)

    def predict(self, X):
        return self.kmeans.predict(X)


class my_classifier():
    def __init__(self, cfg=None):
        self.n_neighbors = cfg["MODEL"]["n_neighbors"]
        self.metric = cfg["MODEL"]["n_neighbors"]
        self.p = cfg["MODEL"]["n_neighbors"]
        self.X = None
        self.y = None

    def dis(self, a, b):
        if self.metric == "minkowski":
            return np.linalg.norm(a-b, ord=self.p)
        elif self.metric == "":
            pass

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def predict(self, X_pred):
        dis = {}
        pred = []
        for i in X_pred:
            dis[i] = self.dis(self.X[i], i)
            dis = sorted(dis.itmes(), key=lambda item: item[1])
            logits = {}
            tot = 0
            for i in dis:
                if tot == self.n_neighbors:
                    break
                if i.key() not in logits:
                    logits[i.key()] = 0
                logits[i.key()] += 1
            logits = sorted(logits.itmes(), key=lambda item: item[1])
            for i in logits:
                pred.append(logits)
                break
        return pred
