import numpy as np

class WaterMelon():
    def __init__(self):
        self.X = np.array([[0.697, 0.460],
                           [0.774, 0.376],
                           [0.634, 0.264],
                           [0.608, 0.318],
                           [0.556, 0.215],
                           [0.403, 0.237],
                           [0.481, 0.149],
                           [0.437, 0.211],
                           [0.666, 0.091],
                           [0.243, 0.267],
                           [0.343, 0.099],
                           [0.639, 0.161],
                           [0.657, 0.198],
                           [0.360, 0.370],
                           [0.593, 0.042],
                           [0.719, 0.103],
                           ])
        self.y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])

    def __getitem__(self, item):
        return [self.X[item], self.y[item]]

    def all(self):
        return [self.X, self.y]

    def train(self):  # train:test = 3:1
        return [[self.X[i] for i in range(16) if i % 4 <= 2], [self.y[i] for i in range(16) if i % 4 <= 2]]

    def test(self):
        return [[self.X[i] for i in range(16) if i % 4 > 2], [self.y[i] for i in range(16) if i % 4 > 2]]
