import matplotlib.pyplot as plt


def visualise(data):
    plt.title("PCA visualisation of all nodes")
    l = int(len(data[1]))
    plt.scatter([data[0][i][0] for i in range(l) if data[1][i] == 0],
                [data[0][i][1] for i in range(l) if data[1][i] == 0], c="red", label="Setosa")
    plt.scatter([data[0][i][0] for i in range(l) if data[1][i] == 1],
                [data[0][i][1] for i in range(l) if data[1][i] == 1], c="green", label="Versicolour")
    plt.scatter([data[0][i][0] for i in range(l) if data[1][i] == 2],
                [data[0][i][1] for i in range(l) if data[1][i] == 2], c="orange", label="Virginica")
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()
