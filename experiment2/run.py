import yaml
from datasets import watermelon, iris
import model.classifier_helper
import matplotlib.pyplot as plt


def load_cfg():
    with open("config/config.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg

def work(cfg=None):
    if cfg["DATASET"]["NAME"] == "watermelon":
        data = watermelon.WaterMelon()
        all_data = data.all()
        train_data = data.train()
        test_data = data.test()

        '''
        plt.title("all_nodes_watermelon")
        plt.scatter([all_data[0][i][0] for i in range(16) if all_data[1][i] == 0],
                    [all_data[0][i][1] for i in range(16) if all_data[1][i] == 0],
                    c="orange", label="good watermelon")
        plt.scatter([all_data[0][i][0] for i in range(16) if all_data[1][i] == 1],
                    [all_data[0][i][1] for i in range(16) if all_data[1][i] == 1],
                    c="green", label="bad watermelon")
        plt.xlabel("density")
        plt.ylabel("sugar")
        plt.legend()
        plt.show()
        '''

        if cfg["MODEL"]["SELF_MODEL"] == False:
            m = model.classifier_helper.sklearn_classifier(cfg)
        else:
            m = model.classifier_helper.my_classifier(cfg)
        m.fit(train_data[0], train_data[1])
        pred = m.predict(test_data[0])
        acc = 0
        for i in range(len(test_data[1])):
            if pred[i] == test_data[1][i]:
                acc += 1
        acc = acc / len(test_data[1]) * 100
        print(acc, "%")
        plt.title("prediction_watermelon, n_neighbors = 4")
        plt.scatter([train_data[0][i][0] for i in range(12) if train_data[1][i] == 0],
                    [train_data[0][i][1]
                        for i in range(12) if train_data[1][i] == 0],
                    c="orange", label="training_data: good watermelon")
        plt.scatter([train_data[0][i][0] for i in range(12) if train_data[1][i] == 1],
                    [train_data[0][i][1]
                        for i in range(12) if train_data[1][i] == 1],
                    c="green", label="training_data: bad watermelon")
        plt.scatter([test_data[0][i][0] for i in range(4) if pred[i] == 0],
                    [test_data[0][i][1] for i in range(4) if pred[i] == 0],
                    c="red", label="test_data: good watermelon")
        plt.scatter([test_data[0][i][0] for i in range(4) if pred[i] == 1],
                    [test_data[0][i][1] for i in range(4) if pred[i] == 1],
                    c="purple", label="test_data: bad watermelon")
        plt.xlabel("density")
        plt.ylabel("sugar")
        plt.legend(fontsize=7)
        plt.show()

    elif cfg["DATASET"]["NAME"] == "iris":
        data = iris.Iris(cfg)
        train_data = data.train()
        test_data = data.test()
        ans = []
        ans.append(0.)

        if cfg["MODEL"]["SELF_MODEL"] == False:
            m = model.classifier_helper.sklearn_classifier(cfg)
        else:
            m = model.classifier_helper.my_classifier(cfg)
        m.fit(train_data[0], train_data[1])
        pred = model.predict(test_data[0])
        acc = 0
        for i in range(len(test_data[1])):
            if pred[i] == test_data[1][i]:
                acc += 1
        acc = acc / len(test_data[1]) * 100
        print(acc, "%")


if __name__ == "__main__":
    cfg = load_cfg()
    # print(cfg)
    work(cfg)
