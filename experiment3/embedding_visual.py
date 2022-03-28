import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import DeepWalk
import numpy as np

file = open("word_embedding_visual.txt", "r")
content = file.readlines()
node = np.zeros((821, 2))
tot = -1
for line in content:
    tot += 1
    x, y, z = line.split(" ")
    node[tot][0] = float(x)
    node[tot][1] = float(y)

color = {0:"red", 1:"blue", 2:"brown", 3:"green", 4:"black", 5:"pink", 6:'purple'}
belong = np.zeros(821)
belong[0] = 1
belong[9] = 1
belong[4] = 1
belong[6] = 1
belong[47] = 1
belong[44] = 1
belong[7] = 1
belong[11] = 1

belong[66] = 2
belong[391] = 2

belong[215] = 3
belong[246] = 3
belong[165] = 3
belong[307] = 3
belong[314] = 3
belong[192] = 3
belong[231] = 3
belong[232] = 3
belong[238] = 3
belong[243] = 3
belong[311] = 3
belong[244] = 3
belong[30] = 3
belong[35] = 3
belong[308] = 3
belong[79] = 3
belong[309] = 3
belong[513] = 3
belong[706] = 3
belong[707] = 3
belong[735] = 3
belong[736] = 3
belong[774] = 3
belong[770] = 3
belong[777] = 3

def line(node1, node2):
    x1=node1[0]
    x2=node2[0]
    y1=node1[1]
    y2=node2[1]
    x=np.linspace(x1, x2, 20)
    plt.plot(x, (y2-y1)/(x2-x1)*(x-x1)+y1, color="blue", linewidth = 0.1)

fig = plt.figure()
for i in range(821):
    plt.scatter(node[i][0],node[i][1],c=color[belong[i]])
plt.scatter(node[1][0],node[1][1],c=color[belong[1]],label="unlabeled")
plt.scatter(node[0][0],node[0][1],c=color[belong[0]],label="communication")
plt.scatter(node[66][0],node[66][1],c=color[belong[66]],label="information theory")
plt.scatter(node[215][0],node[215][1],c=color[belong[215]],label="signal theory")
plt.legend()

Side = DeepWalk.main()

for side in Side:
    line(node[int(side[0])], node[int(side[1])])

plt.show()
