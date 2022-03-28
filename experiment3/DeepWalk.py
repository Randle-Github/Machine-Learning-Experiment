import pandas as pd
import torch
import random
import numpy as np


def Deep_Walk(graph, node, left_steps, temp):
    if left_steps <= 0:
        return
    temp[node] += 1
    r = random.randint(0, len(graph[node]) - 1)
    target = graph[node][r]
    Deep_Walk(graph, target, left_steps - 1, temp)


def walk(graph, walk_length, times_each_node, path):
    for i in range(821):
        temp = torch.zeros(821)
        for j in range(times_each_node):
            Deep_Walk(graph, i, walk_length, temp)
            path[i] += temp


def main():
    content = pd.read_excel("data.xlsx")
    set_from = list(content["from"])
    set_rel = list(content['rel'])
    set_to = list(content["to"])
    Side = []

    for i in range(len(set_to)):
        Side.append((set_from[i], set_to[i], set_rel[i]))  # (from, to, relationship)
        Side.append((set_to[i], set_from[i], set_rel[i] + " by"))
    word_list = []
    word_hash = {}
    word_dehash = {}
    tot = -1
    graph = {}
    for side in Side:
        if side[0] not in word_list:
            tot += 1
            word_list.append(side[0])
            word_hash[side[0]] = tot
            word_dehash[tot] = side[0]
        if side[1] not in word_list:
            tot += 1
            word_list.append(side[1])
            word_hash[side[1]] = tot
            word_dehash[tot] = side[1]
    for side in Side:
        if word_hash[side[0]] not in graph:
            graph[word_hash[side[0]]] = []
        graph[word_hash[side[0]]].append(word_hash[side[1]])

    length = 821
    path = torch.zeros(length, length)
    walk(graph, 6, 20, path)
    SIDE = np.zeros((len(Side), 2))
    tot = -1
    for side in Side:
        tot += 1
        i = word_hash[side[0]]
        j = word_hash[side[1]]
        SIDE[tot][0] = i
        SIDE[tot][1] = j
    return SIDE # path
