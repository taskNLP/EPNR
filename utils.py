import logging
import pickle
import time
import numpy as np
import sys
from collections import defaultdict

sys.setrecursionlimit(3000)


def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_label_to_relation(labels):
    left = 0
    right = 0
    relation = ""
    relations = []
    for label in labels:
        if label == 0:
            break
        relation = relation + "-" + str(label)
        if label == 1:
            right += 1
        elif label == 2:
            left += 1
            if right == left and right != 0:
                relations.append(relation)
                relation = ''
    return relations


def get_neighbor_list(adjacent_matrix, length):
    adjacent_list = {}
    for i in range(length):
        for j in range(length):
            if i == j:
                continue
            if adjacent_matrix[i, j] != 0:
                if str(i) in adjacent_list:
                    if j not in adjacent_list[str(i)]:
                        adjacent_list[str(i)].append(j)
                else:
                    adjacent_list[str(i)] = [j]
                if str(j) in adjacent_list:
                    if i not in adjacent_list[str(j)]:
                        adjacent_list[str(j)].append(i)
                else:
                    adjacent_list[str(j)] = [i]
    return adjacent_list


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def neighbor(v, lista):
    """retorna os vertices adjacentes a v"""
    return lista[str(v)]


def union(lst1, lst2):
    return lst1 + lst2


def get_all_vertex(lista):
    """retorna P (all vertex from adjacency list)"""
    vertices = []
    for key, value in lista.items():
        vertices.append(int(key))
    return vertices


def bron_kerb_algorithm_no_pivot(R, P, X, lista, relations):
    """"
    based on: https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm


    algorithm BronKerbosch1(R, P, X) is
    if P and X are both empty then
        report R as a maximal clique
    for each vertex v in P do
        BronKerbosch1(R U v, P intersect N(v), X intersect N(v))
        P := P \  v
        X := X U v
        """
    # golfinhos_txt = read_file('soc-dolphins.txt')
    # lista = create_dolphin_list(golfinhos_txt)

    #  Retirado dos slides das aulas:
    #  R: vertices que seriam parte do clique
    #  P: vertices que tem ligacao com todos os vertices de R (candidatos).
    #  Conjunto X: vertices ja analisados e que nao  levam a uma extensao do conjunto R.

    if len(P) == 0 and len(X) == 0:
        relations.append(R)

    for v in P[:]:
        bron_kerb_algorithm_no_pivot(
            R=union(R, [v]),
            P=intersection(P, neighbor(v, lista)),
            X=intersection(X, neighbor(v, lista)),
            lista=lista,
            relations=relations)
        P.remove(v)
        X.append(v)


def get_entity_seq(entity_list, matrix):
    graph = Graph(isDirected=True)
    for i in entity_list:
        for j in entity_list:
            if (i != j) and matrix[i][j] != 0:
                graph.addEdge(i, j)
    return graph.topoSortDfs()


def get_relation(entity_seq, matrix):
    length = len(entity_seq)
    if length == 0:
        return False
    elif length == 1:
        return entity_seq[0]
    elif length == 2:
        if matrix[entity_seq[0]][entity_seq[1]] > 1:
            return [entity_seq[0], entity_seq[1], matrix[entity_seq[0]][entity_seq[1]]]
        else:
            return entity_seq[0]
    else:
        for i in range(length):
            for j in range(length - 1, -1, -1):
                if (matrix[entity_seq[i]][entity_seq[j]] > 1) and (entity_seq[i] != entity_seq[j]):
                    return [get_relation(entity_seq[:j], matrix), get_relation(entity_seq[j:], matrix),
                            matrix[entity_seq[i]][entity_seq[j]]]


def decode_relation(relation, entity_list, vocab, sentence, orig_list=None):
    if isinstance(relation[0], int):
        head = {
            "mention": sentence[entity_list[relation[0]][0]:entity_list[relation[0]][-1] + 1],
            "start_idx": int(entity_list[relation[0]][0]),
            "end_idx": int(entity_list[relation[0]][-1]) + 1
        }
    else:
        head = decode_relation(relation[0], entity_list, vocab, sentence)


    if isinstance(relation[1], int):
        tail = {
            "type": "mention",
            "mention": sentence[entity_list[relation[1]][0]:entity_list[relation[1]][-1] + 1],
            "start_idx": int(entity_list[relation[1]][0]),
            "end_idx": int(entity_list[relation[1]][-1]) + 1
        }
    else:
        tail = decode_relation(relation[1], entity_list, vocab, sentence)
        tail["type"] = "relation"

    result = {
        "head": head,
        "relation": vocab.id_to_label(int(relation[-1])),
        "tail": tail
    }
    return result


class Graph:
    def __init__(self, isDirected=False):
        self.graph = defaultdict(list)
        self.isDirected = isDirected

    def addEdge(self, start, end):
        self.graph[start].append(end)

        if self.isDirected is False:
            self.graph[end].append(start)
        else:
            self.graph[end] = self.graph[end]

    def topoSortvisit(self, s, visited, sortlist):
        visited[s] = 1

        succ = True
        for i in self.graph[s]:
            if 0 == visited[i]:
                succ = succ and self.topoSortvisit(i, visited, sortlist)
            elif 1 == visited[i]:
                return False
        if succ:
            visited[s] = 2
            sortlist.insert(0, s)
            return True
        else:
            return False

    def topoSortDfs(self):

        visited = {i: 0 for i in self.graph}

        sortlist = []

        for u in self.graph:
            if 0 == visited[u]:
                if not self.topoSortvisit(u, visited, sortlist):
                    return None
        return sortlist


def decode(outputs, labels, lengths):
    r, p, c = 0, 0, 0
    decode_result = []

    for output, label, length in zip(outputs, labels, lengths):

        predict_result = []
        result = []
        predicts_neighbor_list = get_neighbor_list(output, length)
        predicts = []
        bron_kerb_algorithm_no_pivot(R=[],
                                     P=get_all_vertex(predicts_neighbor_list),
                                     X=[],
                                     lista=predicts_neighbor_list,
                                     relations=predicts)
        for predict in predicts:
            entity_seq = get_entity_seq(predict, output)
            if entity_seq is None or not entity_seq:
                entity_seq = predict
            if len(entity_seq) == 2 and output[entity_seq[0]][entity_seq[1]] <= 1:
                continue
            try:
                relation = get_relation(entity_seq, output)
            except:
                print("===========RecursionError: maximum recursion depth exceeded in comparison==============")
            if not relation:
                continue
            predict_result.append(str(relation))
            result.append(relation)
        decode_result.append(result)
        r += len(label)
        p += len(predict_result)
        c += len(label) - len(set(label).difference(set(predict_result)))
    return c, p, r, decode_result


def cal_f1(c, p, r):
    if r == 0 or p == 0:
        return 0, 0, 0

    r = c / r if r else 0
    p = c / p if p else 0

    if r and p:
        return 2 * p * r / (p + r), p, r
    return 0, p, r

