# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import time
import json

from networkx.readwrite import json_graph
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import gfile

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'ppi', 'Dataset string.')
flags.DEFINE_string('data_prefix', 'data/', 'Datapath prefix.')
flags.DEFINE_integer('low_degree', 0, 'minimum node degree.')


def load_graphsage_data(dataset_path, dataset_str, low_degree=0):
    graph_json = json.load(
        gfile.Open('{}/{}/{}-G.json'.format(dataset_path, dataset_str,
                                            dataset_str)))
    graph_nx = json_graph.node_link_graph(graph_json)

    id_map = json.load(
        gfile.Open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str,
                                                 dataset_str)))
    class_map = json.load(
        gfile.Open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str,
                                                    dataset_str)))
    feats = np.load(
        gfile.Open(
            '{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str),
            'rb')).astype(np.float32)

    print("load finish")

    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}

    deg = np.zeros((len(id_map),))
    for edge in graph_nx.edges():
        if edge[0] in id_map and edge[1] in id_map:
            deg[id_map[edge[0]]] += 1
            deg[id_map[edge[1]]] += 1

    to_remove_node = []
    degree_throd = low_degree
    for node in graph_nx.nodes():
        if deg[node] <= degree_throd:
            to_remove_node.append(node)

    for node in graph_json['nodes']:
        if node['test'] and node['id'] in to_remove_node:
            to_remove_node.remove(node['id'])

    for node in graph_json['nodes']:
        if node['val'] and node['id'] in to_remove_node:
            to_remove_node.remove(node['id'])

    print("to remove node %i" % len(to_remove_node))

    print("before del node %i " % len(graph_json['nodes']))
    tmpN = [i for i in graph_json['nodes'] if i['id'] not in to_remove_node]

    cntNode = 0
    dictMapId = {}
    for node in tmpN:
        dictMapId[node['id']] = cntNode
        cntNode += 1
    print("cntNode %i" % cntNode)


    graph_json['nodes'] = tmpN
    for nodeId in graph_json['nodes']:
        nodeId['id'] = dictMapId[nodeId['id']]
    print("after del node %i " % len(graph_json['nodes']))

    print("before del edge %i " % len(graph_json['links']))
    tmpE = [i for i in graph_json['links'] if i['target'] not in to_remove_node and i['source'] not in to_remove_node]
    graph_json['links'] = tmpE
    for edge in graph_json['links']:
        # print(edge['source'], edge['target'],dictMapId[edge['source']], dictMapId[edge['target']])
        edge['target'] = dictMapId[edge['target']]
        edge['source'] = dictMapId[edge['source']]
    print("after del edge %i " % len(graph_json['links']))

    tmpId_map = {}
    for i in range(cntNode):
        tmpId_map[i] = id_map[i]

    cnt = 0
    feats_new = np.empty(shape=[0, 50], dtype=np.float32())
    for tmp_fea in feats:
        if cnt not in to_remove_node:
            feats_new = np.append(feats_new, [tmp_fea], axis=0)
        cnt = cnt + 1

    tmpC = [i for i in class_map if int(i) not in to_remove_node]
    class_map_new = {}
    for d in tmpC:
        class_map_new[str(dictMapId[int(d)])] = class_map[d]

    print("deal finish")

    jsonStr = json.dumps(graph_json)
    f1 = open("/media/zhaoyy/0A570EFD0A570EFD/test/py/preprocess-remove-nodes/data/ppi/ppi-" + str(degree_throd) + "-G.json", "w+")
    print(jsonStr, file=f1, flush=True)

    np.save('/media/zhaoyy/0A570EFD0A570EFD/test/py/preprocess-remove-nodes/data/ppi/ppi-' + str(degree_throd) + '-feats', feats_new)

    jsonStrIdMap = json.dumps(tmpId_map)
    f2 = open("/media/zhaoyy/0A570EFD0A570EFD/test/py/preprocess-remove-nodes/data/ppi/ppi-" + str(degree_throd) + "-id_map.json",
              "w+")
    print(jsonStrIdMap, file=f2, flush=True)

    jsonStrClassMap = json.dumps(class_map_new)
    f3 = open("/media/zhaoyy/0A570EFD0A570EFD/test/py/preprocess-remove-nodes/data/ppi/ppi-" + str(degree_throd) + "-class_map.json",
              "w+")
    print(jsonStrClassMap, file=f3, flush=True)


def main():
    load_graphsage_data(FLAGS.data_prefix, FLAGS.dataset, FLAGS.low_degree)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

