import re
import json
import os
import numpy as np
import pandas as pd
import networkx as nx
import pydot
from scipy.sparse.csgraph import floyd_warshall
import torch
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from Dataset import data_preparing
from Feature import sensitive_node, adjacency_matrix_n_hops, vectorized
from Model import test_model

test_path0 = "./Dataset/training/malware/ARM/00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a.json"
test_path1 = "./Dataset/training/malware/ARM/00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a.dot"
# D:\Lab\Reserch\Detector\Vectorize\training_valid_vectorized\malware\ARM\00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a.json
test_path2 = "./Vectorize/training_valid_vectorized/malware/ARM/00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a.json"
test_path3 = "./Vectorize/training_valid_vectorized/malware/ARM/00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a_adj_matrix.npy"
test_path4 = "./Vectorize/training_valid_vectorized/malware/ARM/00271448d979eb01e0fb9b8a421e0653914c9963c210de7e61f88403ab83529a_index_function.json"

# import word2vec model
test_path5 = "./Feature/word2vec_with_valid.model"



def main():
    '''
    new_json = {}
    # read the json file
    print("0")
    with open(f"{test_path0}", "r") as f:
        json_data = f.read()
        data = json.loads(json_data)

        # initialize the vectorized feature
        vector = [0]*512

        for node in data.values():
            node['vector'] = vector

        new_json = data
    # write the new json to the file
    with open(f"./Feature/test/test0001b.json", "w") as f:
        json.dump(new_json, f)

    print("1")

    G = nx.Graph(nx.nx_pydot.read_dot(f"{test_path1}"))
    # create the adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix_np = adj_matrix.toarray()
    print(adj_matrix_np[0])
    # save the adjacency matrix
    with open(f"./Feature/test/{test_path1.split('/')[5].split('.')[0]}_adj_matrix.npy", "wb") as f:
        np.save(f, adj_matrix_np)

    print("2")

    # create the index function
    index_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    # save the index function
    with open(f"./Feature/test/{test_path1.split('/')[5].split('.')[0]}_index_function.json", "w") as f:
        json.dump(index_mapping, f)

    print("3")
    model = Word2Vec.load("./Feature/word2vec.model")
    with open(f"./Feature/test/test0001b.json", "r") as f:
        json_data = f.read()
        data = json.loads(json_data)
    for node in data.values():
        for function_name in node['function_name']:
            function_vector = model.wv[function_name]
            # print(function_vector, type(function_vector))
            node['vector'][1:512] = function_vector.tolist()
    with open(f"./Feature/test/test0001b.json", "w") as f:
        json.dump(data, f)

    '''

    # print("Start Genrate Dataset")
    # # test_model.generate_dataset("./Vectorize/training_vectorized", "./Vectorize/valid_vectorized", "./Dataset/FCG_list")
    # print("Finish Genrate Dataset")
    # print("Start model training")
    # GCNmodel_state = test_model.train_GCN_model(20)
    # # store the model
    # torch.save(GCNmodel_state, "./Model/GCNmodel_test.pth")
    # print("Finish model training")

    # repair
    print("Start repair")
    with open(f"{test_path0}", "r") as f:
        json_data = f.read()
        data = json.loads(json_data)

        # initialize the vectorized feature
        vector = [0]*512

        for node in data.values():
            node['vector'] = vector

        new_json = data
    # write the new json to the file
    with open(f"{test_path2}", "w") as f:
        json.dump(new_json, f)

    G = nx.Graph(nx.nx_pydot.read_dot(f"{test_path1}"))
    # create the adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix_np = adj_matrix.toarray()
    # save the adjacency matrix
    with open(f"{test_path3}", "wb") as f:
        np.save(f, adj_matrix_np)

    # create the index function
    index_mapping = {node: idx for idx, node in enumerate(G.nodes())}
    # save the index function
    with open(f"{test_path4}", "w") as f:
        json.dump(index_mapping, f)

    rule = re.compile(r'svc 0x900([0-9A-Fa-f]{3})')
    with open(f"{test_path2}", "r") as f:
        # print(f"{file}.json")
        json_data = f.read()
        data = json.loads(json_data)
    for node in data.values():
        for instruction in node['instructions']:
            if rule.match(instruction):
                node['vector'][0] = 1
    with open(f"{test_path2}", "w") as f:
        json.dump(data, f)


    model = Word2Vec.load(f"{test_path5}")
    with open(f"{test_path2}", "r") as f:
        json_data = f.read()
        data = json.loads(json_data)
    for node in data.values():
        for function_name in node['function_name']:
            function_vector = model.wv[function_name]
            node['vector'][1:512] = function_vector.tolist()
    with open(f"{test_path2}", "w") as f:
        json.dump(data, f)

    print("Finish repair")




if __name__ == "__main__":
    main()
