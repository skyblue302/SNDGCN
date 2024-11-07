import numpy as np
import os
import json
import pandas as pd
import sys
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import ParameterGrid

from typing import Any
from malwareDetector.detector import detector
from malwareDetector.config import write_config_to_file, read_config

from Dataset import data_preparing
from Feature import sensitive_node, adjacency_matrix_n_hops, vectorized
from Model import model


class subDetector(detector):
    def __init__(self) -> None:
        
        super().__init__()
        # path of raw data
        self.config.path.set_param("malware_raw_data_path", "./Dataset/Raw_data/malware")
        self.config.path.set_param("benignware_raw_data_path", "./Dataset/Raw_data/benign")

        # path of malware csv
        self.config.path.set_param("malware_csv", "./Dataset/malware_selected.csv")
        # path of benignware csv
        self.config.path.set_param("benignware_csv", "./Dataset/benignware_selected.csv")

        # path of Function_calls
        self.config.path.set_param("reverse_path", "./Dataset/Function_calls")

        # path of final_data_set
        self.config.path.set_param("Final_path", "./Dataset/Final_dataset")

        # path_of_dataset_statistics
        self.config.path.set_param("data_statistics", "./Dataset/data_statistics.csv")
        self.config.path.set_param("not_empty_FCG_list_path", "./Dataset/FCG_list")
        self.config.path.set_param("train_test_statistics", "./Dataset/train_test_statistics.csv")

        # path of split_dataset
        self.config.path.set_param("split_dataset_training", "./Dataset/training")
        self.config.path.set_param("split_dataset_valid", "./Dataset/validation")
        self.config.path.set_param("split_dataset_testing", "./Dataset/testing")
        
        # path of sentences for word2vec
        self.config.path.set_param("sentences", "./Feature/sentences.txt")
        self.config.path.set_param("sentences_with_valid", "./Feature/sentences_with_valid.txt")
        # path of word2vec model
        self.config.path.set_param("word2vec_model_no_valid", "./Feature/word2vec_no_valid.model")
        self.config.path.set_param("word2vec_model_with_valid", "./Feature/word2vec_with_valid.model")



        # path of function_name_list
        self.config.path.set_param("function_name_list", "./Feature/function_name_list.csv")

        # path of sensitive_node_list
        self.config.path.set_param("sensitive_node_list", "./Feature/sensitive_node.csv")

        # path of FCG_adjecency_matrix and index_function
        self.config.path.set_param("FCG_adjecency_matrix", "./Feature/FCG_adjecency_matrix")

        # path of FCG_adjecency_matrix and index_function after n_hops
        self.config.path.set_param("FCG_adjecency_matrix_n_hops", "./Feature/FCG_adjecency_matrix_n_hops")

        # path of vectorized feature
        self.config.path.set_param("training_vectorized_feature", "./Vectorize/training_vectorized")
        self.config.path.set_param("valid_vectorized_feature", "./Vectorize/valid_vectorized")
        self.config.path.set_param("training_valid_vectorized_feature", "./Vectorize/training_valid_vectorized")
        self.config.path.set_param("testing_vectorized_feature", "./Vectorize/testing_vectorized")

        # path of final graph
        self.config.path.set_param("path_train_graph", "./Vectorize/train_graph.pth")
        self.config.path.set_param("path_valid_graph", "./Vectorize/valid_graph.pth")
        self.config.path.set_param("path_train_valid_graph", "./Vectorize/train_valid_graph.pth")
        self.config.path.set_param("path_test_graph", "./Vectorize/test_graph.pth")


        self.config.model.set_param("gamma", 0)

        write_config_to_file(self.config, "config0_0.json")


        #self.config = read_config(config_file_path="config.json")

    def extractFeature(self) -> Any:
        data_statistics = {
            "arch": [],
            "family": [],
            "number_of_raw_data": [],
            "number_of_reverse_data": [],
            "number_of_blank_data": [],
            "number_of_not_blank_data": [],
            "nodes_average": [],
            "edges_average": []
        }

        train_test_statistics = {
            "arch": [],
            "family": [],
            "number_of_training_data": [],
            "number_of_validation_data": [],
            "number_of_testing_data": [],
        }

        '''
        # read csv and mkdir for reversing
        for arch in ["ARM", "PowerPC or cisco 4500", "MIPS", "Intel 80386"]:
            data_preparing.read_csv_and_mkdir(self.config.path.malware_raw_data_path, self.config.path.malware_csv, arch, "malware", self.config.path.reverse_path)
        for arch in ["ARM", "PowerPC or cisco 4500", "MIPS", "x86-64"]:
            data_preparing.read_csv_and_mkdir(self.config.path.benignware_raw_data_path, self.config.path.benignware_csv, arch, "benign", self.config.path.reverse_path)
        '''
        
        # reverse the software
        # run bash script to reverse the software
        
        
        
        # check the numbers of softwares in the dataset 
        # and copy the not empty FCG to the final dataset
        '''
        print("Start data statistics")
        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                print(f"Start checking {family} {arch} dataset")
                data_preparing.check_dataset(self.config.path.reverse_path, self.config.path.Final_path, self.config.path.not_empty_FCG_list_path, arch, family, data_statistics)
                print(f"Finish checking {family} {arch} dataset")

        # Write the data_statistics to a csv file
        df = pd.DataFrame(data_statistics)
        df.to_csv(self.config.path.data_statistics, index=False)
        print("Finish data statistics")
        '''

        #############################################
        # test section
        '''
        with open("./Dataset/ARM_malware_empty_FCG_list.txt", "r") as f:
            empty_FCG_list = f.read().splitlines()
            data_preparing.test_for_reversing(self.config.path.malware_reverse_path, "./Dataset/test", "ARM", empty_FCG_list)
        with open("./Dataset/MIPS_malware_empty_FCG_list.txt", "r") as f:
            empty_FCG_list = f.read().splitlines()
            data_preparing.test_for_reversing(self.config.path.malware_reverse_path, "./Dataset/test", "MIPS", empty_FCG_list)
        '''
        #data_preparing.check_dataset_test("./Dataset/output_ARM", "ARM")
        #data_preparing.check_dataset_test("./Dataset/output_MIPS", "MIPS")
        #############################################

        # cut dataset
        # training:valid:test = 7:1:2
        
        '''
        print("Start spliting dataset")
        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                num_train, num_valid, num_test = data_preparing.split_dataset(self.config.path.Final_path, self.config.path.not_empty_FCG_list_path, self.config.path.split_dataset_training, self.config.path.split_dataset_valid, self.config.path.split_dataset_testing, family, arch, 0.1, 0.2)
                train_test_statistics['arch'].append(arch)
                train_test_statistics['family'].append(family)
                train_test_statistics['number_of_training_data'].append(num_train)
                train_test_statistics['number_of_validation_data'].append(num_valid)
                train_test_statistics['number_of_testing_data'].append(num_test)
        df = pd.DataFrame(train_test_statistics)
        df.to_csv(f"{self.config.path.train_test_statistics}", index=False)
        print("Finish spliting dataset")
        '''

        # On training set
        # collect all the function names in the graph
        # Find sensitive nodes in the graph
        
        # print("Start collecting function names on training set")
        # for family in ["malware", "benign"]:
        #     for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #         sensitive_node.collect_function_names(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.function_name_list, family, arch, True)
        # print("Finish collecting function names on training set")
        
        # print("Start finding sensitive nodes")
        # sensitive_node.find_sensitive_node(self.config.path.train_test_statistics, self.config.path.function_name_list, self.config.path.sensitive_node_list, 0.3)
        # print("Finish finding sensitive nodes")
        

        # # Generate Sentences for word2vec
        # print("Start generating sentences for word2vec")
        # for family in ["malware", "benign"]:
        #     for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #         data_preparing.generate_word2vec_sentence(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.sentences, arch, family, False)
        #         print(f"Finish generating sentences for {family} {arch}")
        # print("Finish generating sentences for word2vec")
        
        # # word2vec training
        # print("Start training word2vec")
        # for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #     data_preparing.word2vec_train(self.config.path.sentences, arch, self.config.path.word2vec_model_no_valid_sep)
        # print("Finish training word2vec")

        # On training set + validation set
        # collect all the function names in the graph
        # Find sensitive nodes in the graph

        '''
        print("Start collecting function names on training set and validation set")
        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                sensitive_node.collect_function_names(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.function_name_list, family, arch, True)
        print("Finish collecting function names on training set and validation set")
        
        
        print("Start finding sensitive nodes")
        sensitive_node.find_sensitive_node(self.config.path.train_test_statistics, self.config.path.function_name_list, self.config.path.sensitive_node_list, 0.3)
        print("Finish finding sensitive nodes")
        '''

        # Generate Sentences for word2vec
        print("Start generating sentences for word2vec")
        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                data_preparing.generate_word2vec_sentence(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.sentences_with_valid, arch, family, True)
                data_preparing.generate_word2vec_sentence(self.config.path.split_dataset_valid, self.config.path.not_empty_FCG_list_path, self.config.path.sentences_with_valid, arch, family, False)
                print(f"Finish generating sentences for {family} {arch}")
        print("Finish generating sentences for word2vec")
        
        # word2vec training
        print("Start training word2vec")
        for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
            data_preparing.word2vec_train(self.config.path.sentences_with_valid, arch, self.config.path.word2vec_model_with_valid)
        print("Finish training word2vec")



        
        

    def vectorize(self, nhops:int) -> np.array:
        # sensitive_node_list = []
        # with open(self.config.path.sensitive_node_list, "r") as f:
        #     df = pd.read_csv(self.config.path.sensitive_node_list)
        #     sensitive_node_list = df['function_name'].tolist()

        # vectorize the graph on training set
        # for family in ["malware", "benign"]:
        #     for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #         # initialize the vectorized feature and adjacency matrix 
        #         print(f"Start initialize vectorizing on {family} {arch} training set")
        #         os.makedirs(f"{self.config.path.training_vectorized_feature}/{family}/{arch}", exist_ok=True)
        #         vectorized.vectorized_initialize(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.training_vectorized_feature, arch, family, True)
        #         vectorized.adjacency_matrix(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.training_vectorized_feature, arch, family, True)
        #         print(f"Finish initialize vectorizing on {family} {arch} training set")
        # # vectorize the graph on validation set
        #         # initialize the vectorized feature and adjacency matrix 
        #         print(f"Start initialize vectorizing on {family} {arch} validation set")
        #         os.makedirs(f"{self.config.path.valid_vectorized_feature}/{family}/{arch}", exist_ok=True)
        #         vectorized.vectorized_initialize(self.config.path.split_dataset_valid, self.config.path.not_empty_FCG_list_path, self.config.path.valid_vectorized_feature, arch, family, False)
        #         vectorized.adjacency_matrix(self.config.path.split_dataset_valid, self.config.path.not_empty_FCG_list_path, self.config.path.valid_vectorized_feature, arch, family, False)
        #         print(f"Finish initialize vectorizing on {family} {arch} validation set")

        # if nhops > 0:
        #     for family in ["malware", "benign"]:
        #         for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #             # simplify the graph
        #             print(f"Start simplify on {family} {arch} training set")
        #             vectorized.do_n_hops(self.config.path.training_vectorized_feature, self.config.path.training_vectorized_feature, sensitive_node_list, nhops)
        #             # vectorized.do_n_hops(self.config.path.training_vectorized_feature, self.config.path.training_vectorized_feature, sensitive_node_list, 1)
        #             print(f"Finish simplify on {family} {arch} training set")
        #             print(f"Start simplify on {family} {arch} validation set")
        #             vectorized.do_n_hops(self.config.path.valid_vectorized_feature, self.config.path.valid_vectorized_feature, sensitive_node_list, nhops)
        #             # vectorized.do_n_hops(self.config.path.valid_vectorized_feature, self.config.path.valid_vectorized_feature, sensitive_node_list, 1)
        #             print(f"Finish simplify on {family} {arch} validation set")

        # for family in ["malware", "benign"]:
        #     for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
        #         # vectorize the graph
        #         # sys_api
        #         # print(f"Start vectorizing sys_api {family} {arch} on training set")
        #         # vectorized.sys_api_detect(self.config.path.training_vectorized_feature, self.config.path.not_empty_FCG_list_path, arch, family, True)
        #         # print(f"Finish vectorizing sys_api {family} {arch} on training set")
        #         # print(f"Start vectorizing sys_api {family} {arch} on validation set")
        #         # vectorized.sys_api_detect(self.config.path.valid_vectorized_feature, self.config.path.not_empty_FCG_list_path, arch, family, False)
        #         # print(f"Finish vectorizing sys_api {family} {arch} on validation set")

        #         # word2vec embedding
        #         print(f"Start word2vec embedding {family} {arch}")
        #         # vectorized.word2vec_embedding(self.config.path.training_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.word2vec_model_no_valid, arch, family, True)
        #         vectorized.word2vec_embedding(self.config.path.valid_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.word2vec_model_no_valid, arch, family, False)
        #         print(f"Finish word2vec embedding {family} {arch}")
        

        # testing set
        # vectorize the graph on training set
        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                # initialize the vectorized feature and adjacency matrix 
                print(f"Start initialize vectorizing on {family} {arch} training set")
                os.makedirs(f"{self.config.path.training_valid_vectorized_feature}/{family}/{arch}", exist_ok=True)
                vectorized.vectorized_initialize(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.training_valid_vectorized_feature, arch, family, 0)
                vectorized.adjacency_matrix(self.config.path.split_dataset_training, self.config.path.not_empty_FCG_list_path, self.config.path.training_valid_vectorized_feature, arch, family, 0)
                vectorized.vectorized_initialize(self.config.path.split_dataset_valid, self.config.path.not_empty_FCG_list_path, self.config.path.training_valid_vectorized_feature, arch, family, 1)
                vectorized.adjacency_matrix(self.config.path.split_dataset_valid, self.config.path.not_empty_FCG_list_path, self.config.path.training_valid_vectorized_feature, arch, family, 1)
                print(f"Finish initialize vectorizing on {family} {arch} training set")
        # vectorize the graph on testing set
                # initialize the vectorized feature and adjacency matrix 
                print(f"Start initialize vectorizing on {family} {arch} testing set")
                os.makedirs(f"{self.config.path.testing_vectorized_feature}/{family}/{arch}", exist_ok=True)
                vectorized.vectorized_initialize(self.config.path.split_dataset_testing, self.config.path.not_empty_FCG_list_path, self.config.path.testing_vectorized_feature, arch, family, 2)
                vectorized.adjacency_matrix(self.config.path.split_dataset_testing, self.config.path.not_empty_FCG_list_path, self.config.path.testing_vectorized_feature, arch, family, 2)
                print(f"Finish initialize vectorizing on {family} {arch} testing set")

        for family in ["malware", "benign"]:
            for arch in ["ARM", "PowerPC", "MIPS", "x86"]:
                # vectorize the graph
                # sys_api
                print(f"Start vectorizing sys_api {family} {arch} on training set")
                vectorized.sys_api_detect(self.config.path.training_valid_vectorized_feature, self.config.path.not_empty_FCG_list_path, arch, family, True)
                print(f"Finish vectorizing sys_api {family} {arch} on training set")
                print(f"Start vectorizing sys_api {family} {arch} on testing set")
                vectorized.sys_api_detect(self.config.path.testing_vectorized_feature, self.config.path.not_empty_FCG_list_path, arch, family, False)
                print(f"Finish vectorizing sys_api {family} {arch} on testing set")

                # word2vec embedding
                print(f"Start word2vec embedding {family} {arch}")
                vectorized.word2vec_embedding(self.config.path.training_valid_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.word2vec_model_with_valid, arch, family, True)
                vectorized.word2vec_embedding(self.config.path.testing_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.word2vec_model_with_valid, arch, family, False)
                print(f"Finish word2vec embedding {family} {arch}")


        return 'This is the implementation of the vectorize function from the derived class.'

    def model(self) -> Any:
        # Prepare the dataset
        model.generate_dataset(self.config.path.training_vectorized_feature, self.config.path.valid_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.path_train_graph, self.config.path.path_valid_graph)
        
        #hyperparameters tuning
        param_grid = {
            'alpha': [1.2, 1.5, 1.8],
            'lambd': [0.2, 0.4, 0.6],
            'mu': [0.5, 0.7, 0.9],
            'K_l': [1.5, 1.8, 2.0, 2.2, 2.4]
        }
        param_combinations = list(ParameterGrid(param_grid))

        print("Start hyperparameters tuning...")
        _, param = model.train_GCN_model_find_hyperparameters(50, param_combinations)
        print("Finish hyperparameters tuning")
        print(param)

        # # model training
        # print("Start model training")
        # GCNmodel_state = model.train_GCN_model(30)
        # # store the model
        # torch.save(GCNmodel_state, "./Model/GCNmodel.pth")
        # print("Finish model training")

        return 'This is the implementation of the model function from the derived class.'

    def predict(self) -> np.array:
        # Prepare the dataset
        model.generate_dataset(self.config.path.training_valid_vectorized_feature, self.config.path.testing_vectorized_feature, self.config.path.not_empty_FCG_list_path, self.config.path.path_train_valid_graph, self.config.path.path_test_graph)

        # final training
        print("Start model training")
        GCNmodel_state = model.train_GCN_model(50, 1.8, 0.6, 0.5, 2.0, self.config.path.path_train_valid_graph, self.config.path.path_test_graph)

        # store the model
        torch.save(GCNmodel_state, "./Model/GCNmodel.pth")
        print("Finish model training")

        return 'This is the implementation of the predict function from the derived class.'

if __name__ == '__main__':
    myDetector = subDetector()
    # myDetector.extractFeature()
    myDetector.vectorize(0)
    # myDetector.model()
    myDetector.predict()
    # myDetector.mkdir()
