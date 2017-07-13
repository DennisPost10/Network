import math
import os

from sklearn.utils import shuffle

import numpy as np


class CNN_Inputparser:
    
    def read_data(self, prot_name_file):
        base_name = os.path.splitext(prot_name_file)[0]  
        if os.path.exists(base_name + ".cnn_train_windows.npy") and os.path.exists(base_name + ".cnn_train_one_hots.npy") and os.path.exists(base_name + ".cnn_test_windows.npy") and os.path.exists(base_name + ".cnn_test_one_hots.npy"):
            print("reading data...")
            self.prot_matrices = np.load(base_name + ".cnn_train_windows.npy")
            print("read train prots")
            self.prot_outcomes = np.load(base_name + ".cnn_train_one_hots.npy").astype(float)
            print("read train one_hots")
            self.test_set = np.load(base_name + ".cnn_test_windows.npy")
            print("read test prots")
            self.test_set_o = np.load(base_name + ".cnn_test_one_hots.npy").astype(float)
            print("read test one_hots")
            print("...finished reading")
            print(self.prot_matrices.shape)
            print(self.prot_outcomes.shape)
            print(self.test_set.shape)
            print(self.test_set_o.shape)
        else:
            self.prots = []
            with open(prot_name_file) as file:
                for line in file:
                    line = line.strip()
                    self.prots.append(line)
            self.length = len(self.prots)
            print(self.length)
            
            self.prot_matrices = []
            self.prot_outcomes = []
            for prot in self.prots:
                data = np.loadtxt(self.prot_directory + "/" + prot + "/" + prot + ".tab", delimiter="\t", skiprows=1, usecols=range(2, 22), dtype=float)
                if len(data) > self.max_prot_length:
                    print("skipped protein: " + prot + " length: " + str(len(data)))
                    continue
                for i in range(len(data)):
                    for j in range(len(data[i])):
                        data[i][j] = 1.0 / (1.0 + math.exp(data[i][j]))
                data2 = np.zeros([self.max_prot_length, 20], dtype=float)
                data2[:data.shape[0], :data.shape[1]] = data
                
                self.prot_matrices.append(data2)
                one_hots = np.loadtxt(self.prot_directory + "/" + prot + "/" + prot + ".ss_one_hot", dtype = int, delimiter = "\t")
                one = np.zeros([self.max_prot_length, 3], dtype = float)
                one[:one_hots.shape[0], :one_hots.shape[1]] = one_hots
                self.prot_outcomes.append(one)
        
            # save 10% as test set: simply take every 10th row
            self.test_set = self.prot_matrices[::20]
            self.test_set_o = self.prot_outcomes[::20]
            print(self.prots[::20])
            self.prot_matrices = np.delete(self.prot_matrices, list(range(0, len(self.prot_matrices), 20)), axis=0)
            self.prot_outcomes = np.delete(self.prot_outcomes, list(range(0, len(self.prot_outcomes), 20)), axis=0)
            
            self.test_set = np.asarray(self.test_set)
            self.test_set_o = np.asarray(self.test_set_o)
            self.prot_matrices = np.asarray(self.prot_matrices)
            self.prot_outcomes = np.asarray(self.prot_outcomes)
            print(self.test_set.shape)
            print(self.test_set_o.shape)
            
            np.save(base_name + ".cnn_train_windows", self.prot_matrices)
            np.save(base_name + ".cnn_train_one_hots", self.prot_outcomes.astype(int))
            np.save(base_name + ".cnn_test_windows", self.test_set)
            np.save(base_name + ".cnn_test_one_hots", self.test_set_o.astype(int))
            print(base_name + " files written")
            
        self.index = 0
    
    def test_batches(self):
        return self.test_set, self.test_set_o
    
    def next_prots(self, size):
        rows_left = len(self.prot_matrices) - self.index
        first_pull = min(rows_left, size)
        ret_prots = self.prot_matrices[self.index:(self.index + first_pull)]
        ret_prots_o = self.prot_outcomes[self.index:(self.index + first_pull)]
        self.index += first_pull

        if(self.index > len(self.prot_matrices)):
            print("self_index error!!")
        if(self.index == len(self.prot_matrices)):
            self.index = 0
            self.prot_matrices, self.prot_outcomes = shuffle(self.prot_matrices, self.prot_outcomes, random_state=0)
            print("shuffled")
        if(first_pull < size):
            second_pull = size - first_pull
            ret_prots = np.vstack((ret_prots, self.prot_matrices[self.index:(self.index + second_pull):1]))
            ret_prots_o = np.vstack((ret_prots_o, self.prot_outcomes[self.index:(self.index + second_pull):1]))
            self.index += second_pull
            
        return ret_prots, ret_prots_o
    
    def __init__(self, prot_name_file, prot_directory, max_prot_length):
        self.prot_directory = prot_directory
        self.max_prot_length = max_prot_length
        self.read_data(prot_name_file)
        