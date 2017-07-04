import os
import sys

from sklearn.utils import shuffle

from ConfigFileParser import Configurations
import numpy as np


class Inputparser:
    
    def read_data(self, prot_name_file):
        base_name = os.path.splitext(prot_name_file)[0]  
        if os.path.exists(base_name + ".cnn_train_windows") and os.path.exists(base_name + ".cnn_train_one_hots") and os.path.exists(base_name + ".cnn_test_windows") and os.path.exists(base_name + ".cnn_test_one_hots"):
            print("reading data...")
            self.all_windows = np.loadtxt(base_name + ".cnn_train_windows", dtype = float)
            print("read train windows")
            self.all_one_hots = np.loadtxt(base_name + ".cnn_train_one_hots", dtype = float)
            print("read train one_hots")
            self.test_set = np.loadtxt(base_name + ".cnn_test_windows", dtype = float)
            print("read test windows")
            self.test_set_o = np.loadtxt(base_name + ".cnn_test_one_hots", dtype = float)
            print("read test one_hots")
            print("...finished reading")
        else:
            self.prots = []
            with open(prot_name_file) as file:
                for line in file:
                    line = line.strip()
                    self.prots.append(line)
            self.length = len(self.prots)
            print(self.length)
            
            self.all_windows = []
            self.all_one_hots = []
            
            for prot in self.all_prots:
                windows = np.loadtxt("D:/Dennis/Uni/bachelor/parsed/" + prot + "/" + prot + ".tab", dtype = float, delimiter = "\t")
                for x in windows:
                    self.all_windows.append(x)
                one_hots = np.loadtxt("D:/Dennis/Uni/bachelor/parsed/" + prot + "/" + prot + ".ss_one_hot", dtype = int, delimiter = "\t")
                for x in one_hots:
                    self.all_one_hots.append(x)
            self.all_windows = np.reshape(self.all_windows, [-1, 20])
            self.all_one_hots = np.reshape(self.all_one_hots, [-1, 20])
            self.all_prots = None
            self.l = len(self.all_windows)
            print(self.l)
        
            # save 10% as test set: simply take every 10th row
            self.test_set = self.all_windows[::10]
            self.test_set_o = self.all_one_hots[::10]
            self.all_windows = np.delete(self.all_windows, list(range(0, self.l, 10)), axis=0)
            self.all_one_hots = np.delete(self.all_one_hots, list(range(0, self.l, 10)), axis=0)

            np.savetxt(base_name + ".cnn_train_windows", self.all_windows)
            np.savetxt(base_name + ".cnn_train_one_hots", self.all_one_hots.astype(int))
            np.savetxt(base_name + ".cnn_test_windows", self.test_set)
            np.savetxt(base_name + ".cnn_test_one_hots", self.test_set_o.astype(int))
            print(base_name + " files written")
            
        self.index = 0
    
    def test_batches(self):
        return self.test_set, self.test_set_o
    
    def next_batches(self, batch_size, size):
        batches = []
        batches_o = []
        for x in range(batch_size):
            a, b = self.next_window(size)
            batches.append(a)
            batches_o.append(b)
        return batches, batches_o
    
    def next_window(self, size):
        windows = []
        windows_o = []
        rows_left = len(self.all_windows) - self.index
        first_pull = min(rows_left, size)
        windows.append(self.all_windows[self.index:(self.index + first_pull):1])
        windows_o.append(self.all_one_hots[self.index:(self.index + first_pull):1])
        self.index += first_pull

        if(self.index > len(self.all_windows)):
            print("self_index error!!")
        if(self.index == len(self.all_windows)):
            self.index = 0
            self.all_windows, self.all_one_hots = shuffle(self.all_windows, self.all_one_hots, random_state=0)
            print("shuffled")
        if(first_pull < size):
            second_pull = size - first_pull
            windows.append(self.all_windows[self.index:(self.index + second_pull):1])
            windows_o.append(self.all_one_hots[self.index:(self.index + second_pull):1])
            self.index += second_pull
        
        windows = np.reshape(windows, [-1])
        windows_o = np.reshape(windows_o, [-1])
                
        return windows, windows_o
    
    def __init__(self, config_file):
        configs = Configurations(config_file).configs
        self.prot_directory = configs["protein_directory"]
        self.output_directory = configs["output_directory"]
        self.name = configs["name"]
        self.learning_rate = configs["learning_rate"]
        self.batch_size = configs["batch_size"]
        self.steps = configs["max_steps"]
        self.keep_prob_val = configs["keep_prob"]
        self.momentum_val = configs["momentum"]
        