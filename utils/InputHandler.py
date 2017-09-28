import math
import os
import sys

import numpy as np

class Input_Handler:
    
    def read_npy_file(self, npy_file):
        return np.load(npy_file)
      
    def normalize_data(self, data, data_normalization_function):
        if data_normalization_function == "standard":
            for i in range(len(data)):
                    for j in range(len(data[i])):
                        data[i][j] = 1.0 / (1.0 + math.exp(data[i][j]))
#        elif data_normalization_function == "":
      
#        elif data_normalization_function == "":
            
        else:
            return data
            
    def reformat_data(self, data, ss_data, indices, features, ss_features, max_length, window_size, cnn_bool):
        data = np.take(data, indices, 0)
        ss_data = np.take(ss_data, indices, 0)
        if cnn_bool:
            full_data = np.zeros(shape=[data.shape[0], max_length, features], dtype = data.dtype)
            full_ss = np.zeros(shape=[ss_data.shape[0], max_length, ss_features], dtype = ss_data.dtype)
            for i in range(len(data)):
                full_data[i][:data[i].shape[0], :features] = data[i]
                full_ss[i][:ss_data[i].shape[0], :ss_features] = ss_data[i]
            print(data.shape)
        else:
            h_w = int(window_size / 2)
            full_data = []
            for i in range(len(data)):
                a = data[i]
                b = np.zeros(shape=[a.shape[0] + 2 * h_w, a.shape[1]], dtype = data.dtype)            
                b[h_w:a.shape[0] + h_w, :a.shape[1]] = a
                full_data.append(b)
            full_data = np.array(full_data)
            full_ss = ss_data
        return full_data, full_ss

    def get_lengths(self, data, index_vec, max_length, cnn_bool):
        # returns everything if max_length <= 0
        lengths = []
        too_big = []
        for i in range(len(index_vec)):
            if max_length > 0 and cnn_bool and len(data[index_vec[i]]) > max_length:
                too_big.append(i)
            else:
                lengths.append(len(data[index_vec[i]]))
        return np.array(lengths), np.array(too_big)
        
    def load_data(self):
        
        self.dat = self.read_npy_file(self.input_directory + "/" + self.file_base + ".matrix.npy")
        self.dat = self.normalize_data(self.dat, self.data_normalization_function)
        self.ss_dat = self.read_npy_file(self.input_directory + "/" + self.file_base + ".one_hots.npy")

        self.train = np.loadtxt(self.ttv_file + "train" + str(self.index) + ".lst", delimiter = "\t", usecols = 1, dtype = int)
        self.val = np.loadtxt(self.ttv_file + "validation" + str(self.index) + ".lst", delimiter = "\t", usecols = 1, dtype = int)
        self.test = np.loadtxt(self.ttv_file + "test" + str(self.index) + ".lst", delimiter = "\t", usecols = 1, dtype = int)

        self.train_lengths, train_too_big = self.get_lengths(self.dat, self.train, self.max_prot_length, self.cnn_bool)
        self.val_lengths, val_too_big = self.get_lengths(self.dat, self.val, self.max_prot_length, self.cnn_bool)
        self.test_lengths, test_too_big = self.get_lengths(self.dat, self.test, self.max_prot_length, self.cnn_bool)       

        self.train = np.delete(self.train, train_too_big, 0)
        self.val = np.delete(self.val, val_too_big, 0)
        self.test = np.delete(self.test, test_too_big, 0)

        self.features = self.dat[0].shape[1]
        print(self.features)
        self.ss_features = self.ss_dat[0].shape[1]
        print(self.ss_features)

        self.train_dat, self.ss_train_dat = self.reformat_data(self.dat, self.ss_dat, self.train, self.features, self.ss_features, self.max_prot_length, self.window_size, self.cnn_bool)
        self.val_dat, self.ss_val_dat = self.reformat_data(self.dat, self.ss_dat, self.val, self.features, self.ss_features, self.max_prot_length, self.window_size, self.cnn_bool)
        self.test_dat, self.ss_test_dat = self.reformat_data(self.dat, self.ss_dat, self.test, self.features, self.ss_features, self.max_prot_length, self.window_size, self.cnn_bool)
        
#        self.train_dat = np.take(self.dat, self.train, 0)
#        self.val_dat = np.take(self.dat, self.val, 0)
#        self.test_dat = np.take(self.dat, self.test, 0)
        
#        self.ss_train_dat = np.take(self.ss_dat, self.train, 0)
#        self.ss_val_dat = np.take(self.ss_dat, self.val, 0)
#        self.ss_test_dat = np.take(self.ss_dat, self.test, 0)
        
        self.prot_index = 0
        self.random_prot_rank = np.random.permutation(len(self.train_dat))
        self.index = 0
    
    def val_batches(self):
        return self.val_dat, self.ss_val_dat, self.val_lengths
    
    def test_batches(self):
        return self.test_dat, self.ss_test_dat, self.test_lengths
    
    def next_prots(self, size):
        if self.cnn_bool:
            return self.next_prots_cnn(size)
        return self.next_prots_not_cnn(size)
    
    def next_prots_cnn(self, size):
        total_pull = 0
        ret_indices = np.empty(size, dtype = int)
        
        while total_pull < size:
            if self.prot_index == len(self.train_dat):
                self.prot_index = 0
                self.random_prot_rank = np.random.permutation(len(self.train_dat))

            rows_left = len(self.train_dat) - self.prot_index
            next_pull = min(rows_left, size - total_pull)
            ret_indices[total_pull:total_pull + next_pull] = self.random_prot_rank[self.prot_index:self.prot_index + next_pull]
            
            self.prot_index += next_pull
            total_pull += next_pull
        return np.take(self.train_dat, ret_indices, 0), np.take(self.ss_train_dat, ret_indices, 0), np.take(self.train_lengths, ret_indices, 0)
        
    def next_prots_not_cnn(self, size):
        
        total_pull = 0
        ret_windows = np.empty(shape=[size, self.window_size * self.features], dtype = self.train_dat.dtype)
        ret_ss = np.empty(shape=[size, self.ss_features], dtype = self.ss_train_dat.dtype)

        while total_pull < size:
            if self.index == self.train_lengths[self.prot_index]:
                self.index = 0
                self.prot_index += 1
                if self.prot_index == len(self.train_dat):
                    self.prot_index = 0
                    self.random_prot_rank = np.random.permutation(len(self.train_dat))

            rows_left = self.train_lengths[self.prot_index] - self.index
            next_pull = min(rows_left, size - total_pull)
            for i in range(next_pull):
                ret_windows[i + total_pull] = self.train_dat[self.prot_index][self.index:(self.index + self.window_size)].reshape(-1)
                ret_ss[i + total_pull] = self.ss_train_dat[self.prot_index][self.index]
                self.index += 1
            total_pull += next_pull

#            if total_pull >= size:
#                break
            
        return ret_windows, ret_ss, None
    
    def __init__(self, input_directory, data_file_base_name, ttv_file, index, max_prot_length, cnn_bool, window_size, data_normalization_function, prot_name_file = None):
        self.input_directory = input_directory # dir where input npys are
        self.file_base = data_file_base_name # "psi_prots"|"cull_pdb" -> .matrix.npy and .one_hots.npy will be appended
        self.ttv_file = ttv_file # ttv_file of ttvs: "ttv_dir + psi_" + validation --> ttv_file = "psi_"
        self.index = index # index of ttv_file: e.g. 2 for train2, test2, validation2 etc.
        self.max_prot_length = max_prot_length
        self.cnn_bool = cnn_bool
        self.window_size = window_size
        self.data_normalization_function = data_normalization_function
        self.load_data()
    
def main(argv):
    netw = Input_Handler("D:/Dennis/Uni/bachelor/data/prot_data/cull_prot_name_files/dat_files/", "cull_pdb_prots", "D:/Dennis/Uni/bachelor/data/prot_data/cull_prot_name_files/dat_files/", 1, 700, True, -1, "nothing")
#    print(netw.val_batches())
#    print(netw.test_batches())
    for i in range(1000):
        netw.next_prots(50)

if __name__ == "__main__":
    main(sys.argv[1:])
        