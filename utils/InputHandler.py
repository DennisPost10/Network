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
                    for k in range(len(data[i][j])):
                        data[i][j][k] = 1.0 / (1.0 + math.exp(data[i][j][k]))
#        elif data_normalization_function == "":
      
#        elif data_normalization_function == "":
            
        return data
        
    def reformat_data(self, data, ss_data, aa_seq_data, indices, features, ss_features, max_length, window_size):
        data = np.take(data, indices, 0)
        ss_data = np.take(ss_data, indices, 0)
        self.aa_seq_add = 0
        if self.load_aa_seq:
            aa_seq_data = np.take(aa_seq_data, indices, 0)
            self.aa_seq_add = self.aa_codes
        if self.network_type == "conv":
            full_data = np.zeros(shape=[data.shape[0], max_length, features + self.aa_seq_add], dtype=data.dtype)
            full_ss = np.zeros(shape=[ss_data.shape[0], max_length, ss_features], dtype=ss_data.dtype)
            for i in range(len(data)):
                full_data[i][:data[i].shape[0], :features] = data[i]
                full_ss[i][:ss_data[i].shape[0], :ss_features] = ss_data[i]
                if self.load_aa_seq:
                    for j in range(len(data[i])):
                        full_data[i][j][features + aa_seq_data[i][j]] = 1
            #print(data.shape)
        
        else:
            self.h_w = int(window_size / 2)
            full_data = []
            for i in range(len(data)):
                a = data[i]
                b = np.zeros(shape=[a.shape[0] + 2 * self.h_w, a.shape[1] + self.aa_seq_add], dtype=data.dtype)
                b[self.h_w:a.shape[0] + self.h_w, :a.shape[1]] = a
                if self.load_aa_seq:
                    for j in range(len(a)):
                        b[self.h_w + j][features + aa_seq_data[i][j]] = 1
                full_data.append(b)
            full_data = np.array(full_data)
            full_ss = ss_data 

        return full_data, full_ss

    def get_lengths(self, data, index_vec, max_length):
        # returns everything if max_length <= 0
        lengths = []
        too_big = []
        for i in range(len(index_vec)):
            if max_length > 0 and self.network_type == "conv" and len(data[index_vec[i]]) > max_length:
                too_big.append(i)
            else:
                lengths.append(len(data[index_vec[i]]))
        return np.array(lengths), np.array(too_big)
    
    def load_data(self):
        
        self.dat = self.read_npy_file(self.pssm_input_matrix)
        self.dat = self.normalize_data(self.dat, self.data_normalization_function)
        #print(self.dat.shape)
        self.ss_dat = self.read_npy_file(self.one_hots_matrix)

        self.features = self.dat[0].shape[1]
        #print(self.features)
        self.ss_features = self.ss_dat[0].shape[1]
        #print(self.ss_features)

        self.aa_seq = None
        if self.load_aa_seq:
            self.aa_seq = self.read_npy_file(self.aa_seq_matrix)
    
    def load_single_file(self, single_file):

        single_dat = np.loadtxt(single_file, delimiter="\t", usecols=1, dtype=int)
        single_prot_names = np.loadtxt(single_file, delimiter="\t", usecols=0, dtype=str)
        single_lengths, single_too_big = self.get_lengths(self.dat, single_dat, self.max_prot_length)
        single_dat = np.delete(single_dat, single_too_big, 0)
        
        single_dat, ss_single_dat = self.reformat_data(self.dat, self.ss_dat, self.aa_seq, single_dat, self.features, self.ss_features, self.max_prot_length, self.window_size)
    
        return single_dat, ss_single_dat, single_lengths, single_prot_names
    
    
    def load_ttv(self):
        self.train_dat, self.ss_train_dat, self.train_lengths, self.train_names = self.load_single_file(self.train_file)
        self.val_dat, self.ss_val_dat, self.val_lengths, self.val_names = self.load_single_file(self.val_file)
        self.test_dat, self.ss_test_dat, self.test_lengths, self.test_names = self.load_single_file(self.test_file)

        self.val_dat, self.ss_val_dat = self.parse_batch(self.val_dat, self.ss_val_dat, self.val_lengths)
        self.test_dat, self.ss_test_dat = self.parse_batch(self.test_dat, self.ss_test_dat, self.test_lengths)
    
        self.prot_index = 0
        self.random_prot_rank = np.random.permutation(len(self.train_dat))
        self.index = 0
    
        self.ttv_loaded = True
    
    def parse_batch(self, dat, ss_dat, lengths):
        if self.network_type == "conv":
            return dat, ss_dat
        
        parsed_dat = []
        parsed_ss_dat = []
        
        for i in range(len(dat)):
            for j in range(lengths[i]):
                parsed_ss_dat.append(ss_dat[i][j])
        
        if self.network_type == "mixed":
            for i in range(len(dat)):
                for j in range(lengths[i]):
                    parsed_dat.append(dat[i][j:(j + self.window_size)])
                    
        else:
            for i in range(len(dat)):
                for j in range(lengths[i]):
                    if self.single_aa_seq:
                        window = np.empty(shape=[self.window_size * self.features + self.aa_seq_add], dtype=dat.dtype)
                        window[0:self.window_size * self.features] = dat[i][j:(j + self.window_size), 0:self.features].reshape(-1)
                        window[self.window_size * self.features:] = dat[i][j + self.h_w][self.features:]
                        parsed_dat.append(window)
                    else:
                        parsed_dat.append(dat[i][j:(j + self.window_size)].reshape(-1))
            
        return np.array(parsed_dat), np.array(parsed_ss_dat)        
    
    def val_batches(self):
        if not self.ttv_loaded:
            self.load_ttv()
        return self.val_dat, self.ss_val_dat, self.val_lengths
    
    def test_batches(self):
        if not self.ttv_loaded:
            self.load_ttv()
        return self.test_dat, self.ss_test_dat, self.test_lengths
            
    def next_prots(self, size):
        if not self.ttv_loaded:
            self.load_ttv()
        if self.network_type == "conv":
            return self.next_prots_cnn(size)
        elif self.network_type == "mixed":
            return self.next_prots_mixed(size)
        return self.next_prots_not_cnn(size)
    
    def next_prots_cnn(self, size):
        total_pull = 0
        ret_indices = np.empty(size, dtype=int)
        
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
        if self.single_aa_seq:
            ret_windows = np.empty(shape=[size, self.window_size * self.features + self.aa_seq_add], dtype=self.train_dat.dtype)
        else:
            ret_windows = np.empty(shape=[size, self.window_size * (self.features + self.aa_seq_add)], dtype=self.train_dat.dtype)
        ret_ss = np.empty(shape=[size, self.ss_features], dtype=self.ss_train_dat.dtype)

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
                if self.single_aa_seq:
                    window = np.empty(shape=[self.window_size * self.features + self.aa_seq_add], dtype=self.train_dat.dtype)
                    window[0:self.window_size * self.features] = self.train_dat[self.prot_index][self.index:(self.index + self.window_size), 0:self.features].reshape(-1)
                    window[self.window_size * self.features:] = self.train_dat[self.prot_index][self.index + self.h_w, self.features:]
                    ret_windows[i + total_pull] = window
                else:
                    ret_windows[i + total_pull] = self.train_dat[self.prot_index][self.index:(self.index + self.window_size)].reshape(-1)
                ret_ss[i + total_pull] = self.ss_train_dat[self.prot_index][self.index]
                self.index += 1
            total_pull += next_pull
     
        return ret_windows, ret_ss, []
    
    def next_prots_mixed(self, size):
        
        total_pull = 0
        ret_windows = np.empty(shape=[size, self.window_size, self.features + self.aa_seq_add], dtype=self.train_dat.dtype)
        ret_ss = np.empty(shape=[size, self.ss_features], dtype=self.ss_train_dat.dtype)

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
                ret_windows[i + total_pull] = self.train_dat[self.prot_index][self.index:(self.index + self.window_size)]
                ret_ss[i + total_pull] = self.ss_train_dat[self.prot_index][self.index]
                self.index += 1
            total_pull += next_pull
  
        return ret_windows, ret_ss, []
    
    def get_prot_by_prot(self, test_file):
        dat, ss_dat, lengths, prot_names = self.load_single_file(test_file)
        ret_dat = []
        ret_ss_dat = []
        #parse each prot to one batch and append it to array of prot_batches
        for i in range(len(dat)):
            prot_dat, prot_ss_dat = self.parse_batch([dat[i]], [ss_dat[i]], [lengths[i]])
            ret_dat.append(prot_dat)
            ret_ss_dat.append(prot_ss_dat)
        return np.array(ret_dat), np.array(ret_ss_dat), lengths, prot_names
        
    
    def __init__(self, prot_set, pssm_matrix, aa_seqs_matrix, one_hots_matrix, index, train, val, test, max_prot_length, network_type, window_size, data_normalization_function, load_aa_seq=False, aa_codes=23, single_aa_seq=False):
        self.prot_set = prot_set
        self.pssm_input_matrix = pssm_matrix
        self.aa_seq_matrix = aa_seqs_matrix
        self.one_hots_matrix = one_hots_matrix
        self.index = index  # index of ttv_file: e.g. 2 for train2, test2, validation2 etc.
        self.train_file = train
        self.val_file = val
        self.test_file = test
        self.max_prot_length = max_prot_length
        self.network_type = network_type
        self.window_size = window_size
        self.data_normalization_function = data_normalization_function
        self.load_aa_seq = load_aa_seq
        self.aa_codes = aa_codes
        self.single_aa_seq = single_aa_seq
        if not self.load_aa_seq:
            self.single_aa_seq = False
        self.load_data()
        self.ttv_loaded = False
    
def main(argv):
    # netw = Input_Handler("D:/Dennis/Uni/bachelor/data/prot_data/cull_prot_name_files/dat_files/", "cull_pdb_prots", "D:/Dennis/Uni/bachelor/data/prot_data/cull_prot_name_files/dat_files/", 1, 700, "conv", -1, "nothing")
    netw = Input_Handler("/home/proj/tmp/postd/prot_data/psi_prot_name_files/dat_files/", "psi_prots", "/home/proj/tmp/postd/prot_data/psi_prot_name_files/dat_files/", 1, 700, "a", 20, "standard", load_aa_seq=True, single_aa_seq=True)
#    print(netw.val_batches())
#    print(netw.test_batches())
    for i in range(1):
        a, b, c = netw.next_prots(1)
        print(a)
        print(c)
        
if __name__ == "__main__":
    main(sys.argv[1:])
        
