from collections import Counter
import sys

class Stats:

    def sov(self):
        len_o = 1
        len_p = 1
        p = self.pred[0]
        o = self.obs[0]
        i = 1
        j = 1
        while i < self.l:
            while self.obs[i] == o:
                len_o += 1
                i += 1
            len_o = 1

    def q3_score(self):
        
        
    def count_occs(self):
        self.counts = Counter(self.obs)
        self.h_count = self.counts['H']
        self.c_count = self.counts['C']
        self.e_count = self.counts['E']
        
    def __init__(self, pred, obs):
        if(len(pred) != len(obs)):
            print("different length!")
            print(pred)
            print(obs)
            sys.exit()
        self.pred = pred
        self.obs = obs
        self.l = len(pred)
        self.count_occs()