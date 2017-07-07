from collections import Counter
import sys

class Stats:

    def sov(self):
        p = self.pred[0]
        o = self.obs[0]
        i_p = 0
        j_p = 1
        i_o = 0
        j_o = 1
        while i_o < self.l:
            while self.obs[j_o] == o:
                j_o += 1
            
            while j_p < i_o:
                while self.pred[j_p] == p:
                    j_p += 1
                
                #check if overlap
                if j_p >= i_o: # overlap         
                    if p == o:
                        print("")
                
                p = self.pred[j_p]
                
            
            i_o = j_o
            j_o += 1
#    def q3_score(self):
        
        
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