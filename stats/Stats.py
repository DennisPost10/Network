from collections import Counter
import sys

class Stats:

    def sov(self):
        o = ''
        i_o = 0
        j_o = 1
        p = ''
        i_p = 0
        j_p = -1
        print(self.obs)
        print(self.pred)
        while j_o != -1:
            o, i_o, j_o = self.next_interval(self.obs, i_o)
            print(str(i_o) + " "  + str(j_o) + " " + str(i_p) + " " + str(j_p))
            print(o + " " + p)

            while j_p < j_o or j_p == -1:
                while j_p <= i_o:
                    p, i_p, j_p = self.next_interval(self.pred, i_p)
                    print(str(i_o) + " "  + str(j_o) + " " + str(i_p) + " " + str(j_p))
                    print(o + " " + p)
            if o == p:
                low = max(i_o, i_p)
                high = min(j_o, j_p)
                overlap = high - low
                sov = self.sov_val(j_o - i_o, j_p - i_p, overlap)
                print(sov)
                print(str(i_o) + " "  + str(j_o) + " " + str(i_p) + " " + str(j_p) + " " + str(overlap))
            i_o = j_o
                
    def next_interval(self, ss, lower):
        l = len(ss)
        if l <= lower:
            return None, -1, -1
        c = ss[lower]
        upper = lower + 1
        while(upper < l and ss[upper] == c):
            upper += 1
        return c, lower, upper

    def sov_val(self, size_obs, size_pred, overlap):
        ret = overlap + self.sov_delta(size_obs, size_pred, overlap)
        maxOV = (size_obs + size_pred - overlap)*1.0
        return ret/maxOV
    
    def sov_delta(self, size_obs, size_pred, overlap):
        ret = min((size_obs+size_pred-overlap), overlap)
        ret = min(ret, overlap)
        ret = min(ret, int(0.5*size_obs))
        return min(ret, int(0.5*size_pred))
    
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
        
if __name__ == '__main__':
    ss1 = "CCCCCCCCCHHHHHHHHHCCEEEECCC"
    ss2 = "EECCCCCHHHHHCCHHHCCCEEECCCC"
    s = Stats(ss1, ss2)
    s.sov()