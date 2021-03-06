from collections import Counter
import os.path
import subprocess
import sys

import matplotlib


# from plotting.stat_matrix_reader_pandas import stats_plotter
def read_structures(prot_directory, prot, observed_in_subdir, predicted_dir, predicted_in_subdir):
    
    observed_ss_file = prot_directory + "/" + prot
    if observed_in_subdir == 1:
        observed_ss_file += "/" + prot
    observed_ss_file += ".ss_q3"
        
    predicted_ss_file = predicted_dir + "/" + prot
    if predicted_in_subdir == 1:
        predicted_ss_file += "/" + prot
    predicted_ss_file += ".secondary_structure"
    
    ss_obs = open(observed_ss_file).read().strip()
    ss_pred = open(predicted_ss_file).read().strip()
    
    if len(ss_obs) != len(ss_pred):
        print("Error different lengths!")
        print("Checking for - in fasta...")
        aa_sequence_file = prot_directory + "/" + prot
        if observed_in_subdir == 1:
            aa_sequence_file += "/" + prot
#        aa_sequence_file += ".aa_sequence"
        aa_sequence_file += ".fasta"
#        aa_sequence = open(aa_sequence_file).read().replace("\n", "")
        aa_sequence = ""
        with open(aa_sequence_file) as fasta:
            fasta.readline()
            line = fasta.readline()
            while(line):
                aa_sequence += line.strip()
                line = fasta.readline()
            
        unknown = []
        for i in range(len(aa_sequence)):
            if aa_sequence[i] == '-':
                unknown.append(i - len(unknown))
        for i in unknown:
            ss_obs = ss_obs[:i] + ss_obs[i + 1:]
        if len(ss_obs) != len(ss_pred):
            print("Error different lengths!")
            print(ss_obs)
            print(ss_pred)
            sys.exit()
    
    l = len(ss_obs) 
    
    return ss_obs, ss_pred, l

def analyze(prot_name_file, prot_directory, observed_in_subdir, predicted_dir, predicted_in_subdir, output_matrix_file, overwrite_matrix):
    if os.path.exists(output_matrix_file) and not overwrite_matrix == 1:
        print("WARNING: overwriting older output_matrix")
        print("Execution interrupted")
        sys.exit()
        
    with open(prot_name_file) as f:
        prots = f.read().split("\n")
        if prots[len(prots) - 1] == "":
            prots = prots[:-1]
    print(prots)
    
    output_matrix = open(output_matrix_file, 'w')
    output_matrix.write("prot\tlength\tnum_h\tnum_c\tnum_e\tpred_h\tpred_c\tpred_e\tcorrect\th_correct\tc_correct\te_correct\tmean_score\th_score\tc_score\te_score\tavg_score\tsov\th_sov\tc_sov\te_sov\n")
    
    for prot in prots:
        
        ss_obs, ss_pred, length = read_structures(prot_directory, prot, observed_in_subdir, predicted_dir, predicted_in_subdir)
        
        correct, correct_counts, counts_observed, counts_predicted, mean, h_score, c_score, e_score, avg = q3(ss_obs, ss_pred, length)
        
        h_sov, c_sov, e_sov, sov3 = sov(ss_obs, ss_pred, length)
#        sov3 = sov(ss_obs, ss_pred, length)
        print("Q3: " + str(mean) + "\tSOV: " + str(sov3))
        output_matrix.write(prot + "\t" + str(length) + "\t" + str(counts_observed['H']) + "\t" + str(counts_observed['C']) + "\t" + str(counts_observed['E']) + "\t" + str(counts_predicted['H']) + "\t" + str(counts_predicted['C']) + "\t" + str(counts_predicted['E']) + "\t" + str(correct) + "\t" + str(correct_counts['H']) + "\t" + str(correct_counts['C']) + "\t" + str(correct_counts['E']) + "\t" + str(mean) + "\t" + str(h_score) + "\t" + str(c_score) + "\t" + str(e_score) + "\t" + str(avg) + "\t" + str(sov3) + "\t" + str(h_sov) + "\t" + str(c_sov) + "\t" + str(e_sov) + "\n")
        
    output_matrix.close()

def q3(ss_obs, ss_pred, length):    
    counts_observed = Counter(ss_obs)
    counts_predicted = Counter(ss_pred)
    
    correct = 0
    correct_counts = {'H': 0, 'C': 0, 'E': 0}
    for i in range(length):
        if ss_obs[i] == ss_pred[i]:
            correct += 1
            correct_counts[ss_obs[i]] += 1
    
    mean = correct / length
    i = 0
    if counts_observed['H'] > 0:
        h_score = correct_counts['H'] / counts_observed['H']
        i += 1
    else:
        h_score = -1
    
    if counts_observed['C'] > 0:
        c_score = correct_counts['C'] / counts_observed['C']
        i += 1
    else:
        c_score = -1
        
    if counts_observed['E'] > 0:
        e_score = correct_counts['E'] / counts_observed['E']
        i += 1
    else:
        e_score = -1
        
#    avg = (h_score + c_score + e_score)/i
    avg = (max(h_score, 0) + max(c_score, 0) + max(e_score, 0)) / i

    return correct, correct_counts, counts_observed, counts_predicted, mean, h_score, c_score, e_score, avg

def sov(obs, pred, length):
    sov_arr = {'H': [], 'C': [], 'E': []}
    non_match = {'H': 0, 'C': 0, 'E': 0}
    non_match_size = 0
    i = 0
    while i < length:
        c = obs[i]
        s = i
        while i < length - 1 and obs[i + 1] == c:
                i += 1
        match = False
        for x in range(s, i + 1):
            if pred[x] == c:
                match = True
                break
        if not match:
            non_match[c] += i + 1 - s
            non_match_size += i + 1 - s
        i += 1
#    print(non_match)
        
    counts_observed = Counter(obs)
#    print(obs)
#    print(pred)
    i = 0
    weighted_sov_sum = 0.0
    normalization_factor = 0
    normalization_factor_arr = {'H': 0, 'C': 0, 'E': 0}
    while i < length:
        if obs[i] == pred[i]:
            j_o = i
            i_o = i
            j_p = i
            i_p = i
            while i_o > 0 and obs[i_o - 1] == obs[i]:
                i_o -= 1
            while j_o < length - 1 and obs[j_o + 1] == obs[i]:
                j_o += 1
            while i_p > 0 and pred[i_p - 1] == pred[i]:
                i_p -= 1
            while j_p < length - 1 and pred[j_p + 1] == pred[i]:
                j_p += 1
            j_o += 1
            j_p += 1
                
            low = max(i_o, i_p)
            high = min(j_o, j_p)
            overlap = high - low
            o_len = j_o - i_o
            sov = sov_val(o_len, j_p - i_p, overlap)
            weighted_sov_sum += sov * (o_len)
            normalization_factor += o_len
            normalization_factor_arr[obs[i]] += o_len            
            sov_arr[obs[i]].append(sov * o_len)
#            print(str(i_o) + " "  + str(j_o) + " " + str(i_p) + " " + str(j_p) + " " + str(overlap) + ": " + str(sov)) 
#            i = min(j_o, j_p)
            i = j_p
        else:
            i += 1
#    print(sov_arr)
    if counts_observed['H'] > 0:
        h_sov = sum(sov_arr['H']) / (non_match['H'] + normalization_factor_arr['H'])
    else:
        h_sov = -1
    if counts_observed['C'] > 0:
        c_sov = sum(sov_arr['C']) / (non_match['C'] + normalization_factor_arr['C'])
    else:
        c_sov = -1
    if counts_observed['E'] > 0:
        e_sov = sum(sov_arr['E']) / (non_match['E'] + normalization_factor_arr['E'])
    else:
        e_sov = -1
#    print(h_sov)
#    print(c_sov)
#    print(e_sov)
#    print(sov_arr)
    sov3 = weighted_sov_sum / (normalization_factor + non_match_size)
#    print(sov3)
    return h_sov, c_sov, e_sov, sov3
#    return sov3
    
def sov_val(size_obs, size_pred, overlap):
    maxOV = (size_obs + size_pred - overlap)
    ret = overlap + sov_delta(size_obs, size_pred, overlap, maxOV) * 1.0
    return ret / maxOV
    
def sov_delta(size_obs, size_pred, overlap, maxOV):
    ret = min(maxOV - overlap, overlap)
    ret = min(ret, int(0.5 * size_obs))
#    return ret
    return min(ret, int(0.5 * size_pred))
      
def main():
    args = sys.argv[1:]
    print(args)
    if len(args) < 7:
        print("Usage: prot_name_file, prot_directory, observed_in_subdir, predicted_dir, predicted_in_subdir, output_matrix_file, overwrite_matrix")
        sys.exit()
    analyze(args[0], args[1], int(args[2]), args[3], int(args[4]), args[5], int(args[6]))
    output_dir = os.path.splitext(args[5])[0]
#    stat_plot = stats_plotter(args[5], output_dir)
#    stat_plot.plot_all_stats()
    
if __name__ == '__main__':
    main()
