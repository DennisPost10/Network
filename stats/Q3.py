from collections import Counter
import sys
import os.path

def read_structures(secondary_structure_observed, secondary_structure_predicted):
    ss_obs = open(secondary_structure_observed).read().strip()
    ss_pred = open(secondary_structure_predicted).read().strip()
    
    if len(ss_obs) - len(ss_pred) != 0:
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
    print(prots)
    
    output_matrix = open(output_matrix_file, 'w')
    output_matrix.write("prot\tlength\tnum_h\tnum_c\tnum_e\tpred_h\tpred_c\tpred_e\tcorrect\th_correct\tc_correct\te_correct\tmean_score\th_score\tc_score\te_score\tavg_score\n")
    
    for prot in prots:
        observed_ss_file = prot_directory + "/" + prot
        if observed_in_subdir == 1:
            observed_ss_file += "/" + prot
        observed_ss_file += ".ss_q3"
        
        predicted_ss_file = predicted_dir + "/" + prot
        if predicted_in_subdir == 1:
            predicted_ss_file += "/" + prot
        predicted_ss_file += ".secondary_structure"
        
        ss_obs, ss_pred, length = read_structures(observed_ss_file, predicted_ss_file)
        
        correct, correct_counts, counts_observed, counts_predicted, mean, h_score, c_score, e_score, avg = q3(ss_obs, ss_pred, length)
        
        output_matrix.write(prot + "\t" + str(length) + "\t" + str(counts_observed['H']) + "\t" + str(counts_observed['C']) + "\t" + str(counts_observed['E']) + "\t" + str(counts_predicted['H']) + "\t" + str(counts_predicted['C']) + "\t" + str(counts_predicted['E']) + "\t" + str(correct) + "\t" + str(correct_counts['H']) + "\t" + str(correct_counts['C']) + "\t" + str(correct_counts['E']) + "\t" + str(mean) + "\t" + str(h_score) + "\t" + str(c_score) + "\t" + str(e_score) + "\t" + str(avg) +"\n")
        
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
    h_score = correct_counts['H'] / counts_observed['H']
    c_score = correct_counts['C'] / counts_observed['C']
    e_score = correct_counts['E'] / counts_observed['E']
    avg = (h_score + c_score + e_score)/3

    return correct, correct_counts, counts_observed, counts_predicted, mean, h_score, c_score, e_score, avg

def sov(ss_obs, ss_pred, length):
    counts_observed = Counter(ss_obs)
    counts_predicted = Counter(ss_pred)
   
def main():
    args = sys.argv[1:]
    print(args)
    if len(args) < 7:
        print("Usage: prot_name_file, prot_directory, observed_in_subdir, predicted_dir, predicted_in_subdir, output_matrix_file, overwrite_matrix")
        sys.exit()
    analyze(args[0], args[1], args[2], args[3], args[4], args[5], args[6])
    
if __name__ == '__main__':
    main()