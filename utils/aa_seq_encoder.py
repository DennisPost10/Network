import sys
import numpy as np

def parse_aa_seq(aa_seq, aa_indices):
    aa_ind = np.array(len(aa_seq), dtype = int)
    aa_ind = aa_ind.fill(20)
    print(aa_seq)
    for i in range(len(aa_seq)):
        print(aa_indices[aa_seq[i]])
        aa_ind[i] = aa_indices[aa_seq[i]]
        
    print(aa_ind)
    return aa_ind

def read_aa_seq_file(aa_seq_file, aa_indices):
    
    aa_seq = ""
    with open(aa_seq_file) as aa:
        for line in aa:
            if not line.startswith(">"):
                aa_seq += line.strip()
    return parse_aa_seq(aa_seq, aa_indices)

def read_prot_file(prot_file, input_prot_dir, aa_indices):
    ret = []
    with open(prot_file) as prot_f:
        for line in prot_f: 
            line = line.strip()
            aa_seq = read_aa_seq_file(input_prot_dir + "/" + line +"/" + line + ".fasta", aa_indices)
            ret.append(aa_seq)
    return np.array(ret)
    
def get_encoded_aa_seq(prot_file, input_prot_dir, output_file, aa_indices):
    encoded_aa_seqs = read_prot_file(prot_file, input_prot_dir, aa_indices)
    np.save(output_file, encoded_aa_seqs)
    
def main(argv):
    if len(argv) < 3:
        print("prot_file, input_dir, output_file")
        sys.exit()
    prot_file = argv[0]
    input_prot_dir = argv[1]
    output_file = argv[2]
    print(argv)
    aa_indices = {"A":0, "R":1, "N":2, "D":3, "C":4, "Q":5, "E":6, "G":7, "H":8, "I":9, "L":10, "K":11, "M":12, "F":13, "P":14, "S":15, "T":16, "W":17, "Y":18, "V":19, "-":20}
    get_encoded_aa_seq(prot_file, input_prot_dir, output_file, aa_indices)
        
    
if __name__ == '__main__':
    main(sys.argv[1:])