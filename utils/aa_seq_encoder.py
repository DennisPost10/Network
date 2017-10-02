import sys
import numpy as np

aa_indices = {"A":0, "R":1, "N":2, "D":3, "C":4, "Q":5, "E":6, "G":7, "H":8, "I":9, "L":10, "K":11, "M":12, "F":13, "P":14, "S":15, "T":16, "W":17, "Y":18, "V":19, "-":20}

def parse_aa_seq(aa_seq):
    aa_ind = np.array(len(aa_seq), dtype = int)
    aa_ind = aa_ind.fill(20)
    for i in range(len(aa_seq)):
        aa_ind[i] = aa_indices[aa_seq[i]]
        
    return aa_ind

def read_aa_seq_file(aa_seq_file):
    
    aa_seq = ""
    with open(aa_seq_file) as aa:
        for line in aa:
            if line.startswith(">"):
                aa_seq += line.strip()
    return parse_aa_seq(aa_seq)

def read_prot_file(prot_file, input_prot_dir):
    ret = []
    with open(prot_file) as prot_f:
        for line in prot_f: 
            line = line.strip()
            aa_seq = read_aa_seq_file(input_prot_dir + "/" + line +"/" + line + ".fasta")
            ret.append(aa_seq)
    return np.array(ret)
    
def get_encoded_aa_seq(prot_file, input_prot_dir, output_file):
    encoded_aa_seqs = read_prot_file(prot_file, input_prot_dir)
    np.save(output_file, encoded_aa_seqs)
    
def main(argv):
    if len(argv) < 3:
        print("prot_file, input_dir, output_file")
        sys.exit()
    prot_file = argv[0]
    input_prot_dir = argv[1]
    output_file = argv[2]
    print(argv)
    get_encoded_aa_seq(prot_file, input_prot_dir, output_file)
        
    
if __name__ == '__main__':
    main(sys.argv[1:])