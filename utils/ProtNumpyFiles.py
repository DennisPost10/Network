import numpy as np
import sys

def write_prot_mat(prot_file, ss_directory, tab_directory, output_file):
    prots = []
    with open(prot_file) as file:
        for line in file:
            line = line.strip()
            prots.append(line)
                        
    prot_matrices = []
    prot_outcomes = []
    for prot in prots:
        prot_matrices.append(np.loadtxt(tab_directory + "/" + prot + "/" + prot + ".tab", delimiter="\t", skiprows=1, usecols=range(2, 22), dtype=float))
        prot_outcomes.append(np.loadtxt(ss_directory + "/" + prot + "/" + prot + ".ss_one_hot", dtype = int, delimiter = "\t"))
        
        
    print(prot_matrices.shape)
    print(prot_outcomes.shape)
    np.save(output_file + ".matrix", prot_matrices)
    np.save(output_file + ".one_hots", prot_outcomes)

def check(matrix, one_hots):
    data=np.load(matrix)
    print(data.shape)
    one_hots=np.load(one_hots)
    print(one_hots.shape)
    
def main(argv):
    if len(argv) < 3:
        print("prot_name_file, ss_directory, tab_directory, output_file")
        sys.exit()
    prot_name_file = argv[0]
    ss_directory = argv[1]
    tab_directory = argv[1]
    output_file = argv[2]
    write_prot_mat(prot_name_file, ss_directory, tab_directory, output_file)
    check(output_file + ".matrix.npy", output_file + ".one_hots.npy")

if __name__ == "__main__":
    main(sys.argv[1:])