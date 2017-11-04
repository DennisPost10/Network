import os
import sys

import numpy as np


def read_prot_file(prot_file, file_directory, output_directory, validation_factor):
    prots = []
    with open(prot_file) as prot_f:
        for line in prot_f: 
            line = line.strip()
            prots.append(line)
            
    for file in os.listdir(file_directory):
        if file.endswith(".lst"):
            print(file)
            i = os.path.join(file_directory, file)
            o = os.path.join(output_directory, file)
            indices = []            
            with open(i) as input_f:
                for line in input_f:
                    line = line.strip().split("\t")[0]
                    indices.append(prots.index(line))
                indices = np.sort(indices)
            if "train" in file:
                a = np.random.permutation(len(indices))
                a = a[0:int(len(a) / validation_factor)]
                val = np.sort(np.take(indices, a))
                indices = np.sort(np.delete(indices, a))
                with open(os.path. join(output_directory, file.replace("train", "validation")), "w") as output:
                    for j in val:
                        output.write(prots[j] + "\t" + str(j) + "\n")
            
            with open(o, "w") as output:
                for j in indices:
                    output.write(prots[j] + "\t" + str(j) + "\n")

def write_files(index, data, test_set, train_set, validation_set, output_directory, prefix=""):
    with open(output_directory + "/" + prefix + "test" + str(index) + ".lst", "w") as t:
        for p in test_set:
            t.write(data[p] + "\t" + str(p) + "\n")
    with open(output_directory + "/" + prefix + "train" + str(index) + ".lst", "w") as t:
        for p in train_set:
            t.write(data[p] + "\t" + str(p) + "\n")
    with open(output_directory + "/" + prefix + "validation" + str(index) + ".lst", "w") as t:
        for p in validation_set:
            t.write(data[p] + "\t" + str(p) + "\n")
    
def main(argv):
    prot_name_file = "C:/Users/Dennis/Desktop/data/psi_prot_name_files/all_protein_names"
    output_directory = "C:/Users/Dennis/Desktop/data/psi_prot_name_files/dat_files/"
    input_directory = "C:/Users/Dennis/Desktop/data/psi_prot_name_files/"
    read_prot_file(prot_name_file, input_directory, output_directory, 10)

if __name__ == "__main__":
    main(sys.argv[1:])
