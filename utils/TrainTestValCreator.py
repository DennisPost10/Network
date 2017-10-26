import sys

import numpy as np

def read_prot_file(file):
    return np.loadtxt(file, dtype=str, delimiter="\n")

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
    
    
def split_data(data, sets, validation_factor, output_directory, prefix=""):
    a = np.arange(len(data))
    i = np.random.permutation(len(data))
    print(data)
    # split into test sets (indices)
    test_sets = np.array_split(i, sets)
#    print(test_sets_indices)
#    test_sets = []
    train_sets = []
    validation_sets = []
    
    for j in range(sets):
        print(j)
        # (data)
        test_sets[j] = np.sort(test_sets[j])
        train_sets.append(np.delete(a, test_sets[j]))
        # (random indices)
        i = np.random.permutation(len(train_sets[j]))
        i = i[0:int(len(i) / validation_factor)]
        # (data)
        validation_sets.append(np.sort(np.take(train_sets[j], i)))
        train_sets[j] = np.sort(np.delete(train_sets[j], i))
#        test_sets.append(np.take(data, test_sets_indices[j]))
        write_files(j + 1, data, test_sets[j], train_sets[j], validation_sets[j], output_directory, prefix)            
    
def main(argv):
    prot_name_file = argv[0]
    output_directory = argv[1]
    prefix = argv[2]
    if prefix == "None":
        prefix = ""
    prots = read_prot_file(prot_name_file)
    split_data(prots, int(argv[3]), int(argv[4]), output_directory, prefix)

if __name__ == "__main__":
    main(sys.argv[1:])
