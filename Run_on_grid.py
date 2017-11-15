import fileinput
import os
import shutil
import sys
import numpy as np
from utils.ConfigFileParser import Configurations
from mutable.Mutable_network import mutable_network

def iterate(config_file, index):
    configs = Configurations(config_file).configs
    if(configs == None):
            print("Error: no input")
            sys.exit()

    output_dir = configs.get("output_directory")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_files = configs.get("input_files")
    keys, inputs = parse_input_files(input_files)
    runs = configs.get("runs")
    
    filename = os.path.basename(config_file)
    name = configs.get("name")
    ss_features = configs.get("ss_features")
    window_sizes = [15, 30, 45, 60]
    conv_window_sizes = [5, 10, 15]
    out_channels = [100, 200]
    count = 0
    
    for a in range(2):
        for b in range(2):
            if a == 0 and b == 1:
                continue
            for i in window_sizes:
                use_aa_seq = (a == 1)
                single_aa_seq = (b == 1)
                window_size = i
                for j in conv_window_sizes:
                    for k in out_channels:
                        
                        out = output_dir + name + "_"
                        out += str(i) + "_" + str(j) + "_" + str(k) + "_"
                        out += str(a) + "_" + str(b) + "/"
                                
                        for l in inputs:
                            count += 1
                            if count < index:
                                continue
                            if count > index:
                                return

                            print(count)
                            #print(out)
                            
                            next_out = out + "/" + l[1] + "_" + l[0] + "/"
                            next_out = next_out.replace("//", "/")
                            os.makedirs(next_out)
                            shutil.copy(config_file, next_out)
                        
                            for line in fileinput.input([next_out + filename], inplace=True):
                                if line.strip().startswith("output_directory"):
                                    line = "output_directory\t" + next_out + "\tstr\n"
                                sys.stdout.write(line)

                            with open(next_out + filename, 'a') as conf:
                                conf.write("\n")
                                conf.write("window_size" + "\t" + str(window_size) + "\tint\n")
                                conf.write("use_aa_seq" + "\t" + str(use_aa_seq) + "\tbool\n")
                                conf.write("single_aa_seq" + "\t" + str(single_aa_seq) + "\tbool\n")
                        
                                conf.write("#layer\tlayer_" + str(1) + "\tconv\tTrue\t" + str(k) + "\t" + str(j) + "\n")
                                conf.write("#layer\tlayer_out\tfully\tFalse\t1\t" + str(ss_features) + "\n")
                                
                                for m in range(len(keys)):
                                    conf.write(keys[m] + "\t" + l[m] + "\tstr\n")
                                
                            for r in range(runs):
                                net = mutable_network(next_out + filename, r)
                                net.train(next_out + "run_" + str(r) + "/train.log")
                                net.predict(l[4], next_out + "run_" + str(r) + "/train.stats")

def parse_input_files(self, input_files):
        inputs = np.loadtxt(input_files, delimiter="\t", dtype=str)
        return inputs[0], inputs[1:]

def main(argv):
    config_file = argv[0]
    index = int(argv[1])
    print(config_file)
    iterate(config_file, index)

if __name__ == "__main__":
    main(sys.argv[1:])