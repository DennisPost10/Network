import os 
import sys
import shutil
import numpy as np
import fileinput
from utils.ConfigFileParser import Configurations
from mutable_network import mutable_network

class wrapper:
     
    def __init__(self, config_file):
        configs = Configurations(config_file).configs
        filename = os.path.basename(config_file)
        input_files = configs.get("input_files")
        output_dir = configs.get("output_directory")
        runs = configs.get("runs")
        keys, inputs = self.parse_input_files(input_files)
        for i in inputs:
            next_out = output_dir + "/" + i[1] + "_" + i[0] + "/"
            next_out = next_out.replace("//", "/")
            os.makedirs(next_out)
            shutil.copy(config_file, next_out)
            
            for line in fileinput.input([next_out + filename], inplace = True):
                if line.strip().startswith("output_directory"):
                    line = "output_directory\t" + next_out + "\tstr\n"
                sys.stdout.write(line)
            
            with open(next_out + filename, 'a') as conf:
                conf.write("\n")
                for j in range(len(keys)):
                    conf.write(keys[j] + "\t" + i[j] + "\tstr\n")
            
            for j in range(runs):
                net = mutable_network(next_out + filename)
                net.train(next_out + "train_run_" + str(j) + ".log")
                net.predict(i[2], next_out + "train_" + str(j) + ".stats")
        
    def parse_input_files(self, input_files):
        inputs = np.loadtxt(input_files, delimiter="\t", dtype = str)
        return inputs[0], inputs[1:]
    
def main(argv):
    wrapper(argv[0])
        
if __name__ == "__main__":
    main(sys.argv[1:])