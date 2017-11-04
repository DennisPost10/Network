# iterate through
# optimizer
# learning rates
# window size
# number of layers
# batch size
import fileinput
import os
import shutil
import sys
from tempfile import NamedTemporaryFile

from NetworkWrapper import wrapper
from utils.ConfigFileParser import Configurations


def iterate_2(config_file):
    configs = Configurations(config_file)
    if(configs == None):
            print("Error: no input")
            sys.exit()

    output_directory = configs.get("output_directory")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # # file where unique names are written to
    save_file = configs.get("short_name_file")
    name = configs.get("base_name")
    features = configs.get("features")
    ss_features = configs.get("ss_features")
    learning_rates = configs.get("learning_rates")
    batch_sizes = configs.get("batch_sizes")
    steps = configs.get("max_steps")
    keep_prob_vals = configs.get("keep_probs")
    optimizer = configs.get("optimizer")
    window_sizes = configs.get("window_sizes")
    momentum_vals = configs.get("momentums")
    
    for window_size in window_sizes:
        net_name = name + "_" + str(window_size) + "_"
        for keep_prob_val in keep_prob_vals:
            net_name += str(keep_prob_val) + "_"
            for learning_rate in learning_rates:
                net_name += str(learning_rate) + "_"
                for batch_size in batch_sizes:
                    net_name += str(batch_size) + "_"
                    for optimizer_ in optimizer:
                        net_name += optimizer
                        if optimizer_ == "momentum":
                            for momentum_val in momentum_vals:
                                net_name += "_" + str(momentum_val)
                        else:
                            print("")
   
def iterate(config_file):
    configs = Configurations(config_file)
    if(configs == None):
            print("Error: no input")
            sys.exit()

    output_directory = configs.get("output_directory")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    filename = os.path.basename(config_file)
    name = configs.get("name")
    ss_features = configs.get("ss_features")
    layer_sizes = [50, 100]
    
    for a in range(2):
        for b in range(2):
            if a == 0 and b == 1:
                continue
            for i in range(len(layer_sizes)):
                use_aa_seq = (a == 1)
                single_aa_seq = (b == 1)
                layer_size = layer_sizes[i]
                for j in range(3):
                    out = output_directory + name + "_"
                    for k in range(j + 1):
                        out += str(layer_size) + "_"
                    out += str(a) + "_" + str(b) + "/"
                
                os.makedirs(out)
                shutil.copy(config_file, out)
                for line in fileinput.input([out + filename], inplace = True):
                    if line.strip().startswith("output_directory"):
                        line = "output_directory\t" + out + "\tstr\n"
                    sys.stdout.write(line)

                with open(out + filename, 'a') as conf:
                    conf.write("\n")
                    conf.write("use_aa_seq" + "\t" + use_aa_seq + "\tbool\n")
                    conf.write("single_aa_seq" + "\t" + single_aa_seq + "\tbool\n")
                    for k in range(j + 1):
                        conf.write("#layer\tlayer_" + str(k) + "\tfully\tFalse\t1\t" + str(layer_size) + "\n")
                    conf.write("#layer\tlayer_out\tfully\tFalse\t1\t" + str(ss_features) + "\n")
                        
                wrapper(out + filename)

def generate_random_short_name(output_directory, save_file, real_name):
    config_file = NamedTemporaryFile(dir=output_directory, delete=False)
    short_name = config_file.name
    with open(save_file, mode='a') as save:
        save.write(short_name + "\t" + real_name + "\n")
    os.makedirs(output_directory + short_name + "/")
    os.rename(config_file, output_directory + short_name + "/" + "configs.config")
    return config_file, short_name

def write_nw_config_file(output_directory, save_file, features, ss_features, window_size, layer, keep_prob, learning_rate, batch_size, max_steps, optimizer, momentum_val, nw_name):
    real_name = ""
    config_file, short_name = generate_random_short_name(output_directory, save_file, real_name)
    with open(config_file, 'w') as conf:
        conf.write("#short_name\t" + short_name + "\n")
        conf.write("name" + "\t" + short_name + "\tstr\n")
        conf.write("output_directory" + "\t" + output_directory + "/" + short_name + "\tstr\n")
        conf.write("features" + "\t" + features + "\tint\n")
        conf.write("ss_features" + "\t" + ss_features + "\tint\n")
        conf.write("learning_rate" + "\t" + learning_rate + "\tfloat\n")
        conf.write("keep_prob" + "\t" + keep_prob + "\tfloat\n")
        conf.write("batch_size" + "\t" + batch_size + "\tint\n")
        conf.write("max_steps" + "\t" + max_steps + "\tint\n")
        conf.write("optimizer" + "\t" + optimizer + "\tstr\n")        
        if not momentum_val is None:
            conf.write("momentum" + "\t" + momentum_val + "\tfloat\n")
        
        
def main(argv):
    config_file = argv[0]
    print(config_file)
    iterate(config_file)

if __name__ == "__main__":
    main(sys.argv[1:])
