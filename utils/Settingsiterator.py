# iterate through
# optimizer
# learning rates
# window size
# number of layers
# batch size
import sys

from utils.ConfigFileParser import Configurations
from network import keep_prob


def iterate(config_file):
    configs = Configurations(config_file)
    if(configs == None):
            print("Error: no input")
            sys.exit()

    prot_directory = configs["protein_directory"]
    output_directory = configs["output_directory"]
    name = configs["base_name"]
    learning_rates = configs["learning_rates"]
    batch_sizes = configs["batch_sizes"]
    steps = configs["max_steps"]
    keep_prob_vals = configs["keep_probs"]
    optimizer = configs["optimizer"]
    window_sizes = configs["window_sizes"]
    momentum_vals = configs["momentums"]
    layers = configs["layers"]
    
    for window_size in window_sizes:
        net_name = name + "_" + str(window_size) + "_"
        for layer in layers:
            net_name += str(layer) + "_"
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
                                
    

def write_nw_config_file(output_directory, prot_directory, window_size, layer, keep_prob, learning_rate, batch_size, max_steps, optimizer, momentum_val, nw_name):
    

def main(argv):
    config_file = argv[0]
    print(config_file)
    iterate(config_file)

if __name__ == "__main__":
    main(sys.argv[1:])
