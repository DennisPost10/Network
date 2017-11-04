'''
Created on 20.06.2017

@author: Dennis
'''
from utils.Layer import Layer


class Configurations:
    
    def __init__(self, config_file):
        self.configs = {}
        with open(config_file) as config:
            layers = list()
            for line in config:
                if line.strip():
                    # print(line.strip())
                    if not line.startswith("#"):
                        key, val, typ = line.strip().split("\t")
                        if typ == "int":
                            self.configs[key] = int(val)
                        elif typ == "float":
                            self.configs[key] = float(val)
                        elif typ == "bool":
                            self.configs[key] = (val == "true" or val == "TRUE" or val == "True" or val == "T")
                        elif typ == "none":
                            self.configs[key] = None
                        elif typ == "int_array":
                            arr = val.split(",")
                            arr = list(map(int, arr))
                            self.configs[key] = arr
                        elif typ == "float_array":
                            arr = val.split(",")
                            arr = list(map(float, arr))
                            self.configs[key] = arr
                        elif typ == "string_array":
                            arr = val.split(",")
                            self.configs[key] = arr
                        else:
                            self.configs[key] = val
                    elif line.startswith("#layer"):
                        useless, name, layer_type, relu, output_channels, window_size = line.split("\t")
                        relu = (relu == "true" or relu == "TRUE" or relu == "True" or relu == "T")
                        output_channels = int(output_channels)
                        window_size = int(window_size)
                        layers.append(Layer(name, layer_type, relu, output_channels, window_size))
        self.configs["parsed_layers"] = layers
        print(self.configs)
# Configurations("D:/Dennis/Uni/bachelor/double_network/config.txt")
