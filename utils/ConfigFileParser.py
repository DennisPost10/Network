'''
Created on 20.06.2017

@author: Dennis
'''
from utils import Layer


class Configurations:
    
    def __init__(self, config_file):
        self.configs = {}
        with open(config_file) as config:
            layers = []
            for line in config:
                if not line.startswith("#"):
                    key, val, typ = line.split("\t")
                    typ = typ.strip()
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
                else:
                    if line.startswith("#layer"):
                        useless, name, layer_type, relu, output_channels, conv_window_size = line.split("\t")
                        relu = (val == "true" or val == "TRUE" or val == "True" or val == "T")
                        output_channels = int(output_channels)
                        conv_window_size = int(conv_window_size)
                        layers.add(Layer(name, layer_type, relu, output_channels, conv_window_size))
        self.configs["parsed_layers"] = layers
        print(self.configs)
#Configurations("D:/Dennis/Uni/bachelor/double_network/config.txt")