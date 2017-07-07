'''
Created on 20.06.2017

@author: Dennis
'''



class Configurations:
    
    def __init__(self, config_file):
        self.configs = {}
        with open(config_file) as config:
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
                    else:
                        self.configs[key] = val
        print(self.configs)
#Configurations("D:/Dennis/Uni/bachelor/double_network/config.txt")