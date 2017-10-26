
class Layer():

    # in config file
    # layer    layername
    #    type    conv|dropout|fully
    #    relu    True|False
    #    output_channels    x>0
    #    conv_window_size    x>0(only matters if type is conv)
    def __init__(self, name, layer_type, relu, output_channels, window_size):
        self.name = name
        self.layer_type = layer_type
        self.relu = relu
        self.output_channels = output_channels
        self.window_size = window_size
        if layer_type == "fully":
            self.output_channels = 1
