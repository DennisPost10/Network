
class Layer():

    # in config file
    # layer    layername
    #    type    conv|dropout|fully
    #    relu    True|False
    #    output_channels    x>0
    #    conv_window_size    x>0(only matters if type is conv)
    def __init__(self, output_channels, type="fully", relu=False, conv_window_size=11):
        print("")
