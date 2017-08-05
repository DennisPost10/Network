'''
Created on 12.07.2017

@author: Dennis
'''
import matplotlib.pyplot as plt
import seaborn as sea


class histogramm:
    
    def plot(self, xlab, ylab, bins = -1, kde = True, rug = False, display = False, save = True, vertical = False):
        sea.set(style="whitegrid", color_codes=True)
        plt.figure()
        if bins > -1:
            ax = sea.distplot(self.data, bins = bins, kde = kde, rug = rug, vertical = vertical)
        else:
            ax = sea.distplot(self.data, kde = kde, rug = rug, vertical = vertical)
        ax.set(xlabel=xlab, ylabel=ylab)
        if save:
            plt.savefig(self.output_file)
        if display:
            plt.show()
    
    def __init__(self, data, output_file):
        self.data = data
        self.output_file = output_file