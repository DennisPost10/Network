'''
Created on Jul 14, 2017

@author: postd
'''

import os.path
import sys

import pandas as p
from plotting.histogramm import histogramm
from plotting.barplot import barplot


class stats_plotter:
    def import_stat_matrix(self):
        if not os.path.exists(self.matrix_file):
            print("Couldn't find matrix-file: " + self.matrix_file + " !!!")
            sys.exit()
        self.stats_matrix = p.read_table(self.matrix_file, delimiter='\t')

    def plot_all_stats(self, save = True):
        self.plot_length_distribution()
    
    def plot_length_distribution(self):
        length_hist = histogramm(self.stats_matrix['length'], self.output_directory + ".length_distribution")
        length_hist.plot(ylab="#proteins", xlab="protein length", save=False, display=True)
        
#        ss_counts = barplot(, output_file)
        
    
    def __init__(self, matrix_file, output_directory):
        self.matrix_file = matrix_file
        self.output_directory = output_directory + "/"
        self.import_stat_matrix()