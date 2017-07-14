'''
Created on 12.07.2017

@author: Dennis
'''

import matplotlib.pyplot as plt
import seaborn as sea

class barplot:
    
    def plot(self, xlab, ylab, invert = False, display = False, save = True):
        sea.set(style="whitegrid", color_codes=True)
        plt.figure()
        if invert:
            ax = sea.barplot(x='value', y='categorie', data=self.data)
        else:    
            ax = sea.barplot(x='categorie', y='value', data=self.data)
        ax.set(xlabel = xlab, ylabel = ylab)
        if save:
            plt.savefig(self.output_file)
        if display:
            plt.show()
    
    def __init__(self, data_dict, output_file):
        self.data = data_dict
        self.output_file = output_file
        
# plot HCE counts
# y={'cat': ['H','C','E'], 'val': [sum(d.num_h),sum(d.num_c),sum(d.num_e)]}
# ax = sns.barplot(x='cat', y='val', data = y)
# ax.set(xlabel="secondary structure", ylabel="occurences")

# z={'cat': ['observed', 'predicted', 'correct'], 'val': {'observed':[sum(d.num_h),sum(d.num_c),sum(d.num_e)], 'predicted':[sum(d.pred_h),sum(d.pred_c),sum(d.pred_e)], 'correct':[sum(d.h_correct),sum(d.c_correct),sum(d.e_correct)]}}

# df=pa.DataFrame(a+b+c)
# df=pa.DataFrame([sum(d.num_h), sum(d.num_c), sum(d.num_e), sum(d.pred_h), sum(d.pred_c), sum(d.pred_e), sum(d.h_correct), sum(d.c_correct), sum(d.e_correct)])
# df[1]=pa.Series(["h", "c", "e", "h", "c", "e", "h", "c", "e"], index=df.index)
# df[2]=pa.Series(["observed", "observed", "observed", "predicted", "predicted", "predicted", "correct", "correct", "correct"], index=df.index)
# df.columns=['occurences', 'secondary structure', 'type']
# ax = sns.barplot(x=df[2], y=df[0], hue=df[1], data=df)