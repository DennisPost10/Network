'''
Created on 19.06.2017

@author: Dennis
'''
from network_c import nw1
from network_d import nw2


def run():
    net1 = nw1("D:/Dennis/Uni/bachelor/double_network/config2.txt")
    net1.train("D:/Dennis/Uni/bachelor/tdbdata/train1.lst")
    net1.restore_graph_without_knowing_checkpoint()
    net1.predict("D:/Dennis/Uni/bachelor/tdbdata/train1.lst")
    net1.predict("D:/Dennis/Uni/bachelor/tdbdata/test1.lst")
    net2 = nw2("D:/Dennis/Uni/bachelor/double_network/config2.txt")
    net2.train("D:/Dennis/Uni/bachelor/tdbdata/train1.lst")
    net2.restore_graph_without_knowing_checkpoint()
    net2.predict("D:/Dennis/Uni/bachelor/tdbdata/test1.lst", net1)

if __name__ == '__main__':
    run()
#    prot_dir = "D:/Dennis/Uni/bachelor/parsed/"
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/train1.lst"
#    p = ProtFileParser(prot_file, prot_dir)
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/train2.lst"
#    p = ProtFileParser(prot_file, prot_dir)
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/train3.lst"
#    p = ProtFileParser(prot_file, prot_dir)
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/test1.lst"
#    p = ProtFileParser(prot_file, prot_dir)
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/test2.lst"
#    p = ProtFileParser(prot_file, prot_dir)
#    prot_file = "D:/Dennis/Uni/bachelor/tdbdata/test3.lst"
#    p = ProtFileParser(prot_file, prot_dir)