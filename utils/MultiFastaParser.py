import sys

import requests


def get_source_code(prot_name):
    prot = prot_name[:-1]
    chainId = prot_name[-1:]
    url = 'https://www.rcsb.org/pdb/explore/sequenceText.do?structureId=' + prot + '&chainId=' + chainId

    response = requests.get(url)
    print(response.content)


def read_fasta(mult_fasta_file, filter_set):    
    header = None
    seq = ""
    with open(mult_fasta_file) as f:
        line = f.readline().strip()
        while line:
            if line.startswith(">"):
                if header == None:
                    header = line
                else:
                    add_chain(header, seq, filter_set)
                    header = line
                    seq = ""
            else:
                seq += line
                
            line = f.readline().strip()
        
        if header != None:
            add_chain(header, seq, filter_set)
            
def add_chain(header, sequence, filter_set):
    name, equis = get_parsed_header(header)
    if name in filter_set or any(x in filter_set for x in equis):
        sequence = get_parsed_seq(sequence)
        get_source_code(name) 
        #print(name)
        #print(equis)
        #print(sequence)
    
def get_parsed_seq(sequence):
    
    return sequence
    

def get_parsed_header(header):
    
    split_1 = header.split('||')
    equis = []
    if len(split_1) > 1:
        equis = split_1[1].split()
    name = header.split(None, 1)[0][1:]
        
    return name, equis

def read_filter_file(name_file):
    incl = set()
    with open(name_file) as f:
        line = f.readline().strip()
        while line:
            incl.add(line.split(None, 1)[0]) 
            line = f.readline().strip()
    return incl
    
if __name__ == '__main__':
    read_fasta("/home/p/postd/bachelor/data/cull_pdb/pdbaa.nr", read_filter_file("/home/p/postd/bachelor/data/cull_pdb/cullpdb_pc25_res1.8_R0.25_d170706_chains6421"))