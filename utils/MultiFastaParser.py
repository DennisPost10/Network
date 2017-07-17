import sys
import re
import requests

regex = "class=\"reportsequence\" NOWRAP width=\"85%\".*\n"
pattern = re.compile(regex)

def get_q3_state(q8_state):
        
    if q8_state == 'H':
        return 'H'
    elif q8_state == 'G':
        return 'H'
    elif q8_state == 'E':
        return 'E'
    elif q8_state == 'B':
        return 'E'
    else:
        return 'C'

def get_ss_from_source_code(prot_name):
    prot = prot_name[:-1]
    chainId = prot_name[-1:]
    url = 'https://www.rcsb.org/pdb/explore/sequenceText.do?structureId=' + prot + '&chainId=' + chainId

    response = requests.get(url).text
    matches = re.findall(pattern, response)
    ss = ""
    for match in matches:
        ss+=match.strip().split(">")[1].replace("&nbsp;", "-")

    return ss

def parse_q3(ss):
    ss_q3 = ""
    for x in ss:
        ss_q3 += get_q3_state(x)
    return ss_q3
        
def read_fasta(mult_fasta_file, filter_set):    
    header = None
    seq = ""
    x=0
    with open(mult_fasta_file) as f:
        line = f.readline().strip()
        while line:
            if line.startswith(">"):
                if header == None:
                    header = line
                else:
                    x+=add_chain(header, seq, filter_set)
                    header = line
                    seq = ""
            else:
                seq += line
                
            line = f.readline().strip()
        
        if header != None:
            x+=add_chain(header, seq, filter_set)
    print(x)    
        
def add_chain(header, sequence, filter_set):
    state = 0
    name, equis = get_parsed_header(header)
    ss = None
    if name in filter_set:# or any(x in filter_set for x in equis):
        sequence = get_parsed_seq(sequence)
        ss = get_ss_from_source_code(name)
        ss_q3 = parse_q3(ss)
        print(name)
        state = 1
    else:
        for x in equis:
            if x in filter_set:
                ss = get_ss_from_source_code(x)
                ss_q3 = parse_q3(ss)
                print(name)
                print(x)                
                state = 1
    if ss != None:
        print(equis)
        print(ss)
        print(ss_q3)
    return state

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