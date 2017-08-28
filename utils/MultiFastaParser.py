import os
import re
import sys

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
        ss += match.strip().split(">")[1].replace("&nbsp;", "-")

    return ss

def parse_q3(ss):
    ss_q3 = ""
    for x in ss:
        ss_q3 += get_q3_state(x)
    return ss_q3

found = set()
     
def read_ss_db(ss_db, filter_set, output_protein_directory):     
    header_seq = None
    header_ss = None
    seq = ""
    ss = ""
    in_ss = False
    x = 0
    with open(ss_db) as f:
        line = f.readline().strip()
        while line:
            if line.startswith(">"):
                
                if line.endswith("sequence"):
                    if header_ss != None:
                        x += add_chain(header_ss, seq, ss, filter_set, output_protein_directory)
                    in_ss = False
                    header_seq = line
                    seq = ""
                    ss = ""                       
                else:
                    in_ss = True
                    header_ss = line
            else:
                if in_ss:
                    ss += line
                else:
                    seq += line
                
            line = f.readline().strip('\n')
        
        if header_ss != None:
            x += add_chain(header_ss, seq, ss, filter_set, output_protein_directory)
    print(x)
    print(filter_set.difference(found))
    
def read_fasta(mult_fasta_file, filter_set, output_protein_directory):
    header = None
    seq = ""
    x = 0
    with open(mult_fasta_file) as f:
        line = f.readline().strip()
        while line:
            if line.startswith(">"):
                if header == None:
                    header = line
                else:
                    x += add_chain(header, seq, filter_set, output_protein_directory)
                    header = line
                    seq = ""
            else:
                seq += line
                
            line = f.readline().strip()
        
        if header != None:
            x += add_chain(header, seq, filter_set, output_protein_directory)
    print(x)
    
def add_chain(header, sequence, ss, filter_set, output_protein_directory):
    state = 0
#    name, equis = get_parsed_header(header)
    name = get_parsed_header(header)
    if name in filter_set:  # or any(x in filter_set for x in equis):
        found.add(name)
        sequence = get_parsed_seq(sequence)
#        ss = get_ss_from_source_code(name)
        ss_q3 = parse_q3(ss)
        print(name)
        state = 1

#    if ss != None:
#        print(equis)
        print(sequence)
        print(ss)
        print(ss_q3)
        if len(ss) != len(sequence):
            print("Error: different lengths!!!")
            sys.exit(1)
        output_protein_directory += name + "/"
        if(os.path.exists(output_protein_directory)):
            print("skipped " + name)

        os.mkdir(output_protein_directory)
        output_protein_directory += name
        open(output_protein_directory + ".secondary_structure", "w").write(ss)
        open(output_protein_directory + ".ss_q3", "w").write(ss_q3)
        open(output_protein_directory + ".fasta", "w").write(">" + name + "\n" + sequence)
        one_hot_dict = {'H': "1\t0\t0\n", 'C': "0\t1\t0\n", 'E': "0\t0\t1\n"}
        ss_one_hot = open(output_protein_directory + ".ss_one_hot", "w")
        for sec_struc in ss_q3:
            ss_one_hot.write(one_hot_dict[sec_struc])
        ss_one_hot.close()
        
    return state

def get_parsed_seq(sequence):
    
    return sequence
    

def get_parsed_header(header):
    
#    split_1 = header.split('||')
#    equis = []
#    if len(split_1) > 1:
#        equis = split_1[1].split()
#    name = header.split(None, 1)[0][1:]
#
    name = header[1:].split(":")
    name = name[0] + name[1]
    return name  # , equis

def read_filter_file(name_file):
    incl = set()
    with open(name_file) as f:
        line = f.readline()  # skip header
        line = f.readline().strip()
        while line:
            incl.add(line.split(None, 1)[0]) 
            line = f.readline().strip()
    return incl

def main(argv):
    read_ss_db("/home/p/postd/bachelor/data/cull_pdb/ss.txt", read_filter_file("/home/p/postd/bachelor/data/cull_pdb/cullpdb_pc25_res1.8_R0.25_d170706_chains6421"), "/home/p/postd/bachelor/data/cull_pdb/proteins/")
#    read_fasta("D:/Dennis/Uni/bachelor/data/cull_pdb/pdbaa.nr", read_filter_file("D:/Dennis/Uni/bachelor/data/cull_pdb/cullpdb_pc25_res1.8_R0.25_d170706_chains6421"), "D:/Dennis/Uni/bachelor/data/cull_pdb/proteins/")

if __name__ == '__main__':
    main(sys.argv[1:])
