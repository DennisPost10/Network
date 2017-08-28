#!/usr/bin/env python

import getopt
import glob
import os
from os.path import basename
import subprocess
import sys


def getInfo():
	return "Usage: ./wrapper.py -n network_name -f learning_rate_nw1 -s learning_rate_nw2 -p <prot_directory> -t <training_prot_name_file> -o <output_directory> -e <test_prot_name_file> -b batch_size -x steps"

def main(argv):

	nw_name = ""
	learning_rate_nw1 = "0.001"
	learning_rate_nw2 = "0.001"
	prot_dir = ""
	train_prots = ""
	output_dir = ""
	test_prots = ""
	batch_size = "50"
	steps = "200000"
	try:
		opts, args = getopt.getopt(argv, "hn:f:s:p:t:o:e:b:x:")
	except getopt.GetoptError:
		print("Error")
		print(getInfo())
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(getInfo())
			sys.exit()
		elif opt == '-n':
			nw_name = arg
		elif opt == '-f':
			learning_rate_nw1 = arg
		elif opt == '-s':
			learning_rate_nw2 = arg
		elif opt == '-p':
			prot_dir = arg
		elif opt == '-t':
			train_prots = arg
		elif opt == '-o':
			output_dir = arg
		elif opt == '-e':
			test_prots = arg
		elif opt == '-b':
			batch_size = arg
		elif opt == '-x':
			steps = arg
	if(nw_name == "" or prot_dir == "" or train_prots == "" or output_dir == ""):
		print("EMPTY!")
		print(getInfo())
		sys.exit(2)
	
	print(learning_rate_nw1)
	print(learning_rate_nw2)
	print(batch_size)
	print(steps)

	run_nw1 = "./network.py -t " + train_prots + " -e " + test_prots + " -p " + prot_dir + " -n " + nw_name + " -l " + learning_rate_nw1 + " -o " + output_dir + " -b " + batch_size + " -s " + steps
	
	p = subprocess.Popen(run_nw1, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
	(output, err) = p.communicate()
	print(output)
	print(err)
	
	meta_graph = ""
	ckpt = ""
	meta_dir = output_dir + "/" + nw_name + "/save/"
	for file in os.listdir(meta_dir):
		if file.endswith(".meta"):
			meta_graph = os.path.join(meta_dir, file)
			ckpt = os.path.join(meta_dir, os.path.splitext(os.path.basename(file))[0])

	run_predict1 = "./predict_1.py " + train_prots + " " + prot_dir + " " + meta_graph + " " + ckpt + " " + nw_name
	
	p = subprocess.Popen(run_predict1, shell=True, universal_newlines=True)
	(output, err) = p.communicate()
	print(output)
	print(err)

	run_predict1 = "./predict_1.py " + test_prots + " " + prot_dir + " " + meta_graph + " " + ckpt + " " + nw_name
	
	p = subprocess.Popen(run_predict1, shell=True, universal_newlines=True)
	(output, err) = p.communicate()
	print(output)
	print(err)


	run_nw2 = "./network2.py -t " + train_prots + " -p " + prot_dir + " -n " + nw_name + " -l " + learning_rate_nw2 + " -o " + output_dir + " -b " + batch_size + " -s " + steps
	
	p = subprocess.Popen(run_nw2, shell=True, universal_newlines=True)
	(output, err) = p.communicate()
	print(output)
	print(err)

	meta_graph = ""
	ckpt = ""
	meta_dir = output_dir + "/" + nw_name + "/save2/"
	for file in os.listdir(meta_dir):
		if file.endswith(".meta"):
			meta_graph = os.path.join(meta_dir, file)
			ckpt = os.path.join(meta_dir, os.path.splitext(os.path.basename(file))[0])
	
	run_predict2 = "./predict_2.py " + test_prots + " " + prot_dir + " " + meta_graph + " " + ckpt + " " + nw_name

	p = subprocess.Popen(run_predict2, shell=True, universal_newlines=True)		
	(output, err) = p.communicate()
	print(output)
	print(err)

	print("finished")

if __name__ == "__main__":
	main(sys.argv[1:])
