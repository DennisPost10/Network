#!/usr/bin/env python
import sys
import os
import math
import numpy as np

class InputParser2:

	window_width = 15

	def get_q3_state(self, q8_state):
		
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
# 		elif q8_state == 'I':
# 			return 'C'
# 		elif q8_state == 'S':
# 			return 'C'
# 		elif q8_state == 'T':
# 			return 'C'
# 		elif q8_state == '-':
# 			return 'C'
# 		elif q8_state == 'C':
# 			return 'C'
	
	def getSS_one_hot(self, ss):
		one_hot = [0, 0, 0]
		if ss == 'H':
			one_hot[0] = 1
		elif ss == 'C':
			one_hot[1] = 1
		elif ss == 'E':
			one_hot[2] = 1
		else:
			print("error state = " + str(self.currentSS))
		return one_hot

	def __init__(self, prot_directory, protein_name, nw_name):
		
		self.prot_dir = prot_directory
		self.prot_name = protein_name
# 		print(self.prot_name)
		
		if os.path.exists(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_q3"):
			self.q3_ss = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_q3").read().strip()
		else:
			self.ss = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".secondary_structure").read().strip()
			self.q3_ss = ""
			for a in self.ss:
				self.q3_ss += self.get_q3_state(a)
			f = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_q3", 'w')
			f.write(self.q3_ss)
			f.close()

		if os.path.exists(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot"):
			self.outcomes = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", delimiter="\t", dtype=float)
		else:
			f = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", 'w')
			for a in self.q3_ss:
				b = self.getSS_one_hot(a)
				f.write(str(b[0]) + "\t" + str(b[1]) + "\t" + str(b[2]) + "\n")
			f.close()

			self.outcomes = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", delimiter="\t", dtype=float)
	
		if os.path.exists(self.prot_dir + "/" + self.prot_name + "/" + nw_name + "/" + self.prot_name + ".predicted_ss_one_hot"):
			self.predicted_ss_one_hot = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + nw_name + "/" + self.prot_name + ".predicted_ss_one_hot", delimiter='\t', dtype=float)
#		else:
#			f = open(self.prot_dir + "/" + self.prot_name + "/prot_file.tmp", 'w')
#			f.write(self.prot_name)
#			f.close()
#			meta_graph = ""
#			ckpt = ""
#			meta_dir = "/home/p/postd/bachelor/double_network/output" + "/" + nw_name + "/save/"
#			for file in os.listdir(meta_dir):
#				if file.endswith(".meta"):
#					meta_graph = os.path.join(meta_dir, file)
#					ckpt = os.path.join(meta_dir, os.path.splitext(os.path.basename(file))[0])
#			run_predict1 = "./predict_1.py " + self.prot_dir + "/" + self.prot_name + "/prot_file.tmp" + " " + prot_directory + " " + meta_graph + " " + ckpt + " " + nw_name
#			
#			p = subprocess.Popen(run_predict1, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
#			(output, err) = p.communicate()
#			print(output)
#			print(err)
#			os.remove(self.prot_dir + "/" + self.prot_name + "/prot_file.tmp")
#
#			self.predicted_ss_one_hot = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + nw_name + "/" + self.prot_name + ".predicted_ss_one_hot", delimiter='\t', dtype=float)

		if len(self.predicted_ss_one_hot) < InputParser2.window_width:
			print("Error: prot smaller " + str(InputParser2.window_width))
			print(self.prot_name)
			sys.exit()
		
		self.windows = np.ndarray(shape=(len(self.predicted_ss_one_hot), 45))

		self.window = np.zeros((int(InputParser2.window_width / 2) * 3), dtype=np.float)
		self.currentMid = 0
		
		for i in range(int(InputParser2.window_width / 2) + 1):
			self.window = np.append(self.window, self.predicted_ss_one_hot[i])
		
		self.windows[0] = self.window
		
		empty = np.zeros(3, dtype=float)

		for i in range(1, len(self.predicted_ss_one_hot)):
			
			self.currentMid = i
			self.window = self.window[3:]
			if(self.currentMid >= (len(self.predicted_ss_one_hot) - int(InputParser2.window_width / 2))):
				self.window = np.append(self.window, empty)
			else:
				self.window = np.append(self.window, self.predicted_ss_one_hot[i])
			self.windows[i] = self.window

	def getWindows(self):
		return self.windows
	
	def getOutcomes(self):
		return self.outcomes
