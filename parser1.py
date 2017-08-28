import math
import os
import sys

import numpy as np


class InputParser:

	window_width = 15

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

	def readFile(self, txt):
		self.data = np.loadtxt(txt, delimiter="\t", skiprows=1, usecols=range(2, 22), dtype=float)
		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				self.data[i][j] = 1.0 / (1.0 + math.exp(-self.data[i][j]))
	
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
		

	def __init__(self, prot_directory, protein_name):
	
		self.prot_dir = prot_directory
		self.prot_name = protein_name
# 		print(self.prot_name)

# 		self.aa_sequence=open(self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".aa_sequence").read().replace('\n', '')

# 		self.ss=open(self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".secondary_structure").read().strip()
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
			
		if len(self.q3_ss) < InputParser.window_width:
			print("Error: prot smaller " + str(InputParser.window_width))

			sys.exit()

		if os.path.exists(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot"):
			self.outcomes = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", delimiter="\t", dtype=float)
		else:
			f = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", 'w')
			for a in self.q3_ss:
				b = self.getSS_one_hot(a)
				f.write(str(b[0]) + "\t" + str(b[1]) + "\t" + str(b[2]) + "\n")
			f.close()

			self.outcomes = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".ss_one_hot", delimiter="\t", dtype=float)

		# self.f=open(self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".tab")
		# self.fileName=self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".tab"
		
		if os.path.exists(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".windows"):
			self.windows = np.loadtxt(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".windows", delimiter="\t", dtype=float)
		else:
			
# 		self.aa_rank=self.f.readline().strip().split("\t")
# 		self.aa_rank.pop(0)
# 		self.aa_rank.pop(0)
		
# 		outputFileWind=open(self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".windows", 'w')
# 		outputFileOutcome=open(self.prot_dir+"/"+self.prot_name+"/"+self.prot_name+".one_hot", 'w')
			
			self.readFile(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".tab")
			
			self.windows = np.ndarray(shape=(len(self.q3_ss), 300), dtype=float)

			self.window = np.zeros((int(InputParser.window_width / 2) * 20), dtype=np.float)
			self.currentMid = 0
			
			for i in range(int(InputParser.window_width / 2) + 1):
				self.window = np.append(self.window, self.data[i])
			self.windows[0] = self.window
			
			empty = np.zeros(20, dtype=float)

			for i in range(1, len(self.q3_ss)):
			
				self.currentMid = i
				self.window = self.window[20:]
				if(self.currentMid >= (len(self.q3_ss) - int(InputParser.window_width / 2))):
					self.window = np.append(self.window, empty)
				else:
					self.window = np.append(self.window, self.data[i + int(InputParser.window_width / 2)])
				self.windows[i] = self.window
# 		print(np.shape(self.windows))
			f = open(self.prot_dir + "/" + self.prot_name + "/" + self.prot_name + ".windows", 'w')
			for i in self.windows:
				s = ""
				for j in i:
					s = s + str(j) + "\t"
				s = s[:-1]
				f.write(s + "\n")
			f.close()

	def getCurrentSS(self):
		return self.currentSS

	def getWindows(self):
		return self.windows
	
	def getOutcomes(self):
		return self.outcomes
