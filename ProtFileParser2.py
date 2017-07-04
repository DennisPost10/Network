import os

from sklearn.utils import shuffle

import numpy as np
from parser2 import InputParser2


class ProtFileParser2:
	
	def __init__(self, prot_name_file, main_prot_dir, output_directory, nw_name):
		base = os.path.splitext(prot_name_file)[0]
		base = os.path.split(base)[1]
		base_name = output_directory + "/" + base
		if os.path.exists(base_name + ".test_windows") and os.path.exists(base_name + ".test_one_hots") and os.path.exists(base_name + ".train_windows") and os.path.exists(base_name + ".train_one_hots"):
			print("reading data...")
			self.all_windows = np.loadtxt(base_name + ".train_windows", dtype = float)
			self.all_one_hots = np.loadtxt(base_name + ".train_one_hots", dtype = float)
			self.test_set = np.loadtxt(base_name + ".test_windows", dtype = float)
			self.test_set_o = np.loadtxt(base_name + ".test_one_hots", dtype = float)
			print("...finished reading")
		else:
			self.main_prot_dir = main_prot_dir
			self.nw_name = nw_name

			self.prots = []
			with open(prot_name_file) as file:
				for line in file:
					line = line.strip()
					self.prots.append(line)
			self.length = len(self.prots)
			print(self.length)

			self.all_prots = []
		
			total_size = 0
			for i in range(self.length):
				n_p = self.next_prot(self.prots[i])
				self.all_prots.append(n_p)
				total_size += len(n_p.predicted_ss_one_hot)
		
			self.all_windows = np.ndarray(shape=(total_size, 45), dtype=float)
			self.all_one_hots = np.ndarray(shape=(total_size, 3), dtype=float)
		
			c = 0
			for i in self.all_prots:
				for j in range(len(i.predicted_ss_one_hot)):
					self.all_windows[c] = i.windows[j]
					self.all_one_hots[c] = i.outcomes[j]
					c += 1

			self.all_prots = None
			self.l = total_size
			print(self.l)
		
			# save 10% as test set: simply take every 10th row
			self.test_set = self.all_windows[::10]
			self.test_set_o = self.all_one_hots[::10]
			self.all_windows = np.delete(self.all_windows, list(range(0, self.l, 10)), axis=0)
			self.all_one_hots = np.delete(self.all_one_hots, list(range(0, self.l, 10)), axis=0)

			np.savetxt(base_name + ".train_windows", self.all_windows)
			np.savetxt(base_name + ".train_one_hots", self.all_one_hots.astype(int))
			np.savetxt(base_name + ".test_windows", self.test_set)					
			np.savetxt(base_name + ".test_one_hots", self.test_set_o.astype(int))

# 			self.all_combined = [n.vstack(self.all_combined.transpose()[x][::]).astype(n.float) for x in range(2)]

		self.index = 0

	def next_prot(self, prot):
		return InputParser2(self.main_prot_dir, prot, self.nw_name)

	def next_batch(self, size):
		rows_left = len(self.all_windows) - self.index
		first_pull = min(rows_left, size)
		ret = self.all_windows[self.index:(self.index + first_pull):1]
		ret_o = self.all_one_hots[self.index:(self.index + first_pull):1]
		self.index += first_pull

		if(self.index > len(self.all_windows)):
			print("self_index error!!")
		if(self.index == len(self.all_windows)):
			self.index = 0
			self.all_windows, self.all_one_hots = shuffle(self.all_windows, self.all_one_hots, random_state=0)
			print("shuffled")
		if(first_pull < size):
			second_pull = size - first_pull
			ret = np.vstack((ret, self.all_windows[self.index:(self.index + second_pull):1]))
			ret_o = np.vstack((ret_o, self.all_one_hots[self.index:(self.index + second_pull):1]))
			self.index += second_pull

		self.next_batch_w = ret
		self.next_batch_o = ret_o
		return
