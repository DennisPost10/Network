import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from utils.Layer import Layer
from utils.ConfigFileParser import Configurations
from utils.InputHandler import Input_Handler
from numpy import Infinity


sss = ['H', 'C', 'E']

class mutable_network:

	def one_hot(self, index):
		ret = np.zeros(3)
		ret[index] = 1
		return ret

	def to_ss(self, indices):
		ss = ""
		for i in indices:
			ss += sss[i]
		return ss

	def conv1d(self, x, neurons, name_val):
		return tf.nn.conv1d(x, filters=neurons, stride=1, padding='SAME', name=name_val)

	def weight_variable(self, shape, name_val):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, tf.float32, name=name_val)

	def bias_variable(self, shape, name_val):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, tf.float32, name=name_val)

	def init_optimizer(self):
		if self.optimizer == "adam":
			return tf.train.AdamOptimizer(self.learning_rate)
		elif self.optimizer == "adagrad":
			return tf.train.AdagradOptimizer(self.learning_rate)
		elif self.optimizer == "momentum":
			return tf.train.MomentumOptimizer(self.learning_rate, self.momentum_val)
		elif self.optimizer == "gradient":
			return tf.train.GradientDescentOptimizer(self.learning_rate)
		elif self.optimizer == "adadelta":
			return tf.train.AdadeltaOptimizer(self.learning_rate)
		else:
			return None

	def layer(self, input_layer, input_tf_layer, layer_count):
		if layer_count >= len(self.layers):
			return None
		next_layer = self.layers[layer_count]
		with tf.name_scope(next_layer.name):
			if next_layer.layer_type == "dropout":
				hidden_layer = tf.nn.dropout(input_tf_layer, self.keep_prob_val, "dropout")
			elif next_layer.layer_type == "fully":
				print(input_layer.window_size)
				print(next_layer.window_size)
				hidden_weights = self.weight_variable([input_layer.window_size, next_layer.window_size], "weight")
				bias = self.bias_variable([next_layer.window_size], "bias")
				hidden_layer = tf.add(tf.matmul(input_tf_layer, hidden_weights), bias, name = "output_layer")
			elif next_layer.layer_type == "conv":
				hidden_weights = self.weight_variable([next_layer.window_size, input_layer.output_channels, next_layer.output_channels], "weight")
				bias = self.bias_variable([next_layer.output_channels], "bias")
				hidden_layer = self.conv1d(input_tf_layer, hidden_weights, name = "layer")
			if next_layer.relu:
				hidden_layer = tf.nn.relu(hidden_layer)
		tf_layer = hidden_layer
		self.tf_layers.append(tf_layer)
		return self.layer(next_layer, tf_layer, layer_count + 1)

	def build_graph(self):
		self.g = tf.Graph()
		with self.g.as_default():
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
			
			if self.network_type == "conv":
				# input
				self.x = tf.placeholder(tf.float32, [None, self.max_prot_length, self.features + self.aa_seq_add], name="x")
				
				# final output
				self.y = tf.placeholder(tf.float32, [None, self.max_prot_length, self.ss_features], name="y")
			
				self.layer(Layer("x", "", False, self.features + self.aa_seq_add, self.max_prot_length), self.x, 0)
			
			elif self.network_type == "mixed":
				# input
				self.x = tf.placeholder(tf.float32, [None, self.window_size, self.features + self.aa_seq_add], name="x")
				# final output
				self.y = tf.placeholder(tf.float32, [None, self.ss_features], name="y")
			
				self.layer(Layer("x", "", False, self.features + self.aa_seq_add, self.window_size), self.x, 0) #!!!!!!
			
			else:
				# input
				if self.single_aa_seq:
					self.x = tf.placeholder(tf.float32, [None, self.window_size * self.features + self.aa_seq_add], name="x")				
				else:
					self.x = tf.placeholder(tf.float32, [None, self.window_size * (self.features + self.aa_seq_add)], name="x")
					
				# final output
				self.y = tf.placeholder(tf.float32, [None, self.ss_features], name="y")

				if self.single_aa_seq:
					self.layer(Layer("x", "", False, 1, self.window_size * self.features + self.aa_seq_add), self.x, 0)
				else:
					self.layer(Layer("x", "", False, 1, self.window_size * (self.features + self.aa_seq_add)), self.x, 0)
					

			self.correct_prediction = tf.equal(tf.argmax(self.tf_layers[self.layer_count - 1], 1), tf.argmax(self.y, 1), name="correct_prediction")
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
			self.observed = tf.argmax(self.y, 1)
			self.h_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 0), name = "h_count", dtype = tf.int32)
			self.c_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 1), name = "c_count", dtype = tf.int32)
			self.e_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 2), name = "e_count", dtype = tf.int32)
			self.h_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 0))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.tf_layers[self.layer_count - 1],1), 0))))))[1], self.h_count, name = "h_accuracy")
			self.c_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 1))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.tf_layers[self.layer_count - 1],1), 1))))))[1], self.c_count, name = "c_accuracy")
			self.e_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 2))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.tf_layers[self.layer_count - 1],1), 2))))))[1], self.e_count, name = "e_accuracy")
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.tf_layers[self.layer_count - 1]), name="loss")
			
#			self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum_val, name="train_step").minimize(self.loss, global_step=self.global_step)
			self.train_step = self.init_optimizer().minimize(self.loss, global_step = self.global_step)
			
			self.init_op = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep=1)
			
			self.sess = tf.Session(graph = self.g)
			self.sess.run(self.init_op)
			self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
			self.saver.export_meta_graph(self.output_directory + '/save/' + self.name + "_meta")
			tf.summary.FileWriter(self.output_directory + '/save/' + self.name).add_graph(self.g)
			#python -m tensorflow.tensorboard --logdir="C:\Users\Dennis\Desktop\test\test_afs\save\"
	def restore_graph(self, checkpoint):
		with self.g.as_default():
			print(self.sess.run(self.b_p))
			tf.train.Saver().restore(self.sess, checkpoint)
			self.start_index = self.sess.run(self.global_step)
			self.restored_graph = True
			print(self.sess.run(self.b_p))
	
	def restore_graph_without_knowing_checkpoint(self):
		self.ckpt = None
		for file in os.listdir(self.output_directory + "/save/"):
			if file.endswith(".meta"):
				self.ckpt = self.output_directory + "/save/" + os.path.splitext(file)[0]
		if(self.ckpt == None):
				print("Error: missing ckpt")
				sys.exit()
		print("ckpt: " + self.ckpt)
		self.restore_graph(self.ckpt)
		
	def train(self):
		
		self.prot_it = Input_Handler(self.input_directory, self.data_file_base_name, self.ttv_file, self.index, self.max_prot_length, self.network_type, self.window_size)

		self.val_batch, self.val_batch_o, self.val_batch_l = self.prot_it.val_batches()

		with self.g.as_default():
			if self.restored_graph:
				loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
				self.winner_acc = accuracy_eval
				self.winner_loss = loss_val
			
			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('accuracy', self.accuracy)
			tf.summary.scalar('h_accuracy', self.h_accuracy)
			tf.summary.scalar('c_accuracy', self.c_accuracy)
			tf.summary.scalar('e_accuracy', self.e_accuracy)
			summary = tf.summary.merge_all()
		
			summary_writer = tf.summary.FileWriter(self.output_directory + '/summary', self.g)
		
			batch_count = 0
			better = False
			
			check_range = 100
			alpha = 0.001
			lower_acc = self.winner_acc, lower_loss = self.winner_loss
			
			for step in range(self.start_index, self.start_index + self.steps):
				if step % check_range == 0:
					summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
					if(accuracy_eval - lower_acc > alpha or loss_val - lower_loss < alpha):
						self.winner_acc = max(self.winner_acc, accuracy_eval)
						self.winner_loss = min(self.winner_loss, loss_val)
						lower_acc = accuracy_eval
						lower_loss = loss_val
						better = True
						self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
#				if step % 1000 == 0:
						print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (step, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
						summary_writer.add_summary(summarystr, step)		
#					if step % 10000 == 0:
						if better:
							better = False
						else:
							print("finished early")
							return
				
				ret_prots, ret_prots_o, lengths = self.prot_it.next_prots(self.batch_size)
				_ = self.sess.run(self.train_step, feed_dict={self.x: ret_prots, self.y: ret_prots_o, self.keep_prob: self.keep_prob_val, self.prot_lengths: lengths})
				batch_count += 1
		
			self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
			summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.prot_it.test_set, self.y: self.prot_it.test_set_o, self.keep_prob: 1})
			print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (self.start_index + self.steps, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
			summary_writer.add_summary(summarystr, self.start_index + self.steps)

			print("training finished")
		
	def predict(self, test_file):
		base = os.path.splitext(os.path.split(test_file)[1])[0]
		with self.g.as_default():
			prediction = tf.argmax(self.tf_layers[self.layer_count - 1], 1)

			hce_counts = np.zeros(3, dtype = float)
			hce_matches = np.zeros(3, dtype = float)

			number_correct = 0.0
			looked_at = 0.0
			
			scores_per_prot = []
			with open(test_file, 'r') as prots:
				for prot in prots:
					prot = prot.strip()
					print(prot)
					next_prot = Input_Handler(self.prot_directory, prot)
					prot_nw_dir = self.prot_directory + "/" + prot + "/" + self.name + "/"
					if not os.path.exists(prot_nw_dir):
						os.makedirs(prot_nw_dir)
					pred_ss = open(prot_nw_dir + prot + ".predicted_ss", 'w')
					pred_ss_one_hot = open(prot_nw_dir + prot + ".predicted_ss_one_hot", 'w')
					obs, pred, corr = self.sess.run([self.observed, prediction, self.correct_prediction], feed_dict={self.x: next_prot.getWindows(), self.y: next_prot.getOutcomes(), self.keep_prob: 1})
					pred_ss.write(self.to_ss(pred))
					for p in pred:
						a = self.one_hot(p)
						pred_ss_one_hot.write(str(a[0]) + "\t" + str(a[1]) + "\t" + str(a[2]) + "\n")
					corr_prot = 0.0
					for i in range(len(corr)):
						if corr[i]:
							corr_prot += 1
							number_correct += 1
							hce_matches[pred[i]] += 1
						hce_counts[obs[i]] += 1
					scores_per_prot.append(corr_prot / len(pred) * 100)
					looked_at += len(pred)
					pred_ss.close()
					pred_ss_one_hot.close()
			print(scores_per_prot)
			plt.figure()
			plt.hist(scores_per_prot)
			plt.savefig(self.output_directory + base + "_prot_scores.png")
			print("%d out of %d correct predicted (%.3f)" % (number_correct, looked_at, number_correct / looked_at))
		
	def __init__(self, config_file):
		
		configs = Configurations(config_file).configs
		
		if(configs == None):
			print("Error: no input")
			sys.exit()

#		self.training_file = configs["training_file"]
#		self.test_file = configs["test_file"]
		self.output_directory = configs.get("output_directory")
		self.name = configs.get("name")
		self.learning_rate = configs.get("learning_rate")
		self.batch_size = configs.get("batch_size")
		self.steps = configs.get("max_steps")
		self.keep_prob_val = configs.get("keep_prob")
		self.momentum_val = configs.get("momentum")
		self.optimizer = configs.get("optimizer")
		self.max_prot_length = configs.get("max_prot_length")
		self.features = configs.get("features")
		self.ss_features = configs.get("ss_features")
		self.network_type = configs.get("network_type")
		
		self.use_aa_seq_data = configs.get("use_aa_seq_data")
		self.aa_codes = configs.get("aa_codes")
		self.aa_seq_add = self.aa_codes
		if not self.use_aa_seq_data:
			self.aa_seq_add = 0
		
		self.window_size = configs.get("window_size")
		
		# dir where input npys are
		self.input_directory = configs.get("input_directory")
		# "psi_prots"|"cull_pdb" -> .matrix.npy and .one_hots.npy will be appended
		self.file_base = configs.get("data_file_base_name")
		# ttv_file of ttvs: "ttv_dir + psi_" + validation --> ttv_file = "psi_"
		self.ttv_file = configs.get("ttv_file")
		# index of ttv_file: e.g. 2 for train2, test2, validation2 etc.
		self.index = configs.get("index")
		self.data_normalization_function = configs.get("data_normalization_function")
		
#		self.checkpoint = configs["checkpoint"]
#		self.meta_graph = configs["meta_graph"]
#		self.train = configs["train"]
#		self.predict = configs["predict"]
		self.layers = configs.get("parsed_layers")
		self.layer_count = len(self.layers)
		print("number of layers: " + str(self.layer_count))
		
		if self.layers[len(self.layers) - 1].window_size != self.ss_features:
			print("Error: output_channels unequal ss_features")
			print(str(self.layers[len(self.layers) - 1].output_channels) + " != " + str(self.ss_features))
			sys.exit()
		
		self.tf_layers = list()
		self.output_directory = self.output_directory + "/" + self.name + "/"
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		else:
			print("Error! Network already exists! Sure you want to override?!?")
			sys.exit(1)
		if not os.path.exists(self.output_directory + "save"):
			os.makedirs(self.output_directory + "save")
		if not os.path.exists(self.output_directory + "summary"):
			os.makedirs(self.output_directory + "summary")
		
	#	else:
	#		nw2.
		
		self.winner_acc = -1.0
		self.winner_loss = Infinity
		self.restored_graph = False
			
		self.start_index = 0
		self.build_graph()
		
def main(argv):
#	config_file = argv[0]
	config_file = "C:/Users/Dennis/Desktop/mut_config.txt"
	print(config_file)
	netw = mutable_network(config_file)

if __name__ == "__main__":
	main(sys.argv[1:])
