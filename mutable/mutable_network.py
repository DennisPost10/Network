import os
import sys

import numpy as np
from stats import Stats
import tensorflow as tf
from utils.ConfigFileParser import Configurations
from utils.InputHandler import Input_Handler
from utils.Layer import Layer


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
		return tf.Variable(initial, tf.float64, name=name_val)

	def bias_variable(self, shape, name_val):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, tf.float64, name=name_val)

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
				hidden_layer = tf.nn.dropout(input_tf_layer, self.keep_prob_val, name="dropout")
			elif next_layer.layer_type == "fully":
				# print(input_layer.window_size)
				# print(next_layer.window_size)
				input_shape = input_tf_layer.get_shape().as_list()
				dim = np.prod(input_shape[1:])
				reshaped_input_layer = tf.reshape(input_tf_layer, [-1, dim])
				hidden_weights = self.weight_variable([dim, next_layer.window_size], "weight")
				bias = self.bias_variable([next_layer.window_size], "bias")
				hidden_layer = tf.add(tf.matmul(reshaped_input_layer, hidden_weights), bias, name="output_layer")
			elif next_layer.layer_type == "conv":
				hidden_weights = self.weight_variable([next_layer.window_size, input_layer.output_channels, next_layer.output_channels], "weight")
				# bias = self.bias_variable([next_layer.output_channels], "bias")
				hidden_layer = self.conv1d(input_tf_layer, hidden_weights, "layer")
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
			
				self.layer(Layer("x", "", False, self.features + self.aa_seq_add, self.window_size), self.x, 0)
			
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
					
			self.prot_lengths = tf.placeholder(tf.int64, [None], name="prot_lengths")
			
			self.last_layer = self.tf_layers[self.layer_count - 1]
			#!!!!!!!!!
			if self.network_type == "conv":
				self.observed = tf.argmax(self.y, 2) + 1
				self.predicted = tf.argmax(self.last_layer, 2) + 1
			else:
				self.observed = tf.argmax(self.y, 1) + 1
				self.predicted = tf.argmax(self.last_layer, 1) + 1
			
			with tf.name_scope("accuracy"):
				self.h_obs = tf.cast(tf.equal(self.observed, 1), tf.int64)
				self.c_obs = tf.cast(tf.equal(self.observed, 2), tf.int64)
				self.e_obs = tf.cast(tf.equal(self.observed, 3), tf.int64)
				self.h_count = tf.count_nonzero(self.h_obs, name="h_count", dtype=tf.int64)
				self.c_count = tf.count_nonzero(self.c_obs, name="c_count", dtype=tf.int64)
				self.e_count = tf.count_nonzero(self.e_obs, name="e_count", dtype=tf.int64)
			
			if self.network_type == "conv":
				
				# self.y_o = tf.nn.softmax(self.y_o, name="softmax")
				# self.y_o += tf.constant(1e-15)  # avoid zeros as input for log: log(0) = -inf -> null
				self.mat = tf.multiply(self.y, tf.log(self.last_layer))
				self.mat_shape = tf.shape(self.mat)
				self.mat = tf.reshape(self.mat, [self.mat_shape[0], -1])
				self.mask = tf.sequence_mask(self.prot_lengths * self.mat_shape[2], self.max_prot_length * self.mat_shape[2], dtype=tf.float64)
				self.mat = tf.multiply(self.mat, self.mask)
				self.mat = tf.reshape(self.mat, [self.mat_shape[0], self.mat_shape[1], self.mat_shape[2]])

				self.loss = tf.reduce_mean(-tf.reduce_sum(self.mat, reduction_indices=[1, 2]), name="loss")

				self.one_d_mask = tf.sequence_mask(self.prot_lengths, self.max_prot_length, tf.float64)
				self.correct_prediction = tf.multiply(tf.cast(tf.equal(self.observed, self.predicted), tf.float64), self.one_d_mask, name="correct_prediction")
				self.correct = tf.count_nonzero(self.correct_prediction, name="correct")
				self.accuracy = tf.divide(self.correct, tf.reduce_sum(self.prot_lengths), name="accuracy")
				self.predicted = tf.multiply(self.predicted, self.one_d_mask)

			else:
				
				with tf.name_scope("accuracy"):
					self.correct_prediction = tf.equal(self.predicted, self.observed, name="correct_prediction")
					self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float64), name="accuracy")
				with tf.name_scope("loss"):
					self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.last_layer), name="loss")
			
			with tf.name_scope("accuracy"):
				self.h_accuracy = tf.divide(tf.count_nonzero(tf.equal(tf.add(self.h_obs, tf.cast(tf.equal(self.predicted, 1), tf.int64)), 2)), self.h_count, name="h_accuracy")
				self.c_accuracy = tf.divide(tf.count_nonzero(tf.equal(tf.add(self.c_obs, tf.cast(tf.equal(self.predicted, 2), tf.int64)), 2)), self.c_count, name="c_accuracy")
				self.e_accuracy = tf.divide(tf.count_nonzero(tf.equal(tf.add(self.e_obs, tf.cast(tf.equal(self.predicted, 3), tf.int64)), 2)), self.e_count, name="e_accuracy")
	
			with tf.name_scope("global_step"):
				self.global_step = tf.Variable(0, name='global_step', trainable=False)
			
			self.train_step = self.init_optimizer().minimize(self.loss, global_step=self.global_step)
			
			self.saver = tf.train.Saver(max_to_keep=1)
			self.sess = tf.Session(graph=self.g)
			if not self.load_data:
				self.init_op = tf.global_variables_initializer()
				self.sess.run(self.init_op)
				self.saver.save(self.sess, (self.output_directory + 'save/' + self.name), global_step=self.global_step)
				self.saver.export_meta_graph(self.output_directory + 'save/' + self.name + "_meta.empty")
			else:
				self.restore_graph(self.meta_file)
				self.saver.save(self.sess, (self.output_directory + 'save_continued/' + self.name), global_step=self.global_step)
			tf.summary.FileWriter(self.output_directory + 'summary/' + self.name).add_graph(self.g)
			# python ~/.local/lib/python3.6/site-packages/tensorboard/main.py --logdir="/home/proj/tmp/postd/test/test_mixed/summary/"
			# C:\Program Files\Python36\Lib\site-packages\tensorboard\main.py  --logdir="D:\Dennis\ba\\summary\"
	def restore_graph(self, checkpoint):
		with self.g.as_default():
			print("restoring graph")
			self.saver.restore(self.sess, checkpoint)
			self.start_index = self.sess.run(self.global_step)
			self.restored_graph = True
			print("restored graph")
			print(self.sess.run(self.global_step))
	
	def restore_graph_without_knowing_checkpoint(self):
		self.ckpt = None
		for file in os.listdir(self.output_directory + "save/"):
			if file.endswith(".meta"):
				self.ckpt = self.output_directory + "save/" + os.path.splitext(file)[0]
		if(self.ckpt == None):
			print("Error: missing ckpt")
			sys.exit()
		print("ckpt: " + self.ckpt)
		self.restore_graph(self.ckpt)
		
	def train(self, log_file=None):
		
		if log_file != None:
			orig_stdout = sys.stdout
			f = open(log_file, 'w')
			sys.stdout = f
		
		self.prot_it = Input_Handler(self.prot_set, self.pssm_input_matrix, self.aa_seq_matrix, self.one_hots_matrix, self.index, self.train_file, self.val_file, self.test_file, self.max_prot_length, self.network_type, self.window_size, self.data_normalization_function, self.use_aa_seq_data, self.aa_codes, self.single_aa_seq)
		self.prot_it.load_ttv()
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
		
			summary_writer = tf.summary.FileWriter(self.output_directory + 'summary', self.g)
		
			batch_count = 0
			better = False
			
			check_range = 100
			alpha = 0.001
			# lower_acc = self.winner_acc
			lower_loss = self.winner_loss
			best_global_step = self.start_index
			
			for step in range(self.start_index, self.start_index + self.steps):
				if step % check_range == 0:
					summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
					# if(accuracy_eval - lower_acc > alpha or loss_val - lower_loss < alpha):
					# print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (step, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
					if(loss_val - lower_loss < alpha):
						self.winner_acc = max(self.winner_acc, accuracy_eval)
						self.winner_loss = min(self.winner_loss, loss_val)
						# lower_acc = accuracy_eval
						lower_loss = loss_val
						better = True
						best_global_step = self.sess.run(self.global_step)
						self.saver.save(self.sess, (self.output_directory + 'save/' + self.name), global_step=self.global_step)
				# if step % 1000 == 0:
						print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (step, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
						summary_writer.add_summary(summarystr, step)		
					if step % 10000 == 0:
						if better:
							better = False
						else:
							print("finished early")
							self.restore_graph(self.output_directory + 'save/' + self.name + "-" + str(best_global_step))
							print("restored graph from step " + str(self.sess.run(self.global_step)) + ": best_global=" + str(best_global_step))
							print("testing...")
							test_batch, test_batch_o, test_batch_l = self.prot_it.val_batches()
							accuracy_test, h_acc, c_acc, e_acc = self.sess.run([self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: test_batch, self.y: test_batch_o, self.prot_lengths: test_batch_l, self.keep_prob: 1})
							print('test_accuracy = %.3f H: %.3f C: %.3f E: %.3f' % (accuracy_test, h_acc, c_acc, e_acc))
							return
				
				ret_prots, ret_prots_o, ret_lengths = self.prot_it.next_prots(self.batch_size)
				_ = self.sess.run(self.train_step, feed_dict={self.x: ret_prots, self.y: ret_prots_o, self.keep_prob: self.keep_prob_val, self.prot_lengths: ret_lengths})
				batch_count += 1
		
			self.saver.save(self.sess, (self.output_directory + 'save/' + self.name), global_step=self.global_step)
			summary_writer.add_summary(summarystr, self.start_index + self.steps)
			summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
			print("training finished: reached maximum steps")
			print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (step, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
			
			self.restore_graph(self.output_directory + 'save/' + self.name + "-" + str(best_global_step))
			print("restored graph from step " + str(self.sess.run(self.global_step)) + ": best_global=" + str(best_global_step))
			
			print("testing...")
			test_batch, test_batch_o, test_batch_l = self.prot_it.val_batches()
			accuracy_test, h_acc, c_acc, e_acc = self.sess.run([self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: test_batch, self.y: test_batch_o, self.prot_lengths: test_batch_l, self.keep_prob: 1})
			print('test_accuracy = %.3f H: %.3f C: %.3f E: %.3f' % (accuracy_test, h_acc, c_acc, e_acc))
		
		if log_file != None:
			sys.stdout = orig_stdout
			f.close()
					
	def predict(self, test_file, stats_output_file, ss_output_file=None, log_file=None):
		prots = Input_Handler(self.prot_set, self.pssm_input_matrix, self.aa_seq_matrix, self.one_hots_matrix, self.index, self.train_file, self.val_file, self.test_file, self.max_prot_length, self.network_type, self.window_size, self.data_normalization_function, self.use_aa_seq_data, self.aa_codes, self.single_aa_seq)
		dat, ss_dat, lengths, prot_names = prots.get_prot_by_prot(test_file)
		
		if log_file != None:
			orig_stdout = sys.stdout
			f = open(log_file, 'w')
			sys.stdout = f

		with self.g.as_default():
			stat_file = open(stats_output_file, 'w')
			stat_file.write("prot\tlength\tobs_h\tobs_c\tobs_e\tpred_h\tpred_c\tpred_e\tcorrect\tcorrect_h\tcorrect_c\tcorrect_e\tmean_score\th_score\tc_score\te_score\tavg_score\tsov\th_sov\tc_sov\te_sov\n")
			if ss_output_file != None:
				ss_file = open(ss_output_file, 'w')
				ss_file.write("prot\tlength\tpredicted_ss\n")
				
			for i in range(len(dat)):
				# print("predicting protein: " + prot_names[i])
				obs, pred = self.sess.run([self.observed, self.predicted], feed_dict={self.x: dat[i], self.y: ss_dat[i], self.prot_lengths: [lengths[i]], self.keep_prob: 1})
				pred_ss = self.to_ss(pred - 1)
				obs_ss = self.to_ss(obs - 1)
				correct, correct_counts, counts_observed, counts_predicted, mean, h_score, c_score, e_score, avg = Stats.q3(obs_ss, pred_ss, lengths[i])
				h_sov, c_sov, e_sov, sov3 = Stats.sov(obs_ss, pred_ss, lengths[i])
				
				stat_file.write(prot_names[i] + "\t" + str(lengths[i]) + "\t")
				stat_file.write(str(counts_observed['H']) + "\t" + str(counts_observed['C']) + "\t" + str(counts_observed['E']) + "\t")
				stat_file.write(str(counts_predicted['H']) + "\t" + str(counts_predicted['C']) + "\t" + str(counts_predicted['E']) + "\t")
				stat_file.write(str(correct) + "\t" + str(correct_counts['H']) + "\t" + str(correct_counts['C']) + "\t" + str(correct_counts['E']) + "\t")
				stat_file.write(str(mean) + "\t" + str(h_score) + "\t" + str(c_score) + "\t" + str(e_score) + "\t" + str(avg) + "\t")
				stat_file.write(str(sov3) + "\t" + str(h_sov) + "\t" + str(c_sov) + "\t" + str(e_sov) + "\n")	
				
				if ss_output_file != None:
					ss_file.write(prot_names[i] + "\t" + str(lengths[i]) + "\t" + pred_ss + "\n")
		
				print('%s\tQ3: %.3f\tH: %.3f\tC: %.3f\tE: %.3f\tSOV: %.3f' % (prot_names[i], mean, h_score, c_score, e_score, sov3))
		
		if log_file != None:
			sys.stdout = orig_stdout
			f.close()
		
	def __init__(self, config_file, run):
		
		configs = Configurations(config_file).configs
		
		if(configs == None):
			print("Error: no input")
			sys.exit()
			
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
		self.single_aa_seq = configs.get("single_aa_seq")
		self.aa_codes = configs.get("aa_codes")
		self.aa_seq_add = self.aa_codes
		if not self.use_aa_seq_data:
			self.aa_seq_add = 0
		
		self.window_size = configs.get("window_size")
		
		self.prot_set = configs.get("prot_set")
		self.pssm_input_matrix = configs.get("pssm_matrix")
		self.aa_seq_matrix = configs.get("aa_seqs_matrix")
		self.one_hots_matrix = configs.get("one_hots_matrix")

		self.train_file = configs.get("train_file")
		self.val_file = configs.get("val_file")
		self.test_file = configs.get("test_file")

		self.index = configs.get("index")
		self.runs = configs.get("runs")
		self.data_normalization_function = configs.get("data_normalization_function")

		self.layers = configs.get("parsed_layers")
		self.layer_count = len(self.layers)
		print("number of layers: " + str(self.layer_count))
		
		last_layer = self.layers[self.layer_count - 1]
		if (last_layer.layer_type == "conv" and last_layer.output_channels != self.ss_features) or (last_layer.layer_type == "fully" and last_layer.window_size != self.ss_features):
			print("Error: output_channels unequal ss_features")
			print(last_layer.output_channels + " or " + str(last_layer.window_size) + " != " + str(self.ss_features))
			sys.exit()
		
		self.load_data = configs.get("load_data")
		self.meta_file = configs.get("load_meta_file")
		
		self.tf_layers = list()
		self.winner_acc = -1.0
		self.winner_loss = np.Infinity
		self.restored_graph = False
			
		self.start_index = 0

		self.output_directory = self.output_directory + "run_" + str(run) + "/"
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
			if not os.path.exists(self.output_directory + "save"):
				os.makedirs(self.output_directory + "save")
			if not os.path.exists(self.output_directory + "summary"):
				os.makedirs(self.output_directory + "summary")
		else:
			if not self.load_data:
				print("Error! Network already exists! Sure you want to override?!?")
			elif self.meta_file == None:
				print("Error! No given meta_file to reload!")
	# 	else:
	# 		nw2.
			
		self.build_graph()
		
def main(argv):
	# config_file = argv[0]
	# config_file = "/home/proj/tmp/postd/config2.file"
	# config_file = "/home/proj/tmp/postd/conv_config.file"
	# config_file = "/home/proj/tmp/postd/mixed_config.file"
	config_file = "D:/Dennis/ba/mixed_config.file"
	print(config_file)
	netw = mutable_network(config_file)

if __name__ == "__main__":
	main(sys.argv[1:])
