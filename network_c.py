import os
import sys

from ProtFileParser1 import ProtFileParser
import matplotlib.pyplot as plt
import numpy as np
from parser1 import InputParser
import tensorflow as tf
from utils.ConfigFileParser import Configurations


sss = ['H', 'C', 'E']

class nw1:

	def one_hot(self, index):
		ret = np.zeros(3)
		ret[index] = 1
		return ret

	def to_ss(self, indices):
		ss = ""
		for i in indices:
			ss += sss[i]
		return ss

	def weight_variable(self, shape, name_val):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, tf.float32, name=name_val)

	def bias_variable(self, shape, name_val):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, tf.float32, name=name_val)

	def build_graph(self):
		self.g = tf.Graph()
		with self.g.as_default():
			self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
			# input
			self.x = tf.placeholder(tf.float32, [None, 300], name="x")
			# final output
			self.y = tf.placeholder(tf.float32, [None, 3], name="y")
			# first fully connected layer with weights and biases using relu
			with tf.name_scope('first'):
				self.W = self.weight_variable([300, 75], "weight")
				self.b = self.bias_variable([75], "bias")
				self.y_ = tf.nn.relu(tf.matmul(self.x, self.W) + self.b, name="layer")
			# drop-out layer
			with tf.name_scope('drop_out'):
				self.drop_out = tf.nn.dropout(self.y_, self.keep_prob, name="drop")

			# second fully connected layer for 3-state output
			with tf.name_scope('second'):
				self.W_p = self.weight_variable([75, 3], "weight")
				self.b_p = self.bias_variable([3], "bias")
				self.y_p = tf.add(tf.matmul(self.drop_out, self.W_p), self.b_p, name="layer")

			self.correct_prediction = tf.equal(tf.argmax(self.y_p, 1), tf.argmax(self.y, 1), name="correct_prediction")
			self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
			self.observed = tf.argmax(self.y, 1)
			self.h_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 0), name = "h_count", dtype = tf.int32)
			self.c_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 1), name = "c_count", dtype = tf.int32)
			self.e_count = tf.count_nonzero(tf.equal(tf.argmax(self.y, 1), 2), name = "e_count", dtype = tf.int32)
			self.h_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 0))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p,1), 0))))))[1], self.h_count, name = "h_accuracy")
			self.c_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 1))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p,1), 1))))))[1], self.c_count, name = "c_accuracy")
			self.e_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y,1), 2))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p,1), 2))))))[1], self.e_count, name = "e_accuracy")
			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_p), name="loss")
			self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum_val, name="train_step").minimize(self.loss, global_step=self.global_step)
			self.init_op = tf.global_variables_initializer()
			self.saver = tf.train.Saver(max_to_keep=1)
			
			self.sess = tf.Session(graph = self.g)
			self.sess.run(self.init_op)
	
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
		
	def train(self, training_file):
		
		self.prot_it = ProtFileParser(training_file, self.prot_directory)
		with self.g.as_default():
			if self.restored_graph:
				loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.prot_it.test_set, self.y: self.prot_it.test_set_o, self.keep_prob: 1})
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
			for step in range(self.start_index, self.start_index + self.steps):
				if step % 100 == 0:
					summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.prot_it.test_set, self.y: self.prot_it.test_set_o, self.keep_prob: 1})
					if(accuracy_eval > self.winner_acc or loss_val < self.winner_loss):
						self.winner_acc = accuracy_eval
						self.winner_loss = loss_val
						better = True
						self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
					if step % 1000 == 0:
						print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (step, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
						summary_writer.add_summary(summarystr, step)			
					if step % 10000 == 0:
						if better:
							better = False
						else:
							print("finished early")
							return
				
				self.prot_it.next_batch(self.batch_size)
				_ = self.sess.run(self.train_step, feed_dict={self.x: self.prot_it.next_batch_w, self.y: self.prot_it.next_batch_o, self.keep_prob: self.keep_prob_val})
				batch_count += 1
		
			self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
			summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess.run([summary, self.loss, self.accuracy, self.h_accuracy, self.c_accuracy, self.e_accuracy], feed_dict={self.x: self.prot_it.test_set, self.y: self.prot_it.test_set_o, self.keep_prob: 1})
			print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (self.start_index + self.steps, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
			summary_writer.add_summary(summarystr, self.start_index + self.steps)

			print("training finished")
		
	def predict(self, test_file):
		base = os.path.splitext(os.path.split(test_file)[1])[0]
		with self.g.as_default():
			prediction = tf.argmax(self.y_p, 1)

			hce_counts = np.zeros(3, dtype = float)
			hce_matches = np.zeros(3, dtype = float)

			number_correct = 0.0
			looked_at = 0.0
			
			scores_per_prot = []
			with open(test_file, 'r') as prots:
				for prot in prots:
					prot = prot.strip()
					print(prot)
					next_prot = InputParser(self.prot_directory, prot)
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
		self.prot_directory = configs["protein_directory"]
		self.output_directory = configs["output_directory"]
		self.name = configs["name"]
		self.learning_rate = configs["learning_rate"]
		self.batch_size = configs["batch_size"]
		self.steps = configs["max_steps"]
		self.keep_prob_val = configs["keep_prob"]
		self.momentum_val = configs["momentum"]
#		self.checkpoint = configs["checkpoint"]
#		self.meta_graph = configs["meta_graph"]
#		self.train = configs["train"]
#		self.predict = configs["predict"]
		
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
		self.winner_loss = 1000	
		self.restored_graph = False
			
		self.start_index = 0
		self.build_graph()
		
def main(argv):
	config_file = argv[0]
	print(config_file)
	netw = nw1(config_file, None)

if __name__ == "__main__":
	main(sys.argv[1:])
