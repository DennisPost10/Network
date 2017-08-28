import os
import sys
import tempfile

from ProtFileParser2 import ProtFileParser2
import matplotlib.pyplot as plt
from network_c import nw1
import network_c
import numpy as np
from parser2 import InputParser2
import tensorflow as tf
from utils.ConfigFileParser import Configurations


class nw2:
	def one_hot(self, index):
		ret = np.zeros(3)
		ret[index] = 1
		return ret

	def to_ss(self, indices):
		ss = ""
		for i in indices:
			ss += self.sss[i]
		return ss

	def weight_variable(self, shape, name_val):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial, tf.float32, name=name_val)

	def bias_variable(self, shape, name_val):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial, tf.float32, name=name_val)

	def restore_graph(self, ckpt):
		with self.g2.as_default():
			tf.train.Saver().restore(self.sess2, ckpt)
			self.start_index = self.sess2.run(self.global_step2)
			self.restored_graph = True

	def restore_graph_without_knowing_checkpoint(self):
		self.ckpt = None
		for file in os.listdir(self.output_directory + "/save2/"):
			if file.endswith(".meta"):
				self.ckpt = self.output_directory + "/save2/" + os.path.splitext(file)[0]
		if(self.ckpt == None):
				print("Error: missing ckpt")
				sys.exit()
		print("ckpt: " + self.ckpt)
		self.restore_graph(self.ckpt)

	def build_graph(self):
		self.g2 = tf.Graph()
		with self.g2.as_default():
			self.keep_prob2 = tf.placeholder(tf.float32, name="keep_prob2")
			# input
			self.x2 = tf.placeholder(tf.float32, [None, 45], name="x2")
			# final output
			self.y2 = tf.placeholder(tf.float32, [None, 3], name="y2")

			# first fully connected layer with weights and biases using relu
			with tf.name_scope('first2'):
				self.W2 = self.weight_variable([45, 45], "weight2")
				self.b2 = self.bias_variable([45], "bias2")
				self.y_2 = tf.nn.relu(tf.matmul(self.x2, self.W2) + self.b2, name="layer2")
			# drop-out layer
			with tf.name_scope('drop_out2'):
				self.drop_out2 = tf.nn.dropout(self.y_2, self.keep_prob2, name="drop2")
			# second fully connected layer for 3-state output
			with tf.name_scope('second2'):
				self.w_p2 = self.weight_variable([45, 3], "weight2")
				self.b_p2 = self.bias_variable([3], "bias2")
				self.y_p2 = tf.add(tf.matmul(self.drop_out2, self.w_p2), self.b_p2, name="layer2")

			self.correct_prediction2 = tf.equal(tf.argmax(self.y_p2, 1), tf.argmax(self.y2, 1), name="correct_prediction2")
			self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32), name="accuracy2")
			self.observed2 = tf.argmax(self.y2, 1)
	
			self.h_count2 = tf.count_nonzero(tf.equal(tf.argmax(self.y2, 1), 0), name = "h_count2", dtype = tf.int32)
			self.c_count2 = tf.count_nonzero(tf.equal(tf.argmax(self.y2, 1), 1), name = "c_count2", dtype = tf.int32)
			self.e_count2 = tf.count_nonzero(tf.equal(tf.argmax(self.y2, 1), 2), name = "e_count2", dtype = tf.int32)
			self.h_accuracy2 = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y2,1), 0))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p2,1), 0))))))[1], self.h_count2, name = "h_accuracy2")
			self.c_accuracy2 = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y2,1), 1))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p2,1), 1))))))[1], self.c_count2, name = "c_accuracy2")
			self.e_accuracy2 = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(tf.argmax(self.y2,1), 2))), (tf.transpose(tf.where(tf.equal(tf.argmax(self.y_p2,1), 2))))))[1], self.e_count2, name = "e_accuracy2")
			self.global_step2 = tf.Variable(0, name='global_step2', trainable=False)
			self.loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y2, logits=self.y_p2), name="loss2")
	
			self.train_step2 = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum_val, name="train_step2").minimize(self.loss2, global_step=self.global_step2)
			self.init_op2 = tf.global_variables_initializer()
			self.saver2 = tf.train.Saver(max_to_keep=1)
			
			self.sess2 = tf.Session(graph = self.g2)
			self.sess2.run(self.init_op2)

	def train(self, training_file):
		with self.g2.as_default():
			self.prot_it = ProtFileParser2(training_file, self.prot_directory, self.output_directory, self.name)
		
			if self.restored_graph:
				loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess2.run([self.loss2, self.accuracy2, self.h_accuracy2, self.c_accuracy2, self.e_accuracy2], feed_dict={self.x2: self.prot_it.test_set, self.y2: self.prot_it.test_set_o, self.keep_prob2: 1})
				self.winner_acc = accuracy_eval
				self.winner_loss = loss_val
		
			tf.summary.scalar('loss2', self.loss2)
			tf.summary.scalar('accuracy2', self.accuracy2)
			tf.summary.scalar('h_accuracy2', self.h_accuracy2)
			tf.summary.scalar('c_accuracy2', self.c_accuracy2)
			tf.summary.scalar('e_accuracy2', self.e_accuracy2)
			summary = tf.summary.merge_all()
		
			summary_writer = tf.summary.FileWriter(self.output_directory + '/summary2', self.g2)
			batch_count = 0
			better = False
			for step in range(self.start_index, self.start_index + self.steps):
				if step % 100 == 0:
					summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess2.run([summary, self.loss2, self.accuracy2, self.h_accuracy2, self.c_accuracy2, self.e_accuracy2], feed_dict={self.x2: self.prot_it.test_set, self.y2: self.prot_it.test_set_o, self.keep_prob2: 1})
					if(accuracy_eval > self.winner_acc + 0.0001 or loss_val < self.winner_loss - 0.0001):
						self.winner_acc = accuracy_eval
						self.winner_loss = loss_val
						better = True
						self.saver2.save(self.sess2, (self.output_directory + '/save2/' + self.name), global_step=self.global_step2)
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
				_ = self.sess2.run(self.train_step2, feed_dict={self.x2: self.prot_it.next_batch_w, self.y2: self.prot_it.next_batch_o, self.keep_prob2: self.keep_prob_val})
				batch_count += 1
		
			self.saver2.save(self.sess2, (self.output_directory + '/save2/' + self.name), global_step=self.global_step2)
			summarystr, loss_val, accuracy_eval, h_acc, c_acc, e_acc = self.sess2.run([summary, self.loss2, self.accuracy2, self.h_accuracy2, self.c_accuracy2, self.e_accuracy2], feed_dict={self.x2: self.prot_it.test_set, self.y2: self.prot_it.test_set_o, self.keep_prob2: 1})
			print('Step %d: eval_accuracy = %.3f loss = %.3f H: %.3f C: %.3f E: %.3f (%d)' % (self.start_index + self.steps, accuracy_eval, loss_val, h_acc, c_acc, e_acc, batch_count))
			summary_writer.add_summary(summarystr, self.start_index + self.steps)

			print("training finished at maximum steps")
		
	def predict(self, test_file, nw_1):
		base = os.path.splitext(os.path.split(test_file)[1])[0]
		with self.g2.as_default():
			prediction2 = tf.argmax(self.y_p2, 1)

			hce_counts = np.zeros(3, dtype = float)
			hce_matches = np.zeros(3, dtype = float)

			number_correct = 0.0
			looked_at = 0.0

			scores_per_prot = []
			proteins = []
			missing_data_prots = []
			with open(test_file) as prots:
				for prot in prots:
					prot = prot.strip()
					proteins.append(prot)
					if not os.path.exists(self.prot_directory + "/" + prot + "/" + self.name + "/" + prot + ".predicted_ss_one_hot"):
						missing_data_prots.append(prot)
		
			if len(missing_data_prots) > 0:
				temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
				for p in missing_data_prots:
					temp_file.write(p + "\n")
				temp_file.seek(0)
				if nw_1 == None:
					nw_1 = nw1(self.config_file)
					nw_1.restore_graph_without_knowing_checkpoint()
				nw_1.predict(temp_file.name)
				temp_file.close()
				os.remove(temp_file.name)
				
			for prot in proteins:
				next_prot = InputParser2(self.prot_directory, prot, self.name)
				prot_nw_dir = self.prot_directory + "/" + prot + "/" + self.name + "/"
				pred_ss_final = open(prot_nw_dir + prot + ".predicted_ss_final", 'w')
				pred_ss_one_hot_final = open(prot_nw_dir + prot + ".predicted_ss_one_hot_final", 'w')
				obs, pred, corr = self.sess2.run([self.observed2, prediction2, self.correct_prediction2], feed_dict={self.x2: next_prot.getWindows(), self.y2: next_prot.getOutcomes(), self.keep_prob2: 1})
				pred_ss_final.write(self.to_ss(pred))
				for p in pred:
					a = self.one_hot(p)
					pred_ss_one_hot_final.write(str(a[0]) + "\t" + str(a[1]) + "\t" + str(a[2]) + "\n")
				corr_prot = 0.0
				for i in range(len(corr)):
					if corr[i]:
						corr_prot += 1
						number_correct += 1
						hce_matches[pred[i]] += 1
					hce_counts[obs[i]] += 1
				scores_per_prot.append(corr_prot / len(pred) * 100)
				looked_at += len(pred)
				pred_ss_final.close()
				pred_ss_one_hot_final.close()
		
			print(scores_per_prot)
			plt.figure()
			plt.hist(scores_per_prot)
			plt.savefig(self.output_directory + base + "_final_prot_scores.png")
				
			print("%d out of %d correct predicted (%.2f)" % (number_correct, looked_at, number_correct / looked_at))
		
	def __init__(self, config_file):
		
		self.config_file = config_file
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
		self.underlying_nw1 = None
		for file in os.listdir(self.output_directory + "/" + self.name + "/save/"):
			if file.endswith(".meta"):
				self.underlying_nw1 = self.output_directory + "/" + self.name + "/save/" + os.path.splitext(file)[0]
		if(self.underlying_nw1 == None):
				print("Error: missing underlying network")
				sys.exit()
		print("underlying network: " + self.underlying_nw1)		
		
		self.output_directory = self.output_directory + "/" + self.name + "/"
		if not os.path.exists(self.output_directory):
			os.makedirs(self.output_directory)
		if not os.path.exists(self.output_directory + "save2"):
			os.makedirs(self.output_directory + "save2")
		if not os.path.exists(self.output_directory + "summary2"):
			os.makedirs(self.output_directory + "summary2")
	
		self.sss = ['H', 'C', 'E']
				
		self.winner_acc = -1.0
		self.winner_loss = 1000
		self.restored_graph = False
		
		self.start_index = 0
		self.build_graph()

def main(argv):
	config_file = argv[0]
	print(config_file)
	netw = nw2(config_file)

if __name__ == "__main__":
	main(sys.argv[1:])
