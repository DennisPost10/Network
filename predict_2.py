import os
import sys

import numpy as n
from runner2 import InputParser2
import tensorflow as tf


if len(sys.argv) < 6:
	print("Usage: prot_file prot_dir meta_graph_file ckpt_file nw_name")
	exit()

prot_file = sys.argv[1]
prot_dir = sys.argv[2]
graph_meta = sys.argv[3]
checkpoint_file = sys.argv[4]
nw_name = sys.argv[5]

print(sys.argv)

def one_hot(index):
	ret = n.zeros(3)
	ret[index] = 1
	return ret
	
sss = ['H', 'C', 'E']

def ss(index):
	return sss[index]

with open(prot_file) as f:
	prot_names = f.readlines()

prot_names = [x.strip() for x in prot_names]

with tf.Session() as sess:
		
	saver = tf.train.import_meta_graph(graph_meta)
	
	graph = tf.get_default_graph()

	# input
# 	x = tf.placeholder(tf.float32, [None,315], name = 'x')
	# final output
# 	y = tf.placeholder(tf.float32, [None, 3], name = 'y')

	x = graph.get_collection("x")[0]
	y = graph.get_collection("y")[0]

	W = graph.get_tensor_by_name("weight3:0")
	b = graph.get_tensor_by_name("bias3:0")

	y_ = tf.get_collection("first_layer")[0]
		
	W_p = graph.get_tensor_by_name("weight4:0")
	b_p = graph.get_tensor_by_name("bias4:0")

	y_p = tf.get_collection("second_layer")[0]

	# first fully connected layer with weights and biases using relu
# 	W = tf.Variable([315, 75], tf.float32, name = "weight1")
# 	b = tf.Variable([75], tf.float32, name = "bias1")
# 	y_ = tf.nn.relu(tf.matmul(x, W) + b)
	
	# second fully connected layer for 3-state output
# 	W_p = tf.Variable([75, 3], tf.float32, name = "weight2")
# 	b_p = tf.Variable([3], tf.float32, name = "bias2")
# 	y_p = tf.matmul(y_, W_p) + b_p
	
	saver.restore(sess, checkpoint_file)
	
	observed = tf.argmax(y, 1)
	prediction = tf.argmax(y_p, 1)
	eval_correct = tf.equal(tf.argmax(y_p, 1), tf.argmax(y, 1))
	
	
	number_correct = 0
	looked_at = 0

	for prot in prot_names:
		next_prot = InputParser2(prot_dir, prot, nw_name)
		prot_nw_dir = prot_dir + "/" + prot + "/" + nw_name + "/"
		pred_ss_final = open(prot_nw_dir + prot + ".predicted_ss_final", 'w')
		pred_ss_one_hot_final = open(prot_nw_dir + prot + ".predicted_ss_one_hot_final", 'w')
		# print(prot)
		for i in range(len(next_prot.getWindows())):
			next_window = n.reshape(next_prot.getWindows()[i], (1, 45)).astype(float)
			next_outcome = n.reshape(next_prot.getOutcomes()[i], (1, 3)).astype(float)
			obs, pred, corr = sess.run([observed, prediction, eval_correct], feed_dict={x: next_window, y: next_outcome})
			pred_ss_final.write(sss[pred[0]])
			a = one_hot(pred[0])
			pred_ss_one_hot_final.write(str(a[0]) + "\t" + str(a[1]) + "\t" + str(a[2]) + "\n")
			if corr:
				number_correct += 1
			looked_at += 1
		pred_ss_final.close()
		pred_ss_one_hot_final.close()
	print("%d out of %d correct predicted (%.2f)" % (number_correct, looked_at, float(float(number_correct) / float(looked_at))))
