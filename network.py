#!/usr/bin/env python

import tensorflow as tf
from ProtFileParser1 import ProtFileParser
import sys
import numpy as np
import getopt
import os

def getInfo():
	return "test.py -t <training_prot_name_file> -e <test_prot_name_file> -p <parsed_prots_directory> -o <output_directory> -n name -l learning_rate -b batch_size -s steps"

keep_prob = 0.9

def weight_variable(shape, name_val):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, tf.float32, name=name_val)

def bias_variable(shape, name_val):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, tf.float32, name=name_val)

def run(test_file, training_file, prot_directory, output_directory, adam_opti, name, learning_rate, batch_size, steps):
	
	tf.logging.set_verbosity(tf.logging.INFO)

	# input
	x = tf.placeholder(tf.float32, [None, 300], name="x")
	# final output
	y = tf.placeholder(tf.float32, [None, 3], name="y")
	
	# first fully connected layer with weights and biases using relu
	W = weight_variable([300, 75], "weight1")
	b = bias_variable([75], "bias1")
	y_ = tf.nn.relu(tf.matmul(x, W) + b)

	# second fully connected layer for 3-state output
	W_p = weight_variable([75, 3], "weight2")
	b_p = bias_variable([3], "bias2")
	y_p = tf.matmul(y_, W_p) + b_p

	# loss function applying softmax to y_p
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_p))
	
	global_step = tf.Variable(0, name='global_step', trainable=False)
	
	# train step
	train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
	
	init_op = tf.global_variables_initializer()

	correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y, 1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	tf.add_to_collection("correct_prection", correct_prediction)
	tf.add_to_collection("accuracy", accuracy)
	tf.add_to_collection("first_layer", y_)
	tf.add_to_collection("second_layer", y_p)
	tf.add_to_collection("loss", loss)
	tf.add_to_collection("global_step", global_step)
	tf.add_to_collection("init_op", init_op)
	tf.add_to_collection("x", x)
	tf.add_to_collection("y", y)

	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)
	summary = tf.summary.merge_all()

	with tf.Session() as sess:
	
		sess.run(init_op)
		
# 		saver = tf.train.Saver({'weight1':W, 'bias1':b, 'weight2':W_p, 'bias2': b_p, 'global_step': global_step}, max_to_keep = 100000)
		saver = tf.train.Saver(max_to_keep=1)

		summary_writer = tf.summary.FileWriter(output_directory + '/summary', sess.graph)

		prot_it = ProtFileParser(training_file, prot_directory)

# next_prot = prot_it.next_prot()
# bases_trained = 0
# prots_trained = 0
	
# test_acc = open("/home/p/postd/bachelor/data/formatted/parsed/test1.acc", "w")
		batch_count = 0
		for step in range(steps):
			if step % 1000 == 0:
				saver.save(sess, (output_directory + '/save/' + name), global_step=global_step)
			if step % 100 == 0:
				summarystr, accuracy_eval = sess.run([summary, accuracy], feed_dict={x: prot_it.test_set, y: prot_it.test_set_o})
				print('Step %d: eval_accuracy = %.2f (%d)' % (step, accuracy_eval, batch_count))
				summary_writer.add_summary(summarystr, step)
			
			prot_it.next_batch(batch_size)
			_, loss_val = sess.run([train_step, loss], feed_dict={x: prot_it.next_batch_w, y: prot_it.next_batch_o})
			batch_count += 1
		
		saver.save(sess, (output_directory + '/save/' + name), global_step=global_step)
		summarystr, accuracy_eval = sess.run([summary, accuracy], feed_dict={x: prot_it.test_set, y: prot_it.test_set_o})
		print('Step %d: eval_accuracy = %.2f (%d)' % (steps, accuracy_eval, batch_count))
		summary_writer.add_summary(summarystr, steps)

	print("training finished")

def main(argv):
	print(argv)
	test_file = None
	training_file = None
	prot_directory = None
	output_directory = None
	adam_opti = False
	name = ""
	learning_rate = 1e-3
	batch_size = 100
	steps = 100000
	try:
		opts, args = getopt.getopt(argv, "ht:e:p:o:a:n:l:b:s:")
	except getopt.GetoptError:
		print(getInfo())
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(getInfo())
			sys.exit()
		elif opt == '-t':
			training_file = arg
		elif opt == '-e':
			test_file = arg
		elif opt == '-p':
			prot_directory = arg
		elif opt == '-o':
			output_directory = arg
		elif opt == '-n':
			name = arg
		elif opt == '-l':
			learning_rate = float(arg)
		elif opt == '-b':
			batch_size = int(arg)
		elif opt == '-s':
			steps = int(arg)	
	if(training_file == None or prot_directory == None or output_directory == None):
		print(getInfo())
		sys.exit()
	output_directory = output_directory + "/" + name + "/"
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	if not os.path.exists(output_directory + "save"):
		os.makedirs(output_directory + "save")
	if not os.path.exists(output_directory + "summary"):
		os.makedirs(output_directory + "summary")
	print('training file is ' + training_file)
	print('test file is ' + test_file)
	print('prot directory is' + prot_directory)
	print('output directory is ' + output_directory)
	print('name is ' + name)
	print('learning_rate is ' + str(learning_rate))
	run(test_file, training_file, prot_directory, output_directory, adam_opti, name, learning_rate, batch_size, steps)

if __name__ == "__main__":
	main(sys.argv[1:])

