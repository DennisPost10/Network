#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sys
from ProtFileParser2 import ProtFileParser2
import getopt
import os

def weight_variable(shape, name_val):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=name_val)

def bias_variable(shape, name_val):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=name_val)

def getInfo():
	return "nw2.py -t <training_prot_names> -p <parsed_prots_directory> -o <output_directory> -n name -l learning_rate -b batch_size -s steps"

def run(training_file, parsed_prots_directory, output_directory, name, learning_rate, batch_size, steps):
	
	x = tf.placeholder(tf.float32, [None, 45], name="x")
	
	y = tf.placeholder(tf.float32, [None, 3], name="y")
	
	W = weight_variable([45, 45], "weight3")
	b = bias_variable([45], "bias3")
	
	y_ = tf.nn.relu(tf.matmul(x, W) + b)
	
	w_p = weight_variable([45, 3], "weight4")
	b_p = bias_variable([3], "bias4")
	
	y_p = tf.matmul(y_, w_p) + b_p
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_p))
	
	global_step = tf.Variable(0, name='global_step', trainable=False)
	
	train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(loss, global_step=global_step)
	
	correct_prediction = tf.equal(tf.argmax(y_p, 1), tf.argmax(y, 1))
	
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
	init_op = tf.global_variables_initializer()
	
	tf.summary.scalar('loss', loss)
	tf.summary.scalar('accuracy', accuracy)
	summary = tf.summary.merge_all()

	data = ProtFileParser2(training_file, parsed_prots_directory, name)

	tf.add_to_collection("correct_prection", correct_prediction)
	tf.add_to_collection("accuracy", accuracy)
	tf.add_to_collection("first_layer", y_)
	tf.add_to_collection("second_layer", y_p)
	tf.add_to_collection("loss", loss)
	tf.add_to_collection("global_step", global_step)
	tf.add_to_collection("init_op", init_op)
	tf.add_to_collection("x", x)
	tf.add_to_collection("y", y)
	
	with tf.Session() as sess:
		
		sess.run(init_op)
		
		saver = tf.train.Saver(max_to_keep=1)
		
		summary_writer = tf.summary.FileWriter(output_directory + '/summary2', sess.graph)
		
		batch_count = 0
		for step in range(steps):
			if step % 1000 == 0:
				saver.save(sess, (output_directory + '/save2/' + name), global_step=global_step)
			if step % 100 == 0:
				summarystr, accuracy_eval = sess.run([summary, accuracy], feed_dict={x: data.test_set, y: data.test_set_o})
				print('Step %d: eval_accuracy = %.2f (%d)' % (step, accuracy_eval, batch_count))
				summary_writer.add_summary(summarystr, step)
				
			data.next_batch(batch_size)
			batch_count += 1
			_ = sess.run([train_step], feed_dict={x: data.next_batch_w, y: data.next_batch_o})

		saver.save(sess, (output_directory + '/save2/' + name), global_step=global_step)
		summarystr, accuracy_eval = sess.run([summary, accuracy], feed_dict={x: data.test_set, y: data.test_set_o})
		print('Step %d: eval_accuracy = %.2f (%d)' % (steps, accuracy_eval, batch_count))
		summary_writer.add_summary(summarystr, steps)

	print("training finished")

def main(argv):
	print(argv)
	predicted_directory = None
	training_file = None
	output_directory = None
	prot_directory = None
	adam_opti = False
	name = ""
	learning_rate = 1e-3
	try:
		opts, args = getopt.getopt(argv, "ht:p:o:n:l:b:s:")
	except getopt.GetoptError:
		print(getInfo())
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print(getInfo())
			sys.exit()
		elif opt == '-t':
			training_file = arg
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
	if(training_file == None or output_directory == None or prot_directory == None):
		print(getInfo())
		sys.exit()
	output_directory = output_directory + "/" + name + "/"
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)
	if not os.path.exists(output_directory + "save2"):
		os.makedirs(output_directory + "save2")
	if not os.path.exists(output_directory + "summary2"):
		os.makedirs(output_directory + "summary2")
	print('training file is ' + training_file)
	print('prot directory is' + prot_directory)
	print('output directory is ' + output_directory)
	run(training_file, prot_directory, output_directory, name, learning_rate, batch_size, steps)

if __name__ == "__main__":
	main(sys.argv[1:])

