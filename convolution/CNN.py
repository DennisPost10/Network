import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from parser1 import InputParser
import tensorflow as tf
from utils.ConfigFileParser import Configurations
from utils.InputHandler import Input_Handler


def conv1d(x, neurons, name_val):
    return tf.nn.conv1d(x, filters=neurons, stride=1, padding='SAME', name=name_val)

def weight_variable(shape, name_val):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, tf.float32, name=name_val)

def bias_variable(shape, name_val):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, tf.float32, name=name_val)

class CNN():    
    
    def build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        # input
            self.x = tf.placeholder(tf.float32, [None, self.max_prot_length, 20], name="x")
        # final output
            self.y = tf.placeholder(tf.float32, [None, self.max_prot_length, 3], name="y")

            with tf.name_scope('conv_1'):
                self.W = weight_variable([self.window_sizes[0], 20, self.neurons[0]], "weight")
                self.b = bias_variable([self.neurons[0]], "bias")
                self.y_ = tf.nn.relu(conv1d(self.x, self.W, "conv_1"), name="layer")
                print(self.y_.shape)
            # drop-out layer
#            with tf.name_scope('drop_out'):
#                self.drop_out = tf.nn.dropout(self.y_, self.keep_prob, name="drop")

            with tf.name_scope('conv_2'):
                self.W_p = weight_variable([self.window_sizes[1], self.neurons[0], self.neurons[1]], "weight")
                self.b_p = bias_variable([self.neurons[1]], "bias")
                self.y_p = tf.nn.relu(conv1d(self.y_, self.W_p, "conv_2"), name="layer")
                print(self.y_p.shape)

#            with tf.name_scope('flat'):
#                self.y_f = tf.reshape(self.y_p, [tf.shape(self.y_p)[0], int(self.neurons[1] * self.max_prot_length)], 'flat')
#                print(self.y_f.shape)
            
#            with tf.name_scope('fully'):
#                self.W_o = weight_variable([self.neurons[1] * self.max_prot_length, 3 * self.max_prot_length], "weight")
#                self.b_o = bias_variable([3 * self.max_prot_length], "bias")
#                self.y_o = tf.add(tf.matmul(self.y_f, self.W_o), self.b_o, name="layer")
#                print(self.y_o.shape)
#                self.y_o = tf.reshape(self.y_o, [-1, self.max_prot_length, 3])
#                print(self.y_o.shape)
            
            with tf.name_scope('conv_out'):
                self.W_o = weight_variable([self.max_prot_length, self.neurons[1], 3], "weight")
                self.b_o = bias_variable([3], "bias")
                self.y_o = conv1d(self.y_p, self.W_o, "conv_out")
                print(self.y_o.shape)



            self.prot_lengths = tf.placeholder(tf.int64, [None], name="prot_lengths")
            self.y_o = tf.nn.softmax(self.y_o, name="softmax")
            self.y_o += tf.constant(1e-15)  # avoid zeros as input for log: log(0) = -inf -> null
            self.mat = tf.multiply(self.y, tf.log(self.y_o))
            self.batch_s = tf.shape(self.mat)[0]
            self.mat = tf.reshape(self.mat, [self.batch_s, -1])
#            self.y_o = tf.reshape(self.y_o, [self.sess.run(tf.shape(self.y_o)[0]), -1])
            self.mask = tf.contrib.crf._lengths_to_masks(self.prot_lengths * 3, self.max_prot_length * 3)
            self.mat = tf.multiply(tf.cast(self.mat, tf.float32), self.mask)
            self.mat = tf.reshape(self.mat, [self.batch_s, self.max_prot_length, 3])
            
            self.loss = tf.reduce_mean(-tf.reduce_sum(self.mat, reduction_indices=[1, 2]), name="loss")
        
            self.one_d_mask = tf.contrib.crf._lengths_to_masks(self.prot_lengths, self.max_prot_length)
        
            self.observed = tf.argmax(self.y, 2) + 1
            self.predicted = tf.argmax(self.y_o, 2) + 1

            self.correct_prediction = tf.multiply(tf.cast(tf.equal(self.observed, self.predicted), tf.float32), self.one_d_mask, name="correct_prediction")
#            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32), name="accuracy")
            self.correct = tf.count_nonzero(self.correct_prediction, name="correct")
            
            self.accuracy = tf.divide(self.correct, tf.reduce_sum(self.prot_lengths), name="accuracy")

            self.masked_y = tf.multiply(tf.cast(self.observed, tf.float32), self.one_d_mask)

            self.masked_y_o = tf.multiply(tf.cast(self.predicted, tf.float32), self.one_d_mask)

            self.h_count = tf.count_nonzero(tf.equal(self.masked_y, 1), name="h_count", dtype=tf.int32)
            self.c_count = tf.count_nonzero(tf.equal(self.masked_y, 2), name="c_count", dtype=tf.int32)
            self.e_count = tf.count_nonzero(tf.equal(self.masked_y, 3), name="e_count", dtype=tf.int32)
            
            self.h_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(self.masked_y, 1))), (tf.transpose(tf.where(tf.equal(self.masked_y_o, 1))))))[1], self.h_count, name="h_accuracy")
            self.c_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(self.masked_y, 2))), (tf.transpose(tf.where(tf.equal(self.masked_y_o, 2))))))[1], self.c_count, name="c_accuracy")
            self.e_accuracy = tf.divide(tf.shape(tf.sets.set_intersection(tf.transpose(tf.where(tf.equal(self.masked_y, 3))), (tf.transpose(tf.where(tf.equal(self.masked_y_o, 3))))))[1], self.e_count, name="e_accuracy")
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            
#            self.train_step = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=self.momentum_val, name="train_step").minimize(self.loss, global_step=self.global_step)
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate, name="train_step").minimize(self.loss, global_step=self.global_step)
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=1)
            
            self.sess = tf.Session(graph=self.g)
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
        
    def train(self):
        
#        self.prot_it = CNN_Inputparser(training_file, self.prot_directory, self.max_prot_length)
        self.prot_it = Input_Handler("D:/Dennis/Uni/bachelor/data/prot_data/psi_prot_name_files/dat_files/", "psi_prots", "D:/Dennis/Uni/bachelor/data/prot_data/psi_prot_name_files/dat_files/", 1, self.max_prot_length, True, 11, "", None)
        
        self.val_batch, self.val_batch_o, self.val_batch_l = self.prot_it.val_batches()
        
        with self.g.as_default():
            if self.restored_graph:
                loss_val, accuracy_eval = self.sess.run([self.loss, self.accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
                self.winner_acc = accuracy_eval
                self.winner_loss = loss_val            

            batch_count = 0
            for step in range(self.start_index, self.start_index + self.steps):
                if step % 1 == 0:
                    train_batch, train_batch_o, train_batch_l = self.prot_it.next_prots(self.batch_size)
                    loss_val, accuracy_eval, p, o, mat = self.sess.run([self.loss, self.accuracy, self.masked_y_o, self.masked_y, self.mat], feed_dict={self.x: train_batch, self.y: train_batch_o, self.prot_lengths: train_batch_l, self.keep_prob: 1})
                    print('Step %d: eval_accuracy = %.3f loss = %.3f' % (step, accuracy_eval, loss_val))
                    print(p[4])
                    print(o[4])
                    print(mat[4])
#                    print("next")
#                    fail=np.sum( np.isnan(mat))
#                    if fail > 0:
#                        print(mat)
#                        print(log_yo)
#                        print(out)
#                        print(o)
#                    print(np.sum(np.isnan(log_yo)))
#                    print(np.sum(np.isnan(out)))
#                    print(np.sum(np.isnan(o)))
                    
#                    print(out[3])
                    if(accuracy_eval > self.winner_acc or loss_val < self.winner_loss):
                        self.winner_acc = accuracy_eval
                        self.winner_loss = loss_val

                ret_prots, ret_prots_o, lengths = self.prot_it.next_prots(self.batch_size)
                _ = self.sess.run(self.train_step, feed_dict={self.x: ret_prots, self.y: ret_prots_o, self.keep_prob: self.keep_prob_val, self.prot_lengths: lengths})
                batch_count += 1
        
            self.saver.save(self.sess, (self.output_directory + '/save/' + self.name), global_step=self.global_step)
            loss_val, accuracy_eval = self.sess.run([self.loss, self.accuracy], feed_dict={self.x: self.val_batch, self.y: self.val_batch_o, self.prot_lengths: self.val_batch_l, self.keep_prob: 1})
            print('Step %d: eval_accuracy = %.3f loss = %.3f' % (step, accuracy_eval, loss_val))

            print("training finished")
            
    def predict(self, test_file):
        base = os.path.splitext(os.path.split(test_file)[1])[0]
        with self.g.as_default():
            prediction = tf.argmax(self.y_p, 1)

            hce_counts = np.zeros(3, dtype=float)
            hce_matches = np.zeros(3, dtype=float)

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

#        self.training_file = configs["training_file"]
#        self.test_file = configs["test_file"]
#        self.prot_directory = configs["protein_directory"]
        self.output_directory = configs["output_directory"]
        self.name = configs["name"]
        self.learning_rate = configs["learning_rate"]
        self.batch_size = configs["batch_size"]
        self.steps = configs["max_steps"]
        self.keep_prob_val = configs["keep_prob"]
        self.momentum_val = configs["momentum"]
        self.window_sizes = configs["window_sizes"]
        self.neurons = configs["neurons"]
        self.max_prot_length = configs["max_prot_length"]
#        self.checkpoint = configs["checkpoint"]
#        self.meta_graph = configs["meta_graph"]
#        self.train = configs["train"]
#        self.predict = configs["predict"]
        
        self.output_directory = self.output_directory + "/" + self.name + "/"
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        else:
            print(self.output_directory)
            print("Error! Network already exists! Sure you want to override?!?")
            sys.exit(1)
        if not os.path.exists(self.output_directory + "save"):
            os.makedirs(self.output_directory + "save")
        if not os.path.exists(self.output_directory + "summary"):
            os.makedirs(self.output_directory + "summary")
        
    #    else:
    #        nw2.
        
        self.winner_acc = -1.0
        self.winner_loss = 1000    
        self.restored_graph = False
            
        self.start_index = 0
        self.build_graph()
        
def main(argv):
#    config_file = argv[0]
    config_file = "C:/Users/Dennis/Desktop/mut_config.txt"
    print(config_file)
    cnn = CNN(config_file)
    cnn.train()

if __name__ == "__main__":
    main(sys.argv[1:])
