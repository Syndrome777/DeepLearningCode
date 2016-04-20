#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import numpy


# Min/max sequence length
MIN_LENGTH = 40
MAX_LENGTH = 50
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 256
# Number of training sequences in each batch
N_BATCH = 128
# Optimization learning rate
LEARNING_RATE = 0.001
# All gradients above this will be clipped
GRAD_CLIP = 10
# How often should we check the output?
EPOCH_SIZE = 1000
# Number of epochs to train the net
NUM_EPOCHS = 10000000



###########  Word2Vec Matrix
f_ = open('\\\\amhpcfile02\\data\\users\\v-lifenh\\Dataset\\EmotionClassification\\Data_less_80_more_2.dump', 'rb')
Test_Dialogue = cPickle.load(f_)
Train_Dialogue = cPickle.load(f_)

Word2Vec = cPickle.load(f_)
print ('Test_Dialogue')
print (len(Test_Dialogue))
print ('Train_Dialogue')
print (len(Train_Dialogue))

f_.close()

#Word2Vec_ = numpy.asarray(Word2Vec, dtype=theano.config.floatX)
#Word2Vec = theano.shared(Word2Vec_)

WORD_NUM = 10000
INPUT_DIM = 128


Index2Index = {0:0, 1:1, 2:1, 3:0, 4:1, 5:0, 6:0, 7:0}
Index2Index = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8}

def GetData(index, n_batch=N_BATCH, max_length=MAX_LENGTH, input_dim = INPUT_DIM):
	#X = np.zeros((N_BATCH, max_length, input_dim))
	max_length = Train_Dialogue[index][2]
	mask = np.zeros((n_batch, max_length))
	#y = np.zeros((n_batch,))
	# Compute masks and correct values
	batch_data = Train_Dialogue[index : index + n_batch]
	
	input_ = numpy.zeros((n_batch, max_length), dtype=numpy.int32)
	output_ = numpy.zeros((n_batch), dtype=numpy.int32)

	for i in range(n_batch):
		# Randomly choose the sequence length
		c, l, length = batch_data[i]
		c = c[:max_length]
		for j in xrange(len(c)):
			input_[i, j] = c[j]
		output_[i] = Index2Index[l]
		# Make the mask for this sample 1 within the range of length
		mask[i, :length] = 1				

	return (input_.astype(numpy.int32), output_.astype(numpy.int32),
			mask.astype(theano.config.floatX))

			
def GetData2(index, n_batch=N_BATCH, max_length=MAX_LENGTH, input_dim = INPUT_DIM):
	#X = np.zeros((N_BATCH, max_length, input_dim))
	max_length = Test_Dialogue[index][2]
	mask = np.zeros((n_batch, max_length))
	#y = np.zeros((n_batch,))
	# Compute masks and correct values
	batch_data = Test_Dialogue[index : index + n_batch]
	
	input_ = numpy.zeros((n_batch, max_length), dtype=numpy.int32)
	output_ = numpy.zeros((n_batch), dtype=numpy.int32)

	for i in range(n_batch):
		# Randomly choose the sequence length
		c, l, length = batch_data[i]
		c = c[:max_length]
		for j in xrange(len(c)):
			input_[i, j] = c[j]
		output_[i] = Index2Index[l]
		# Make the mask for this sample 1 within the range of length
		mask[i, :length] = 1				

	return (input_.astype(numpy.int32), output_.astype(numpy.int32),
			mask.astype(theano.config.floatX))


			
			
def main(num_epochs=NUM_EPOCHS):
	print("Building network ...")
	# First, we build the network, starting with an input layer
	# Recurrent layers expect input of shape
	# (batch size, max sequence length, number of features)
	#l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH, INPUT_DIM))
	x_sym = T.imatrix()
	target_values = T.ivector('target_output')
	xmask_sym = T.matrix()
	l_in = lasagne.layers.InputLayer((None, None), input_var = x_sym)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, WORD_NUM, INPUT_DIM, 
									  #W=Word2Vec_,
									  name='Embedding')
	#Here we'll remove the trainable parameters from the embeding layer to constrain 
	#it to a simple "one-hot-encoding". You can experiment with removing this line
	#l_emb.params[l_emb.W].remove('trainable')
	# The network also needs a way to provide a mask for each sequence.  We'll
	# use a separate input layer for that.  Since the mask only determines
	# which indices are part of the sequence for each batch entry, they are
	# supplied as matrices of dimensionality (N_BATCH, MAX_LENGTH)
	l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var = xmask_sym)
	# We're using a bidirectional network, which means we will combine two
	# RecurrentLayers, one with the backwards=True keyword argument.
	# Setting a value for grad_clipping will clip the gradients in the layer
	# Setting only_return_final=True makes the layers only return their output
	# for the final time step, which is all we need for this task
	l_forward = lasagne.layers.LSTMLayer(
		l_emb, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
		cell_init=lasagne.init.HeUniform(),
		hid_init=lasagne.init.HeUniform(),
		nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, name = 'l_forward')

	l_backward = lasagne.layers.LSTMLayer(
		l_emb, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
		cell_init=lasagne.init.HeUniform(),
		hid_init=lasagne.init.HeUniform(),
		nonlinearity=lasagne.nonlinearities.tanh,
		only_return_final=False, backwards=True, name = 'l_backward')
		
	l_concat_forward = lasagne.layers.ConcatLayer([l_emb, l_forward], axis=2)
	l_concat_backward = lasagne.layers.ConcatLayer([l_emb, l_backward], axis=2)

	l_forward2 = lasagne.layers.LSTMLayer(
		l_concat_forward, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
		cell_init=lasagne.init.HeUniform(),
		hid_init=lasagne.init.HeUniform(),
		nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True, name = 'l_forward')

	l_backward2 = lasagne.layers.LSTMLayer(
		l_concat_backward, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP,
		cell_init=lasagne.init.HeUniform(),
		hid_init=lasagne.init.HeUniform(),
		nonlinearity=lasagne.nonlinearities.tanh,
		only_return_final=True, backwards=True, name = 'l_backward')
	# Now, we'll concatenate the outputs to combine them.
	l_concat = lasagne.layers.ConcatLayer([l_forward2, l_backward2])
	# Our output layer is a simple dense connection, with 1 output unit
	softmax = lasagne.nonlinearities.softmax
	l_out = lasagne.layers.DenseLayer(
		lasagne.layers.dropout(l_concat, p=.25), num_units=9, nonlinearity=softmax, name = 'l_out')


	# lasagne.layers.get_output produces a variable for the output of the net
	network_output = lasagne.layers.get_output(l_out)
	# The network output will have shape (n_batch, 1); let's flatten to get a
	# 1-dimensional vector of predicted values
	#predicted_values = network_output.flatten()
	# Our cost will be mean-squared error
	cost = lasagne.objectives.categorical_crossentropy(network_output, target_values)
	cost = cost.mean()
	predict_l = T.argmax(network_output, axis=1)
	#cost = T.mean((predicted_values - target_values)**2)
	# Retrieve all parameters from the network
	all_params = lasagne.layers.get_all_params(l_out, trainable=True)
	# Compute SGD updates for training
	print("Computing updates ...")
	updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
	# Theano functions for training and computing cost
	print("Compiling functions ...")
	train = theano.function([x_sym, target_values, xmask_sym],
							cost, updates=updates)
	#############################################################################################################						
	test_prediction = lasagne.layers.get_output(l_out, deterministic=True)
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values),
					  dtype=theano.config.floatX)
	predict_test = T.argmax(test_prediction, axis=1)
	compute_cost = theano.function(
		[x_sym, target_values, xmask_sym], [test_loss, predict_test])

	###############################################################################################
	
	#f_ = open('./128_Emotion_LSTM_Model_not_sorted.dump_0.001_91_0.800257731959', 'rb')
	#all_param_values = cPickle.load(f_)
	#f_.close()
	#lasagne.layers.set_all_param_values(l_out, all_param_values)
	
	
	print("Trainable Model Parameters")
	print("-"*40)
	for param in all_params:
		print(param, param.get_value().shape)
	print("-"*40)
	
	total_num = len(Train_Dialogue)
	data_index = [i for i in xrange(total_num - 20 - N_BATCH)]
	data_index = data_index[0:len(data_index):16]
	#data_index = data_index[1000:1010]
	#data_index = data_index[0:len(data_index):10000]

	print("Number:{}, {}".format(len(data_index), len(Train_Dialogue)))
	print("Training ...")
	try:
		for epoch in range(num_epochs):
			train_cost = 0.0
			for _ in range(EPOCH_SIZE):
				ind = data_index[np.random.randint(0, len(data_index))]
				X, y, m = GetData(ind)
				#X, y, m = gen_data()
				train_cost = train_cost + train(X, y, m)
			ind = np.random.randint(0, len(Test_Dialogue) - 16 - N_BATCH)
			X, y, m = GetData2(ind)
			cost_val, predict_l = compute_cost(X, y, m)
			print("Epoch {} train cost = {}".format(epoch, train_cost / EPOCH_SIZE))
			print("Epoch {} validation cost = {}".format(epoch, cost_val))
			print(predict_l[:20])
			print(y[:20])
			if epoch % 10 == 1:
				#################################################################################
				c_matrix = numpy.zeros((9,9), dtype=int)
				data_index_t = [i for i in xrange(len(Test_Dialogue))]
				data_index_t = data_index_t[0:len(data_index_t):N_BATCH]
				data_index_t = data_index_t[:-1]
				print("Epoch {}".format(len(data_index_t)))			
				test_cost = 0.0
				test_acc = 0.0
				for ind in range(len(data_index_t)):
					X, y, m = GetData2(data_index_t[ind])
					cost_val, predict_l = compute_cost(X, y, m)
					test_cost = test_cost + cost_val
					test_acc = test_acc + numpy.sum(predict_l == y) * 1.0 / len(predict_l)
					for j in xrange(N_BATCH):
						c_matrix[y[j]][predict_l[j]] = c_matrix[y[j]][predict_l[j]] + 1
				print("Epoch {} test cost = {}".format(epoch, test_cost / len(data_index_t)))
				print("Epoch {} test acc = {}".format(epoch, test_acc / len(data_index_t)))
				print(c_matrix)	
				test_acc = test_acc / len(data_index_t)
				#################################################################################
				all_param_values = lasagne.layers.get_all_param_values(l_out)
				f_ = open('./' + str(N_HIDDEN) + '_Emotion_LSTM_Model_not_sorted.dump' + '_' + str(LEARNING_RATE) + '_' + str(epoch) + '_' + str(test_acc), 'wb')
				cPickle.dump(all_param_values, f_, -1)
				f_.close()
	except KeyboardInterrupt:
		pass

if __name__ == '__main__':
	main()
