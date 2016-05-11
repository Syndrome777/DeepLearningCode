#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import numpy
import random
import os
from DenseLayer3D import DenseLayer3D



class CNNLSTMModel(object):
	def __init__(self):
		print('CNNLSTMModel init...')
	
	def __buildModel__(self,
		trainData,
		validData,
		input_dim = 29,
		output_dim = 2,
		hidden_dim = 256,
		label_skip = 5,
		max_length = 300,
		batch_size = 256,
		learning_rate = 0.001,
		language_id = {},
		grad_clip = 10
		):
		######################
		self.ind = 0
		random.shuffle(trainData)
		self.trainData = trainData
		self.validData = validData
		self.INPUT_DIM = input_dim
		self.OUTPUT_DIM = output_dim
		self.HIDDEN_DIM = hidden_dim
		self.LABEL_SKIP = label_skip
		self.MAX_LENGTH = max_length
		self.BATCH_SIZE = batch_size
		self.LEARNING_RATE = learning_rate
		self.SAMPLE_NUM = len(trainData)
		self.LANGUAGE_ID = language_id
		self.GRAD_CLIP =grad_clip
		self.local_train_data = []
		self.local_valid_data = []
		
		# just for image
		self.image_set = []
		self.name_set = []
		self.label_set = []
		self.random_flag = 0
		self.train_num = 0
		self.train_random_set = []
		self.test_num = 0
		self.test_random_set = []
		self.Name2Id = {}
				  
		theano.config.exception_verbosity = "high"
		
		print("Building network ...")
		
		######################################## Input
		image_set = T.tensor4('image_set')
		label_set = T.ivector('target_output')
		layer_in = lasagne.layers.InputLayer((self.BATCH_SIZE, None, self.INPUT_DIM, self.INPUT_DIM), input_var = image_set, name = "layer_in")
		
		######################################## CNN 0
		layer_conv_0 = lasagne.layers.Conv2DLayer(
			layer_in, num_filters = 32, filter_size = (5, 5), stride = (2, 2), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "layer_conv_0")
		layer_bn_0 = lasagne.layers.BatchNormLayer(layer_conv_0, name = "layer_bn_0")
		layer_relu_0 = lasagne.layers.NonlinearityLayer(layer_bn_0, name = "layer_relu_0")
		
		######################################## CNN 1
		layer_conv_1 = lasagne.layers.Conv2DLayer(
			layer_relu_0, num_filters = 64, filter_size = (5, 5), stride = (2, 2), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "layer_conv_1")
		layer_bn_1 = lasagne.layers.BatchNormLayer(layer_conv_1, name = "layer_bn_1")
		layer_relu_1 = lasagne.layers.NonlinearityLayer(layer_bn_1, name = "layer_relu_1")
		
		######################################## CNN 2
		layer_conv_2 = lasagne.layers.Conv2DLayer(
			layer_relu_1, num_filters = 128, filter_size = (3, 3), stride = (1, 1), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "layer_conv_2")
		layer_max_2 = lasagne.layers.MaxPool2DLayer(
			layer_conv_2, pool_size = (2, 2), ignore_border = False, name = "layer_max_2"
			)
		layer_bn_2 = lasagne.layers.BatchNormLayer(layer_max_2, name = "layer_bn_2")
		layer_relu_2 = lasagne.layers.NonlinearityLayer(layer_bn_2, name = "layer_relu_2")
		
		######################################## CNN 3
		layer_conv_3 = lasagne.layers.Conv2DLayer(
			layer_relu_2, num_filters = 256, filter_size = (3, 3), stride = (1, 1), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "layer_conv_3")
		layer_max_3 = lasagne.layers.MaxPool2DLayer(
			layer_conv_3, pool_size = (2, 2), ignore_border = False, name = "layer_max_3"
			)
		layer_bn_3 = lasagne.layers.BatchNormLayer(layer_max_3, name = "layer_bn_3")
		layer_relu_3 = lasagne.layers.NonlinearityLayer(layer_bn_3, name = "layer_relu_3")
		
		######################################## CNN 4
		layer_conv_4 = lasagne.layers.Conv2DLayer(
			layer_relu_3, num_filters = 512, filter_size = (3, 3), stride = (1, 1), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "layer_conv_4")
		layer_bn_4 = lasagne.layers.BatchNormLayer(layer_conv_4, name = "layer_bn_4")
		layer_relu_4 = lasagne.layers.NonlinearityLayer(layer_bn_4, name = "layer_relu_4")
		layer_max_4 = lasagne.layers.Pool2DLayer(
			layer_relu_4, pool_size = (8, 8), ignore_border = False, mode = "average_inc_pad", name = "layer_max_4"
			)
		
		######################################## Softmax
		softmax = lasagne.nonlinearities.softmax
		self.layer_out = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(layer_max_4, p=.25), num_units=self.OUTPUT_DIM, nonlinearity=softmax, name = 'layer_out'
			)
		
		#############################################################################################################	
		network_output = lasagne.layers.get_output(self.layer_out)
		cost = lasagne.objectives.categorical_crossentropy(network_output, target_values)
		cost = cost.mean()
		predict_l = T.argmax(network_output, axis=1)
		self.all_params = lasagne.layers.get_all_params(self.layer_out, trainable=True)
		print("Computing updates ...")
		updates = lasagne.updates.adagrad(cost, self.all_params, self.LEARNING_RATE)
		print("Compiling functions ...")
		#####################################################
		self.train_step = theano.function([image_set, label_set],
											cost, updates=updates)
		#############################################################################################################						
		test_prediction = lasagne.layers.get_output(self.layer_out, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values),
						  dtype=theano.config.floatX)
		predict_test = T.argmax(test_prediction, axis=1)
		#####################################################
		self.compute_cost = theano.function(
								[image_set, label_set], [test_loss, predict_test, test_acc])
	
	@staticmethod
	def _getRandomSet():
		test_random_set = []
		train_random_set = []
		random_set = random.shuffle([i for i in xrange(len(TotalLabel))])
		for r in random_set:
			if r % 100 == 0:
				test_random_set.append(r)
			else:
				train_random_set.append(r)
		test_random_set = sorted(test_random_set)
		return test_random_set, train_random_set
	
	def _loadH5pyFile(self, file_name):
		# with h5py.File('total_face.h5', 'r') as hf:
		hf = h5py.File('total_face.h5', 'r')
		self.image_set = hf['TotalFace']
		self.name_set = hf['TotalLabel']
		self.random_flag = 1
		self.test_random_set, self.train_random_set = _getRandomSet()
		self.test_num = len(self.test_random_set)
		self.train_num = len(self.train_random_set) - self.test_num
		i = 0
		for name in self.name_set:
			if not self.Name2Id.has_key(name):
				self.Name2Id[name] = i
				i += 1
		self.category_num = i
		assert self.OUTPUT_DIM == self.category_num
			
			
	def _loadValidData(self, min_len = 300):
		for file_name in self.validData:
			#sample = self._loadDumpFile(file_name)
			sample = self._loadNpyFile(file_name)
			if sample[0].shape[0] >= min_len:
				self.local_valid_data.append([sample, sample[0].shape[0]])
		self.local_valid_data = sorted(self.local_valid_data, key=lambda x:x[1], reverse=1)
	
	
	def _getDataFromDatabase(self, index, valid_flag = 0):
		n_batch = self.BATCH_SIZE
		max_length = self.MAX_LENGTH
		input_dim = self.INPUT_DIM
		
		if len(self.train_random_set) < n_batch:
			self.test_random_set, self.train_random_set = _getRandomSet()
		
		local_train_data = []
		for i in xrange(n_batch):
			local_train_data.append(self.train_random_set.pop())
		local_train_data = sorted(local_train_data)
		input_ = self.image_set[local_train_data]
		output_ = self.label_set[local_train_data]
		output_ = [self.Name2Id[n] for n in output_]
		
		input_ =  numpy.asarray(input_, dtype='float32')
		output_ =  numpy.asarray(output_, dtype='int32')
		
		#print(input_.shape, output_.shape, mask.shape)
		return (input_.astype(numpy.float32), output_.astype(numpy.int32))
		
		
		
	def updateParam(self, all_param_values):
		lasagne.layers.set_all_param_values(self.l_out, all_param_values)
		return 0
		
	def getParam(self):
		model_param = lasagne.layers.get_all_param_values(self.l_out)
		return model_param
		
	def saveModel(self, test_acc, epoch = 0, iter = 0):
		all_param_values = self.getParam()
		f_ = open('./dump/' + str(self.HIDDEN_DIM) + '_Emotion_LSTM_Model_not_sorted.dump' + '_' + str(self.LEARNING_RATE) + '_' + str(epoch) + '_' + str(iter) + '_' + str(test_acc), 'wb')
		cPickle.dump(all_param_values, f_, -1)
		f_.close()
	
	def train(self, sync_iter_num):
		train_cost = 0.0
		for _ in range(sync_iter_num):
			X, y= self._getDataFromDatabase(self.ind, 0)
			#print(X)
			train_cost = train_cost + self.train_step(X, y)
		return train_cost
			
	def validation(self, epoch = 0):
		# if using noise
		#self.use_noise.set_value(0.)
		if len(self.local_valid_data) <= 0:
			self._loadValidData()
		c_matrix = numpy.zeros((7,7), dtype=int)
		c_matrix_1 = numpy.zeros((7,7), dtype=int)
		c_matrix_2 = numpy.zeros((7,7), dtype=int)
		#data_index_t = [i for i in xrange(10000)]
		data_index_t = [i for i in xrange(len(self.local_valid_data))]
		data_index_t = data_index_t[0:len(data_index_t):self.BATCH_SIZE]
		data_index_t = data_index_t[:-1]
		print("Epoch {}".format(len(data_index_t)))			
		test_cost = 0.0
		test_acc = 0.0
		test_acc_1 = 0.0
		test_acc_2 = 0.0
		for ind_ in range(len(data_index_t)):
			X, y, m = self._getDataFromDatabase(data_index_t[ind_], 1)
			cost_val, predict_l = self.compute_cost(X, y, m)
			#######
			repeat = y.shape[0] / self.BATCH_SIZE - 1        # need -1
			if repeat == 0:
				repeat = 1
			y_1 = y[::repeat][:self.BATCH_SIZE]
			predict_l_1 = predict_l[::repeat][:self.BATCH_SIZE]
			y_2 = y[1::repeat][:self.BATCH_SIZE]
			predict_l_2 = predict_l[1::repeat][:self.BATCH_SIZE]
			if y.shape[0] <= 256:
				y_2 = y_1
				predict_l_2 = predict_l_1
			#print(y.shape, predict_l.shape)
			#print(y_1.shape, predict_l_1.shape)
			#print(y_2.shape, predict_l_2.shape)
			#######
			y = y[-self.BATCH_SIZE:]
			predict_l = predict_l[-self.BATCH_SIZE:]
			
			test_cost = test_cost + cost_val
			test_acc = test_acc + numpy.sum(predict_l == y) * 1.0 / len(predict_l)
			test_acc_1 = test_acc_1 + numpy.sum(predict_l_1 == y_1) * 1.0 / len(predict_l_1)
			test_acc_2 = test_acc_2 + numpy.sum(predict_l_2 == y_2) * 1.0 / len(predict_l_2)
			for j in xrange(self.BATCH_SIZE):
				c_matrix[y[j]][predict_l[j]] = c_matrix[y[j]][predict_l[j]] + 1
			#for j in xrange(y_1.shape[0]):
				c_matrix_1[y_1[j]][predict_l_1[j]] = c_matrix_1[y_1[j]][predict_l_1[j]] + 1
			#for j in xrange(y_2.shape[0]):
				c_matrix_2[y_2[j]][predict_l_2[j]] = c_matrix_2[y_2[j]][predict_l_2[j]] + 1
		print("Epoch {} test cost = {}".format(epoch, test_cost / len(data_index_t)))
		print("Epoch {} test acc least 0.5s = {}".format(epoch, test_acc_1 / len(data_index_t)))
		print("Epoch {} test acc least 1s = {}".format(epoch, test_acc_2 / len(data_index_t)))
		print("Epoch {} test acc = {}".format(epoch, test_acc / len(data_index_t)))
		print(c_matrix_1)
		print("#####################################################")
		print(c_matrix_2)
		print("#####################################################")
		print(c_matrix)
		print("#####################################################")
					
		return test_acc / len(data_index_t)
		



def main():
	if not os.path.exists('./dump'):
		os.makedirs('./dump')
	#f_ = open('./dump/256_Emotion_LSTM_Model_not_sorted.dump_0.001_2_6500_0.887547348485', 'rb')
	#all_param_values = cPickle.load(f_)
	#f_.close()
	
	CLModel = CNNLSTMModel()
	CLModel.__buildModel__(trainData, validData, language_id = LanguageId, input_dim = 96, output_dim = 2623, batch_size = 256, hidden_dim = 256)
	print("Trainable Model Parameters")
	print("-"*40)
	for param in CLModel.all_params:
		print(param, param.get_value().shape)
	print("-"*40)
	all_params = CLModel.getParam()
	for p in all_params:
		print(p.shape)
	while True:
		train_cost = CLModel.train(10)
		print(train_cost / 10)
		#CLModel.updateParam(all_param_values)
		
		#test_acc = CLModel.validation()
		#CLModel.saveModel(test_acc)
	

if __name__ == '__main__':
	main()
	
	
	
	
	
