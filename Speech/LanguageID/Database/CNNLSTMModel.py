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
from collections import OrderedDict

from Database import Database
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
		
		##### Optimization
		self.velocitys = []
				  
		theano.config.exception_verbosity = "high"
		
		print("Building network ...")
		######################################## Input
		x_sym = T.tensor3()
		target_values = T.ivector('target_output')
		xmask_sym = T.matrix()
		lr_sclce = T.scalar()
		######################################## Input
		l_in = lasagne.layers.InputLayer((self.BATCH_SIZE, None, self.INPUT_DIM), input_var = x_sym, name = "l_in")
		######################################## CNN
		l_transpose = lasagne.layers.DimshuffleLayer(l_in, (0, 'x', 1, 2), name = "l_transpose")
		l_conv0 = lasagne.layers.Conv2DLayer(
			l_transpose, num_filters = 32, filter_size = (5, 7), stride = (2, 3), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "l_conv0")
		l_bn0 = lasagne.layers.BatchNormLayer(l_conv0, name = "l_bn0")
		l_relu_0 = lasagne.layers.NonlinearityLayer(l_bn0, name = "l_relu_0")
		l_conv1 = lasagne.layers.Conv2DLayer(
			l_relu_0, num_filters = 64, filter_size = (5, 5), stride = (2, 2), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "l_conv1")
		l_max = lasagne.layers.MaxPool2DLayer(
			l_conv1, pool_size = (2, 2), ignore_border = False, name = "l_max"
			)
		#l_bn1 = lasagne.layers.BatchNormLayer(l_max, axes = (0, 2), name = "l_bn1")
		l_bn1 = lasagne.layers.BatchNormLayer(l_max, name = "l_bn1")
		l_relu_1 = lasagne.layers.NonlinearityLayer(l_bn1, name = "l_relu_1")
		l_conv2 = lasagne.layers.Conv2DLayer(
			l_relu_1, num_filters = 128, filter_size = (3, 3), stride = (1, 1), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "l_conv2")
		# l_conv1 = (mini_batch, num_filters, length / 2, input_dim)
		#l_bn2 = lasagne.layers.BatchNormLayer(l_conv2, axes = (0, 2), name = "l_bn2")
		l_bn2 = lasagne.layers.BatchNormLayer(l_conv2, name = "l_bn2")
		l_relu_2 = lasagne.layers.NonlinearityLayer(l_bn2, name = "l_relu_2")
		'''
		# l_max = (mini_batch, num_filters, length / 4, input_dim)
		l_conv3 = lasagne.layers.Conv2DLayer(
			l_relu_2, num_filters = 256, filter_size = (3, 3), stride = (1, 1), pad = 'same',
			W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
			name = "l_conv3")
		#l_bn3 = lasagne.layers.BatchNormLayer(l_conv3, axes = (0, 2), name = "l_bn3")
		l_bn3 = lasagne.layers.BatchNormLayer(l_conv3, name = "l_bn3")
		l_relu_3 = lasagne.layers.NonlinearityLayer(l_bn3, name = "l_relu_3")
		# l_conv2 = (mini_batch, num_filters, length / 8, input_dim)
		#l_conv4 = lasagne.layers.Conv2DLayer(
		#	l_conv3, num_filters = 128, filter_size = (1, 1), stride = (1, 1), pad = 'same',
		#	W = lasagne.init.HeUniform(), #nonlinearity=lasagne.nonlinearities.linear, 
		#	name = "l_conv4")
		'''
		######################################## DNN
		l_transpose_reverse = lasagne.layers.DimshuffleLayer(l_relu_2, (0, 2, 1 ,3), name = "l_transpose_reverse")
		l_reshape = lasagne.layers.ReshapeLayer(l_transpose_reverse, ([0], [1], -1), name = "l_reshape")
		l_new_input = DenseLayer3D(l_reshape, self.INPUT_DIM * 4, W = lasagne.init.HeUniform(), name = "l_new_input")		
		#l_emb = lasagne.layers.EmbeddingLayer(l_in, WORD_NUM, INPUT_DIM, 
		#								  #W=Word2Vec_,
		#								  name='Embedding')
		#l_emb.params[l_emb.W].remove('trainable')
		l_mask = lasagne.layers.InputLayer(shape=(self.BATCH_SIZE, None), input_var = xmask_sym)
		#l_mask_pool = lasagne.layers.SliceLayer(l_mask, indices=slice(0, None, 8), axis=-1).output_shape
		l_mask_transpose = lasagne.layers.DimshuffleLayer(l_mask, (0, 'x', 1), name = "l_mask_transpose")
		l_mask_pool = lasagne.layers.MaxPool1DLayer(l_mask_transpose, ignore_border = False, pool_size = (8))
		l_mask_transpose_reverse = lasagne.layers.DimshuffleLayer(l_mask_pool, (0, 2), name = "l_mask_transpose_reverse")
		######################################## LSTM
		l_forward = lasagne.layers.LSTMLayer(
			l_new_input, self.HIDDEN_DIM, mask_input=l_mask_transpose_reverse, grad_clipping=self.GRAD_CLIP,
			cell_init=lasagne.init.HeUniform(),
			hid_init=lasagne.init.HeUniform(),
			nonlinearity=lasagne.nonlinearities.tanh, only_return_final=False, name = 'l_forward')
		#l_backward = lasagne.layers.LSTMLayer(
		#	l_in, self.HIDDEN_DIM, mask_input=l_mask, grad_clipping=GRAD_CLIP,
		#	cell_init=lasagne.init.HeUniform(),
		#	hid_init=lasagne.init.HeUniform(),
		#	nonlinearity=lasagne.nonlinearities.tanh,
		#	only_return_final=True, backwards=True, name = 'l_backward')
		#l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
		
		######################################## Loss
		l_sl_ = lasagne.layers.SliceLayer(l_forward, indices=slice(0, None, self.LABEL_SKIP), axis=1)
		l_sl = lasagne.layers.SliceLayer(l_sl_, indices=slice(1, None, 1), axis=1)
		l_sl_re = lasagne.layers.ReshapeLayer(l_sl, (-1, [2]))

		l_sl_end = lasagne.layers.SliceLayer(l_forward, indices=-1, axis=1)
		l_concat = lasagne.layers.ConcatLayer([l_sl_re, l_sl_end], axis = 0)
		
		softmax = lasagne.nonlinearities.softmax
		self.l_out = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(l_concat, p=.25), num_units=self.OUTPUT_DIM, nonlinearity=softmax, name = 'l_out')
		
		#############################################################################################################	
		network_output = lasagne.layers.get_output(self.l_out)
		cost = lasagne.objectives.categorical_crossentropy(network_output, target_values)
		cost = cost.mean()
		self.all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		#l2_reg = lasagne.regularization.regularize_network_params(self.l_out, lasagne.regularization.l2)
		#cost = cost + l2_reg * 2e-5
		predict_l = T.argmax(network_output, axis=1)
		print("Computing updates ...")
		#updates = lasagne.updates.adagrad(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		#updates = lasagne.updates.adadelta(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		#updates = lasagne.updates.rmsprop(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		#updates = lasagne.updates.nesterov_momentum(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		#updates = self.apply_momentum(self.rmsprop(cost, self.all_params, self.LEARNING_RATE / lr_sclce), params = self.all_params, momentum = 0.9)
		#updates = self.rmsprop(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		updates = self.adam(cost, self.all_params, self.LEARNING_RATE / lr_sclce)
		#updates = self.apply_nesterov_momentum(lasagne.updates.sgd(cost, self.all_params, self.LEARNING_RATE / lr_sclce), params = self.all_params)
		print("Compiling functions ...")
		#####################################################
		self.train_step = theano.function([x_sym, target_values, xmask_sym, lr_sclce],
										cost, updates=updates)
		#############################################################################################################						
		test_prediction = lasagne.layers.get_output(self.l_out, deterministic=True)
		test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_values)
		test_loss = test_loss.mean()
		# As a bonus, also create an expression for the classification accuracy:
		test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_values),
						  dtype=theano.config.floatX)
		predict_test = T.argmax(test_prediction, axis=1)
		#####################################################
		self.compute_cost = theano.function(
								[x_sym, target_values, xmask_sym], [test_loss, predict_test])
	
	
	def apply_momentum(self, updates, params=None, momentum=0.9):
		if params is None:
			params = updates.keys()
		updates = OrderedDict(updates)
		print('apply_momentum : ')
		for param in params:
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
									 broadcastable=param.broadcastable)
			print(velocity.get_value(borrow=True).shape)
			self.velocitys.append(velocity)
			#print(param)
			x = momentum * velocity + updates[param]
			updates[velocity] = x - param
			updates[param] = x
		return updates
		
	def apply_nesterov_momentum(self, updates, params=None, momentum=0.9):
		if params is None:
			params = updates.keys()
		updates = OrderedDict(updates)

		for param in params:
			value = param.get_value(borrow=True)
			velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype),
									 broadcastable=param.broadcastable)
			self.velocitys.append(velocity)
			x = momentum * velocity + updates[param] - param
			updates[velocity] = x
			updates[param] = momentum * x + updates[param]

		return updates
		
	def rmsprop(self, loss_or_grads, params, learning_rate=1.0, rho=0.9, epsilon=1e-6):
		grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
		updates = OrderedDict()
		# Using theano constant to prevent upcasting of float32
		one = T.constant(1)
		print('rmsprop : ')
		for param, grad in zip(params, grads):
			value = param.get_value(borrow=True)
			accu = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								 broadcastable=param.broadcastable)
			print(accu.get_value(borrow=True).shape)
			self.velocitys.append(accu)
			accu_new = rho * accu + (one - rho) * grad ** 2
			updates[accu] = accu_new
			updates[param] = param - (learning_rate * grad / T.sqrt(accu_new + epsilon))
		return updates
		
	def adam(self, loss_or_grads, params, learning_rate=0.001, beta1=0.9,
			 beta2=0.999, epsilon=1e-8):
		all_grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
		#t_prev = theano.shared(utils.floatX(0.))
		t_prev = theano.shared(numpy.asarray(0, dtype="float32"))
		updates = OrderedDict()

		# Using theano constant to prevent upcasting of float32
		one = T.constant(1)

		t = t_prev + 1
		a_t = learning_rate*T.sqrt(one-beta2**t)/(one-beta1**t)

		for param, g_t in zip(params, all_grads):
			value = param.get_value(borrow=True)
			m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								   broadcastable=param.broadcastable)
			v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
								   broadcastable=param.broadcastable)

			m_t = beta1*m_prev + (one-beta1)*g_t
			v_t = beta2*v_prev + (one-beta2)*g_t**2
			self.velocitys.append(m_prev)
			self.velocitys.append(v_prev)
			
			step = a_t*m_t/(T.sqrt(v_t) + epsilon)

			updates[m_prev] = m_t
			updates[v_prev] = v_t
			updates[param] = param - step

		updates[t_prev] = t
		return updates
	
	def _getDataFromDatabase_signal(self, index, valid_flag = 0):
		n_batch=self.BATCH_SIZE         
		max_length=self.MAX_LENGTH
		input_dim = self.INPUT_DIM
		overlap = 160
		
		if len(self.database.local_train_data) < n_batch:
			self.database.loadBatchData(index = index, samp_num = n_batch * 10, min_len = 0, sort_data = True)
				
		local_data = []
		for i in xrange(n_batch):
			if valid_flag == 0:
				dat = self.database.local_train_data.pop()
				local_data.append(dat[0])
				if i % 7 == 0:
					self.database.local_train_data.append(dat)
			else:
				local_data.append(self.database.local_valid_data[(index + i) % len(self.database.local_valid_data)][0])
		data_len = [local_data[i][0].shape[0] for i in xrange(n_batch)]

		max_data_len = max(data_len)	
		#print(max_data_len)
		if max_data_len > max_length:
			max_data_len = max_length
		#print(max_data_len)
		
		max_length = max_data_len
		#label_repeat = (max_length - 1) / self.LABEL_SKIP / 8 + 1
		#new_max_length = max_length
		new_max_length = int((max_length - input_dim) / overlap - 1e-5) + 1
		label_repeat = (new_max_length - 1) / self.LABEL_SKIP + 1
		mask = np.zeros((n_batch, new_max_length))
		
		input_ = numpy.zeros((n_batch, new_max_length, input_dim), dtype=numpy.float32)
		output_ = numpy.zeros((n_batch * label_repeat), dtype=numpy.int32)

		for i in range(n_batch):
			feat = local_data[i][0]
			labe = local_data[i][1]
			#print(feat, labe)
			length = feat.shape[0]
			if length > max_length:
				length = max_length
			#input_[i, :length, :] += feat[:length]
			
			new_length = int((length - input_dim) / overlap - 1e-5) + 1
			for j in xrange(new_length):
				start_i = j*overlap
				input_[i, j, :] += feat[start_i:start_i+input_dim]
			output_[i * (label_repeat - 1) : (i + 1) * (label_repeat - 1)] = self.LANGUAGE_ID[labe]
			output_[n_batch * (label_repeat - 1) + i] = self.LANGUAGE_ID[labe]
			# Make the mask for this sample 1 within the range of length
			mask[i, :new_length] += 1
			
		#print(input_.shape, output_.shape, mask.shape)
		return (input_.astype(numpy.float32), output_.astype(numpy.int32), mask.astype(numpy.float32))
		
	def _getDataFromDatabase(self, index, valid_flag = 0):
		n_batch=self.BATCH_SIZE         
		max_length=self.MAX_LENGTH
		input_dim = self.INPUT_DIM
		overlap = 160
		
		if len(self.database.local_train_data) < n_batch:
			self.database.loadBatchData(index = index, samp_num = n_batch * 10, min_len = 0, sort_data = True)
				
		local_data = []
		for i in xrange(n_batch):
			if valid_flag == 0:
				dat = self.database.local_train_data.pop()
				local_data.append(dat[0])
				if i % 7 == 0:
					self.database.local_train_data.append(dat)
			else:
				local_data.append(self.database.local_valid_data[(index + i) % len(self.database.local_valid_data)][0])
		data_len = [local_data[i][0].shape[0] for i in xrange(n_batch)]

		max_data_len = max(data_len)	
		#print(max_data_len)
		if max_data_len > max_length:
			max_data_len = max_length
		#print(max_data_len)
		#print(local_data[0])
		#print(len(local_data))
		
		max_length = max_data_len
		label_repeat = (max_length - 1) / self.LABEL_SKIP / 8 + 1
		mask = np.zeros((n_batch, max_length))
		
		input_ = numpy.zeros((n_batch, max_length, input_dim), dtype=numpy.float32)
		output_ = numpy.zeros((n_batch * label_repeat), dtype=numpy.int32)

		for i in range(n_batch):
			feat = local_data[i][0]
			labe = local_data[i][1]
			#print(feat, labe)
			length = feat.shape[0]
			if length > max_length:
				length = max_length
			#start_feat_ind = feat.shape[0] / 2 - length / 2
			input_[i, :length, :] += feat[:length]
			output_[i * (label_repeat - 1) : (i + 1) * (label_repeat - 1)] = self.LANGUAGE_ID[labe]
			output_[n_batch * (label_repeat - 1) + i] = self.LANGUAGE_ID[labe]
			# Make the mask for this sample 1 within the range of length
			mask[i, :length] += 1
			
		#print(input_.shape, output_.shape, mask.shape)
		return (input_.astype(numpy.float32), output_.astype(numpy.int32), mask.astype(numpy.float32))
	
	
	def updateParam(self, all_param_values, withoutMoment = False):
		if withoutMoment == True:
			lasagne.layers.set_all_param_values(self.l_out, all_param_values)
		else:
			velocity_num = len(self.velocitys)
			param_num = len(all_param_values) - velocity_num
			#print(param_num, velocity_num)
			lasagne.layers.set_all_param_values(self.l_out, all_param_values[:param_num])
			i = 0
			for v in self.velocitys:
				v.set_value(all_param_values[param_num + i])
				i += 1
		return 0
		
	def getParam(self):
		model_param = lasagne.layers.get_all_param_values(self.l_out)
		for v in self.velocitys:
			model_param.append(v.get_value(borrow=True))
		return model_param
		
	def saveModel(self, test_acc, epoch = 0, iter = 0):
		all_param_values = self.getParam()
		f_ = open('./dump/' + str(self.HIDDEN_DIM) + '_Emotion_LSTM_Model_not_sorted.dump' + '_' + str(self.LEARNING_RATE) + '_' + str(epoch) + '_' + str(iter) + '_' + str(test_acc), 'wb')
		cPickle.dump(all_param_values, f_, -1)
		f_.close()
		
	def loadDatabase(self, data_root, loadDataSource, loadFromDisk = True):
		self.database = Database()
		self.database.init(trainData = self.trainData, validData = self.validData, data_root = data_root, loadFromDisk = loadFromDisk, loadDataSource = loadDataSource)
		#self.database.loadAllData(valid_min_len = self.MAX_LENGTH, train_min_len = 0, rand_data = True)
		self.database.resetTrainingData()
	
	def train(self, sync_iter_num, lr_sclce = 1):
		train_cost = 0.0
		for _ in range(sync_iter_num):
			X, y, m = self._getDataFromDatabase(self.ind, 0)
			#print(X)
			train_cost = train_cost + self.train_step(X, y, m, lr_sclce)
			if self.ind + self.BATCH_SIZE >= self.SAMPLE_NUM - 10:
				#random.shuffle(self.trainData)
				self.database.resetTrainingData()
				self.ind = (self.ind + self.BATCH_SIZE) % self.SAMPLE_NUM
			else:
				self.ind += self.BATCH_SIZE
		return train_cost
			
	def validation(self, epoch = 0):
		# if using noise
		#self.use_noise.set_value(0.)
		if len(self.database.local_valid_data) <= 0:
			self._loadValidData()
		c_matrix = numpy.zeros((7,7), dtype=int)
		c_matrix_1 = numpy.zeros((7,7), dtype=int)
		c_matrix_2 = numpy.zeros((7,7), dtype=int)
		#data_index_t = [i for i in xrange(10000)]
		data_index_t = [i for i in xrange(len(self.database.local_valid_data))]
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
	'''
	#avialable_sample_ = os.listdir('D:\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\GetSpe\\data_sig\\EN')
	avialable_sample_ = os.listdir('\\\\speech-tesla05\\d$\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\GetSpe\\data_sig\\EN')
	avialable_sample = {}
	for sam in avialable_sample_:
		avialable_sample['/EN/' + sam] = 1
	trainData = []
	validData = []
	with open('../data/LFB80/SpeakerIndependentWithNoise/train_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			if avialable_sample.has_key(line):
				trainData.append(line)
			line = f_.readline().strip('\r\n')
	with open('../data/LFB80/SpeakerIndependentWithNoise/test_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			if avialable_sample.has_key(line):
				validData.append(line)
			line = f_.readline().strip('\r\n')
	###########  Word2Vec Matrix
	'''
	trainData = []
	validData = []
	with open('../data/LFB80/SpeakerIndependentWithNoise/train_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			trainData.append(line)
			line = f_.readline().strip('\r\n')
	with open('../data/LFB80/SpeakerIndependentWithNoise/test_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			validData.append(line)
			line = f_.readline().strip('\r\n')
	#trainData = trainData[:50000]
	#validData = validData[:10000]
	print(len(trainData))
	print(len(validData))

	LanguageId = {}
	LanguageId['EN'] = 0
	LanguageId['LM'] = 1
	
	if not os.path.exists('./dump'):
		os.makedirs('./dump')
	f_ = open('./dump/256_Emotion_LSTM_Model_not_sorted.dump_0.001_0_20_0.874486019737', 'rb')
	all_param_values = cPickle.load(f_)
	f_.close()
	
	CLModel = CNNLSTMModel()
	CLModel.__buildModel__(trainData, validData, language_id = LanguageId, learning_rate = 0.002, input_dim = 88, batch_size = 512, 
								hidden_dim = 256, max_length = 300, label_skip = 1)
	data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/GetSpe/data_sig"
	data_root = "\\\\speech-tesla05\\d$\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\GetSpe\\data_sig"
	data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/Get80DimsFea/data"
	print('data loading...')
	CLModel.loadDatabase(data_root, loadDataSource = 'LFB', loadFromDisk = False)
	CLModel.database.loadAllData(valid_min_len = 300, train_min_len = 0, rand_data = True)
	
	print("Trainable Model Parameters")
	print("-"*40)
	for param in CLModel.all_params:
		print(param, param.get_value().shape)
	print("-"*40)
	all_params = CLModel.getParam()
	for p in all_params:
		print(p.shape)
	i = 0
	iter = 0
	CLModel.updateParam(all_param_values)
	while True:
		if i == -1:
			train_cost = CLModel.train(200, 10)
		else:
			train_cost = CLModel.train(500, 1 + 0.1*i)
		print(train_cost / 500)
		#CLModel.updateParam(all_param_values)
		if i < 1000:
			i += 1
		if i == -1:
			CLModel.database._loadTotalValidData(300)
		iter += 1
		test_acc = CLModel.validation()
		CLModel.saveModel(test_acc, epoch = 0, iter = iter)
	

if __name__ == '__main__':
	main()
	
	
	
	
	
