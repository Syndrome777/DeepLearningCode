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
				  
		theano.config.exception_verbosity = "high"
		
		print("Building network ...")
		######################################## Input
		x_sym = T.tensor3()
		target_values = T.ivector('target_output')
		xmask_sym = T.matrix()
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
		predict_l = T.argmax(network_output, axis=1)
		self.all_params = lasagne.layers.get_all_params(self.l_out, trainable=True)
		print("Computing updates ...")
		updates = lasagne.updates.adagrad(cost, self.all_params, self.LEARNING_RATE)
		print("Compiling functions ...")
		#####################################################
		self.train_step = theano.function([x_sym, target_values, xmask_sym],
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
	
	
	def _loadDumpFile(self, file_name):
		data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/GetSpe/data/"
		with open(data_root + file_name, 'rb') as f_:
			data = cPickle.load(f_)
		#[d, label, user_name, file_name] = data
		return data
		
	def _loadNpyFile(self, file_name):
		ID2Language = ['EN', 'LM']
		data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/Get80DimsFea/data"	
		#data_root = "\\\\speech-tesla05\\d$\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\Get80DimsFea\\data"
		fea, slice_info, label = numpy.load(data_root + file_name)
		new_fea = []
		for sli in slice_info:
			[start_i, end_i] = sli
			#if end_i - start_i <= 10:
				#continue
				#print(file_name, start_i, end_i)
			if start_i > 1:
				start_i -= 1
			new_fea.append(fea[start_i : end_i])
		new_fea = numpy.concatenate(new_fea, axis=0)
		data = [new_fea, ID2Language[label]]
		#data = numpy.asarray([fea, slice_info, label])
		return data
	
	def _loadValidData(self, min_len = 300):
		for file_name in self.validData:
			#sample = self._loadDumpFile(file_name)
			sample = self._loadNpyFile(file_name)
			if sample[0].shape[0] >= min_len:
				self.local_valid_data.append([sample, sample[0].shape[0]])
		self.local_valid_data = sorted(self.local_valid_data, key=lambda x:x[1], reverse=1)
	
	def _getDataFromDatabase(self, index, valid_flag = 0):
		n_batch=self.BATCH_SIZE
		max_length=self.MAX_LENGTH
		input_dim = self.INPUT_DIM
		if len(self.local_train_data) < n_batch:
			for i in xrange(n_batch * 10):
			#for i in xrange(n_batch * 10):
				#if valid_flag == 0:
				#sample = self._loadDumpFile(self.trainData[(index + i) % self.SAMPLE_NUM])
				sample = self._loadNpyFile(self.trainData[(index + i) % self.SAMPLE_NUM])
				self.local_train_data.append([sample, sample[0].shape[0]])
			self.local_train_data = sorted(self.local_train_data, key=lambda x:x[1], reverse=0)
				#else:
				#	self.local_valid_data.append(self._loadDumpFile(self.validData[(index + i) % len(self.validData)]))
				
		local_data = []
		for i in xrange(n_batch):
			if valid_flag == 0:
				dat = self.local_train_data.pop()
				local_data.append(dat[0])
				if i % 8 == 0:
					self.local_train_data.append(dat)
			else:
				#local_data.append(self._loadDumpFile(self.validData[(index + i) % len(self.validData)]))
				local_data.append(self.local_valid_data[(index + i) % len(self.local_valid_data)][0])
		data_len = [local_data[i][0].shape[0] for i in xrange(n_batch)]
		'''
		if valid_flag == 0:
			data_len = [self.trainData[(index + i) % self.SAMPLE_NUM][0].shape[0] for i in xrange(n_batch)]
		else:
			data_len = [self.validData[(index + i) % len(self.validData)][0].shape[0] for i in xrange(n_batch)]
		'''
		
		max_data_len = max(data_len)	
		#print(max_data_len)
		if max_data_len > max_length:
			max_data_len = max_length
		#print(max_data_len)
		
		max_length = max_data_len
		label_repeat = (max_length - 1) / self.LABEL_SKIP / 8 + 1
		mask = np.zeros((n_batch, max_length))
		
		input_ = numpy.zeros((n_batch, max_length, input_dim), dtype=numpy.float32)
		output_ = numpy.zeros((n_batch * label_repeat), dtype=numpy.int32)

		for i in range(n_batch):
			'''
			if valid_flag == 0:
				ind = (index + i) % self.SAMPLE_NUM
				feat = self.trainData[ind][0]
				labe = self.trainData[ind][1]
			else:
				ind = (index + i) % len(self.validData)
				feat = self.validData[ind][0]
				labe = self.validData[ind][1]
			'''
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
		return (input_.astype(numpy.float32), output_.astype(numpy.int32),
				mask.astype(numpy.float32))
		
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
			X, y, m = self._getDataFromDatabase(self.ind, 0)
			#print(X)
			train_cost = train_cost + self.train_step(X, y, m)
			if self.ind + self.BATCH_SIZE >= self.SAMPLE_NUM - 10:
				random.shuffle(self.trainData)
				self.ind = (self.ind + self.BATCH_SIZE) % self.SAMPLE_NUM
			else:
				self.ind += self.BATCH_SIZE
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
		
	


def main_():
	###########  Word2Vec Matrix
	f_ = open('../data/BR&RU_SliceAllTrainingData.dump', 'rb')
	AllFeat_ = cPickle.load(f_) # 29
	f_.close()

	print(len(AllFeat_))
	random.seed(10)
	random.shuffle(AllFeat_)

	AllFeat_valid_ = AllFeat_[-10000:]
	AllFeat_ = AllFeat_[:-10000]
	
	trainData = AllFeat_
	validData = AllFeat_valid_
	LanguageId = {}
	LanguageId['BR'] = 0
	LanguageId['RU'] = 1
	
	if not os.path.exsits('./dump'):
		os.makedirs('./dump')
	#f_ = open('./dump/256_Emotion_LSTM_Model_not_sorted.dump_0.001_0_0.854767628205', 'rb')
	#all_param_values = cPickle.load(f_)
	#f_.close()
	
	CLModel = CNNLSTMModel()
	CLModel.__buildModel__(trainData, validData, language_id = LanguageId)
	train_cost = CLModel.train(20)
	print(train_cost / 20)
	#CLModel.updateParam(all_param_values)
	
	test_acc = CLModel.validation()
	CLModel.saveModel(test_acc)



def main():
	trainData = []
	validData = []
	with open('../data/LFB80/train_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			trainData.append(line)
			line = f_.readline().strip('\r\n')
	with open('../data/LFB80/test_list.txt', 'rb') as f_:
		line = f_.readline().strip('\r\n')
		while line:
			validData.append(line)
			line = f_.readline().strip('\r\n')
	###########  Word2Vec Matrix

	print(len(trainData))
	print(len(validData))
	random.seed(10)
	####
	trainData += validData
	random.shuffle(trainData)
	
	validData = trainData[:41000]
	trainData = trainData[41000:]

	LanguageId = {}
	LanguageId['EN'] = 0
	LanguageId['LM'] = 1
	#LanguageId[0] = 0
	#LanguageId[1] = 1
	
	if not os.path.exists('./dump'):
		os.makedirs('./dump')
	f_ = open('./dump/256_Emotion_LSTM_Model_not_sorted.dump_0.001_2_6500_0.887547348485', 'rb')
	all_param_values = cPickle.load(f_)
	f_.close()
	
	CLModel = CNNLSTMModel()
	CLModel.__buildModel__(trainData, validData, language_id = LanguageId, input_dim = 88, batch_size = 256, hidden_dim = 256)
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
		CLModel.updateParam(all_param_values)
		
		test_acc = CLModel.validation()
		CLModel.saveModel(test_acc)
	

if __name__ == '__main__':
	main()
	
	
	
	
	
