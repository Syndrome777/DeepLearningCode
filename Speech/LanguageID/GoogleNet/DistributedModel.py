#coding=utf-8
import multiprocessing, time
import numpy
import cPickle
import random
import argparse
import types
import os
import time

from CNNLSTMModel import CNNLSTMModel as YourModel


SEED = 123
numpy.random.seed(SEED)


def SaveModel(model_params, iter_index, costs_all_mean):
	f_ = open('./MiniBatch_' + str(iter_index) + '.dump' + str(costs_all_mean), 'wb')
	cPickle.dump(model_params, f_, -1)
	f_.close()

def LoadModel(model_params, iter_index, costs_all_mean = 0):
	#f_ = open('./MiniBatch_' + str(iter_index) + '.dump' + str(costs_all_mean), 'rb')
	f_ = open(iter_index, 'rb')
	new_model_params = cPickle.load(f_)
	f_.close()
	return new_model_params

def InitModelParameter(config_params):
	import theano.sandbox.cuda
	theano.sandbox.cuda.use('gpu0')
	import theano
	InitedModel = YourModel()
	InitedModel.__buildModel__( config_params['trainData'],
								config_params['validData'],
								config_params['input_dim'],
								config_params['output_dim'],
								config_params['hidden_dim'],
								config_params['label_skip'],
								config_params['max_length'],
								config_params['batch_size'],
								config_params['learning_rate'],
								config_params['language_id'],
								config_params['grad_clip']
								)
	init_param = InitedModel.getParam()
	print("Trainable Model Parameters")
	print("-"*40)
	for param in InitedModel.all_params:
		print(param, param.get_value().shape)
	print("-"*40)
	return init_param
	
	


def Worker(qQueue, cQueue, thread_num, gpu_index, sync_iter_num, asgd_pattern, config_params):
	import theano.sandbox.cuda
	theano.sandbox.cuda.use(gpu_index)
	import theano
	import theano.tensor as T
	from theano.tensor.shared_randomstreams import RandomStreams
	from collections import OrderedDict
	import lasagne
	
	f_ = open('Thread_' + str(thread_num) + '_LOG.txt', 'wb')
	
	LocalModel = YourModel()
	LocalModel.__buildModel__(  config_params['trainData'],
								config_params['validData'],
								config_params['input_dim'],
								config_params['output_dim'],
								config_params['hidden_dim'],
								config_params['label_skip'],
								config_params['max_length'],
								config_params['batch_size'],
								config_params['learning_rate'],
								config_params['language_id'],
								config_params['grad_clip']
								)
	total_sample_num = len(config_params['trainData'])
	batch_size = config_params['batch_size']
	config_params = []
	model_params = cQueue.get()
	LocalModel.updateParam(model_params)	
	epoch = 0
	iter = 0
	bad_flag = 0	
	while True:	
		loss = 0
		now_time = time.time()
		loss = LocalModel.train(sync_iter_num)
		if bad_flag == 0 and iter % 10 == 0:
			print thread_num, 'training time : ',  time.time() - now_time
		now_time = time.time()
		loss = loss / sync_iter_num 
		local_model_params = LocalModel.getParam()
			
		if asgd_pattern == 'momentum' or asgd_pattern == 'asgd':
			grad_params = [l_p - m_p for l_p, m_p in zip(local_model_params, model_params)]
			qQueue.put([grad_params, loss, thread_num])
		if asgd_pattern == 'easgd':
			qQueue.put([local_model_params, loss, thread_num])
		# write log
		log = 'Epoch : ' + str(epoch) + ' Iter : ' + str(iter) + ' Loss : ' + str(loss) + '\r\n' + '###########\r\n'
		f_.write(log)
		model_params = cQueue.get()
		LocalModel.updateParam(model_params)
		
		if bad_flag == 0 and iter % 100 == 0:
			print thread_num, 'communication time : ',  time.time() - now_time
			print thread_num, epoch, iter, 'training loss : ', loss
			
		if iter % 100 == 0 and thread_num == 0:
			test_acc =  LocalModel.validation()
			f_.write('############################################\r\n\t test_acc : ' + str(test_acc) + '\r\n')
			print "############################################"
			print 'valid_acc : ', test_acc
			print "############################################"
			LocalModel.saveModel(test_acc, epoch = epoch, iter = iter)
		
		iter += 1
		bad_flag = 0
		epoch = int(iter * batch_size / total_sample_num)
	f_.close()



		
def Controller(thread_num = 2, sync_iter_num = 50, gpu_list = ['gpu0', 'gpu1'], save_model_iter = 40, total_commu_iter = 1e10, asgd_pattern = 'easgd', config_params = []):
	
	print thread_num
	print gpu_list
	parentQueue=multiprocessing.Queue(thread_num * 5)
	childQueue=[multiprocessing.Queue(thread_num * 2) for i in xrange(thread_num)]
	
	threads=[multiprocessing.Process(target = Worker,
									 args = (parentQueue, childQueue[i], i, gpu_list[i % len(gpu_list)], sync_iter_num, asgd_pattern, config_params)) 
			for i in xrange(thread_num)]
						
	print "Init Model..."
	model_params = InitModelParameter(config_params)
	commu_iter = 0
	last_iter = commu_iter * 0.9
	model_params = LoadModel(model_params, './dump/256_Emotion_LSTM_Model_not_sorted.dump_0.001_0_100_0.889705882353')
	total_loss = 0
	print model_params[4]
	
	#init all model parameter
	for i in xrange(thread_num):
		time.sleep(15)
		childQueue[i].put(model_params)
	#for thread in threads:
		threads[i].start()
		
	
	momentum_grad_params = []
	for m_p in model_params:
		momentum_grad_params.append(numpy.zeros(m_p.shape))
	print model_params[4]
	
	
	
	
	while total_commu_iter > commu_iter:
		if asgd_pattern == 'easgd':
			alpha = 0.001
			local_model_params, local_loss, i=parentQueue.get()
			for l_p, m_p in zip(local_model_params, model_params):
				diff = alpha * (l_p - m_p)
				m_p += diff
				l_p -= diff
			childQueue[i].put(local_model_params)
		
		if asgd_pattern == 'momentum':
			alpha = 0.95
			grad_params, local_loss, i=parentQueue.get()
			for g_p, m_g_p, m_p in zip(grad_params, momentum_grad_params, model_params):
				m_g_p *= alpha
				m_g_p += (1 - alpha) * g_p
				m_p += m_g_p
			childQueue[i].put(model_params)
		
		if asgd_pattern == 'asgd':
			#alpha = numpy.log(0.5 * (commu_iter - last_iter) / thread_num + 10) / 10
			alpha = 1.0 / thread_num
			if alpha > 1:
				alpha = 1
			grad_params, local_loss, i=parentQueue.get()
			for g_p, m_p in zip(grad_params, model_params):
				m_p += alpha * g_p
				#m_p *= (1 - alpha*1e-5)
			childQueue[i].put(model_params)
		
		commu_iter = commu_iter + 1
		total_loss = total_loss + local_loss 
		#print 'communication thread index', i
		
		if commu_iter % (save_model_iter) == 0:
			param_sum = 0
			param_num = 0
			for p in model_params:
				param_sum += numpy.sum(p ** 2)
				param_num += p.size
			print "#############################"
			print 'params num : ', param_num
			print 'params square sum : ', param_sum
			#print model_params[4][:5, :5]
			print "#############################"
					
		if commu_iter % save_model_iter == 0 and commu_iter > 1:
			print "#############################"
			print 'iter : ', commu_iter
			print 'loss : ', total_loss / save_model_iter
			print "#############################"
			#SaveModel(model_params, commu_iter, total_loss / save_model_iter)
			total_loss = 0
		
	for thread in threads:
		thread.join()  ## Create batches in parallel
		
		
if __name__ == '__main__':
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
	#trainData += validData
	#random.shuffle(trainData)
	#validData = trainData[:36700]
	#trainData = trainData[36700:]

	LanguageId = {}
	LanguageId['EN'] = 0
	LanguageId['LM'] = 1
	
	if not os.path.exists('./dump'):
		os.makedirs('./dump')

	config_param = {	
				'trainData' : trainData,
				'validData' : validData,
				'input_dim' : 88,
				'output_dim' : 2,
				'hidden_dim' : 256,
				'label_skip' : 5,
				'max_length' : 300,
				'batch_size' : 256,
				'learning_rate' : 0.001,
				'language_id' : LanguageId,
				'grad_clip' : 1,
			}

	Controller(thread_num = 8, sync_iter_num = 5, save_model_iter = 500, gpu_list = ['gpu3', 'gpu2', 'gpu1', 'gpu0'], total_commu_iter = 1e10, asgd_pattern = 'asgd', config_params = config_param)
		
		
		
		