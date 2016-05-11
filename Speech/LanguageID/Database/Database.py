import numpy
import random
import cPickle

random.seed(1234)

class Database(object):
	def __init__(self):
		self.trainData = []
		self.validData = []
		self.local_valid_data = []
		self.local_train_data = []
		self.total_valid_data = []
		self.total_train_data = []
		self.SAMPLE_NUM = 0
		self.loadFromDisk = True
		self.loadDataSource = ''
		
	def __iter__(self):
		return self
		
	def next(self):
		print "Next"
		
	def init(self, trainData, validData, data_root, loadFromDisk = True, loadDataSource = 'signal'):
		self.trainData = trainData
		self.validData = validData
		self.SAMPLE_NUM = len(self.trainData)
		self.data_root = data_root
		self.loadFromDisk = loadFromDisk
		self.loadDataSource = loadDataSource
		
	def _loadDumpFile(self, file_name, data_root = ''):
		data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/GetSpe/data/"
		with open(data_root + file_name, 'rb') as f_:
			data = cPickle.load(f_)
		#[d, label, user_name, file_name] = data
		return data
		
	def _loadNpyFileFromLFB(self, file_name, ID2Language = ['EN', 'LM']):
		data_root = self.data_root
		#data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/Get80DimsFea/data"	
		#data_root = "\\\\speech-tesla05\\d$\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\Get80DimsFea\\data"
		fea, slice_info, label = numpy.load(data_root + file_name)
		new_fea = []
		for sli in slice_info:
			[start_i, end_i] = sli
			if start_i > 1:
				start_i -= 1
			new_fea.append(fea[start_i : end_i])
		new_fea = numpy.concatenate(new_fea, axis=0)
		data = [new_fea, ID2Language[label]]
		#data = numpy.asarray([fea, slice_info, label])
		return data
	
	def _loadNpyFileFromSignal(self, file_name, ID2Language = ['EN', 'LM']):
		data_root = self.data_root
		#data_root = "D:/users/v-lifenh/Root_LanguageIdentification/GetLanguageIdentificationData/Feature/GetSpe/data_sig"	
		#data_root = "\\\\speech-tesla05\\d$\\users\\v-lifenh\\Root_LanguageIdentification\\GetLanguageIdentificationData\\Feature\\GetSpe\\data_sig"
		fea, slice_info, label = numpy.load(data_root + file_name)
		new_fea = []
		for sli in slice_info:
			[start_i, end_i] = sli
			if start_i > 1:
				start_i -= 1
			new_fea.append(fea[start_i*160 : end_i*160])
		new_fea = numpy.concatenate(new_fea, axis=0)
		data = [new_fea, ID2Language[label]]
		#data = numpy.asarray([fea, slice_info, label])
		return data

	def _loadLocalValidData(self, min_len = 48000):
		for file_name in self.validData:
			#sample = self._loadDumpFile(file_name)
			if self.loadDataSource == "signal":
				sample = self._loadNpyFileFromSignal(file_name)
			else:
				sample = self._loadNpyFileFromLFB(file_name)
			if sample[0].shape[0] >= min_len:
				self.local_valid_data.append([sample, sample[0].shape[0]])
		self.local_valid_data = sorted(self.local_valid_data, key=lambda x:x[1], reverse=1)
	
	def _loadTotalValidData(self, min_len = 48000):
		self._loadLocalValidData(min_len)
		
	def _loadLocalTrainingData(self, index, samp_num, min_len = 0, sort_data = True):
		for i in xrange(samp_num):
			if self.loadFromDisk == True:
				if self.loadDataSource == "signal":
					sample = self._loadNpyFileFromSignal(self.trainData[(index + i) % self.SAMPLE_NUM])
				else:
					sample = self._loadNpyFileFromLFB(self.trainData[(index + i) % self.SAMPLE_NUM])
			else:
				sample = self.total_train_data[(index + i) % self.SAMPLE_NUM]
			self.local_train_data.append([sample, sample[0].shape[0]])
		if sort_data == True:
			self.local_train_data = sorted(self.local_train_data, key=lambda x:x[1], reverse=0)
		
	def _loadTotalTrainingData(self, min_len = 0, rand_data = True):
		'''
		for file_name in self.trainData:
			#sample = self._loadDumpFile(file_name)
			if self.loadDataSource == "signal":
				sample = self._loadNpyFileFromSignal(file_name)
			else:
				sample = self._loadNpyFileFromLFB(file_name)
			if sample[0].shape[0] >= min_len:
				self.total_train_data.append(sample)
		'''
		'''
		with open('../data/88DimFeaData.dump', 'rb') as f_:
			AllData = cPickle.load(f_)
		for file_name in self.trainData:
			sample = AllData[file_name]
			if sample[0].shape[0] >= min_len:
				self.total_train_data.append(sample)
		AllData = []
		'''
		with open('../data/88DimFeaData_Training.dump', 'rb') as f_:
			self.total_train_data = cPickle.load(f_)
		if rand_data == True:
			random.shuffle(self.total_train_data)
			
	def loadAllData(self, valid_min_len = 48000, train_min_len = 0, rand_data = True):
		self._loadTotalValidData(valid_min_len)
		if self.loadFromDisk == False:
			self._loadTotalTrainingData(train_min_len, rand_data)
			
	
	def loadBatchData(self, index, samp_num, min_len = 0, sort_data = True):
		self._loadLocalTrainingData(index, samp_num, min_len, sort_data)
			
	def resetTrainingData(self):
		if self.loadFromDisk == True:
			random.shuffle(self.trainData)
		else:
			random.shuffle(self.total_train_data)
			
	
























