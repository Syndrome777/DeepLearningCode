import numpy as np
import theano
import theano.tensor as T
import lasagne
import cPickle
import numpy
import random
import os



l_in = lasagne.layers.InputLayer((128, 500, 560), name = "l_in")
l_reshape = lasagne.layers.ReshapeLayer(l_in, (-1, [2]), name = "l_reshape")
######################################## CNN
l_transpose = lasagne.layers.DimshuffleLayer(l_reshape, (0, 'x', 1), name = "l_transpose")
######################################## CNN on Freq and Time
######### Freq -> Time
l_conv_f_a = lasagne.layers.Conv1DLayer(
	l_transpose, num_filters = 40, filter_size = (400), stride = 1, pad = 'valid',
	W = lasagne.init.HeUniform(), nonlinearity=lasagne.nonlinearities.linear, 
	name = "l_conv_f_a"
	)
l_max = lasagne.layers.MaxPool1DLayer(l_conv_f_a, 161)

l_reshape_2 = lasagne.layers.ReshapeLayer(l_max, (128, [1], -1), name = "l_reshape")
l_transpose_2 = lasagne.layers.DimshuffleLayer(l_reshape_2, (0, 2, 1), name = "l_transpose")

l_bn_f_a = lasagne.layers.BatchNormLayer(l_transpose_2, name = "l_bn_f_a")
l_relu_f_a = lasagne.layers.NonlinearityLayer(l_bn_f_a, name = "l_relu_f_a")

#print lasagne.layers.get_output_shape(l_relu_f_a, input_shapes=(128, 100, 560))
print lasagne.layers.get_output_shape(l_relu_f_a)
print lasagne.layers.get_output_shape(l_conv_f_a)
print lasagne.layers.get_output_shape(l_max)


