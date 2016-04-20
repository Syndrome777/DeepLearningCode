from lasagne.layers.base import Layer
from lasagne import init, nonlinearities

import numpy as np
import theano.tensor as T

class DenseLayer3D(Layer):
	def __init__(self, incoming, num_units, W=init.GlorotUniform(),
				 b=init.Constant(0.), nonlinearity=nonlinearities.rectify,
				 **kwargs):
		super(DenseLayer3D, self).__init__(incoming, **kwargs)
		self.nonlinearity = (nonlinearities.identity if nonlinearity is None
							 else nonlinearity)
							 
		assert len(self.input_shape) > 2

		self.num_units = num_units

		num_inputs = int(np.prod(self.input_shape[2:]))

		self.W = self.add_param(W, (num_inputs, num_units), name="W")
		if b is None:
			self.b = None
		else:
			self.b = self.add_param(b, (num_units,), name="b",
									regularizable=False)

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[1], self.num_units)

	def get_output_for(self, input, **kwargs):
		#if input.ndim > 2:
			# if the input has more than two dimensions, flatten it into a
			# batch of feature vectors.
		#	input = input.flatten(2)

		activation = T.dot(input, self.W)
		if self.b is not None:
			activation = activation + self.b.dimshuffle('x', 'x', 0)
		return self.nonlinearity(activation)
	








