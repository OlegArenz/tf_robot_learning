# tf_robot_learning, a all-around tensorflow library for robotics.
#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Emmanuel Pignat <emmanuel.pignat@idiap.ch>,
#
# This file is part of tf_robot_learning.
#
# tf_robot_learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
#
# tf_robot_learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_robot_learning. If not, see <http://www.gnu.org/licenses/>.

import tensorflow as tf
from tensorflow_probability import distributions as ds


class PoE(ds.Distribution):
	def __init__(self, shape, experts, transfs, name='PoE', cost=None):
		"""

		:param shape:
		:param experts:
		:param transfs: 	a list of tensorflow function that goes from product space to expert space
			 or a function f(x: tensor, i: index of transform) -> f_i(x)

		:param cost: additional cost [batch_size, n_dim] -> [batch_size, ]
			a function f(x) ->
		"""

		self._product_shape = shape
		self._experts = experts
		self._transfs = transfs
		self._laplace = None
		self._samples_approx = None

		self._cost = cost


		self.stepsize = tf.Variable(0.01)
		self._name = name


	def get_loc_prec(self):
		raise NotImplementedError
		# return tf.concat([exp.mean() for exp in self.experts], axis=0),\
		# 	   block_diagonal_different_sizes(
		# 		   [tf.linalg.inv(exp.covariance()) for exp in self.experts])

	@property
	def product_shape(self):
		return self._product_shape

	@property
	def experts(self):
		return self._experts

	@property
	def transfs(self):
		return self._transfs


	def _experts_probs(self, x):
		probs = []

		for i, exp in enumerate(self.experts):
			if isinstance(self.transfs, list):
				if hasattr(exp, '_log_unnormalized_prob'):
					print('Using unnormalized prob for expert %d' % i)
					probs += [exp._log_unnormalized_prob(self.transfs[i](x))]
				else:
					probs += [exp.log_prob(self.transfs[i](x))]
			else:
				if hasattr(exp, '_log_unnormalized_prob'):
					probs += [exp._log_unnormalized_prob(self.transfs(x, i))]
				else:
					probs += [exp.log_prob(self.transfs(x, i))]

		return probs

	def _log_unnormalized_prob(self, x, wo_cost=False):
		if x.get_shape().ndims == 1:
			x = x[None]

		if wo_cost:
			return tf.reduce_sum(self._experts_probs(x), axis=0)
		else:
			cost = 0. if self._cost is None else self._cost(x)
			return tf.reduce_sum(self._experts_probs(x), axis=0) - cost

	@property
	def nb_experts(self):
		return len(self.experts)


	def get_transformed(self, x):
		return [f(x) for f in self.transfs]

