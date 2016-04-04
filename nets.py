from itertools import chain

import numpy


class Net(object):

    DEFAULT_HIDDEN_SIZES = (8,)
    DEFAULT_INPUTS = 16
    DEFAULT_OUTPUTS = 4

    DEFAULT_WEIGHTS = None
    DEFAULT_WEIGHT_SPREAD = 2.0
    DEFAULT_WEIGHT_MIDDLE = 0.0

    def __init__(self,
                 hidden_sizes=DEFAULT_HIDDEN_SIZES,
                 weights=DEFAULT_WEIGHTS,
                 inputs=DEFAULT_INPUTS,
                 outputs=DEFAULT_OUTPUTS,
                 weight_spread=None,
                 weight_middle=None):
        """
        @hidden_sizes: An iterable of integers that describe the sizes of the
                       hidden layers of the Net.
        @weights: May be a function that returns arrays to use as weights.
                  If so, must take an iterable of sizes to create weights for
                  and must return the same data as described below.
                  Else it must be numpy.ndarrays of dtype=float and proper sizes
                  in the proper order provided in a sliceable.
        @inputs: The integer number of inputs.
        @outputs: The integer number of outputs.
        """
        if not isinstance(inputs, int) or not isinstance(outputs, int):
            raise ValueError('Number of inputs and outputs must be integers')
        if (not hasattr(hidden_sizes, '__iter__')
                or not all(isinstance(i, int) for i in hidden_sizes)):
            raise ValueError('Sizes of hidden layers must be integers'
                             ' provided in an iterable')

        self.sizes = tuple(chain((inputs,),
                                 hidden_sizes,
                                 (outputs,)))
        if weights and callable(weights):
            weights = weights(self.sizes)
        if (weights and (not hasattr(weights, '__getslice__')
                         or not all(isinstance(arr, numpy.ndarray)
                                     for arr in weights)
                         or not all(arr.dtype == float for arr in weights))):
            raise ValueError('Weights of hidden layers must be numpy.ndarrays'
                             ' with dtype=float provided in a sliceable')

        self.inputs = inputs
        self.outputs = outputs
        self.weights = weights or Net.random_weights(self.sizes,
                                                      weight_spread,
                                                      weight_middle)
        for idx, w in enumerate(self.weights):
            assert(w.shape == (self.sizes[idx], self.sizes[idx+1]))

    @classmethod
    def random_weights(cls, sizes, spread=None, middle=None):
        sizes = tuple(sizes)
        num_layers = len(sizes) - 1
        if num_layers <= 0:
            raise ValueError('Must have more than one layer to make weights')
        spread = spread or cls.DEFAULT_WEIGHT_SPREAD
        middle = middle or cls.DEFAULT_WEIGHT_MIDDLE
        # Need weights per layer equal to layer_n*layer_(n+1)
        # For full connection
        weights = [None for _ in xrange(num_layers)]
        # Move the distribution of numbers
        adjustment = -(spread / 2.0) + middle

        for n in xrange(num_layers):
            weights_n = (sizes[n], sizes[n+1])
            weights[n] = spread*numpy.random.random(weights_n) + adjustment

        return weights

    def _sigmoid(self, vals):
        return 1 / (1 + numpy.exp(-vals))

    def _softmax(self, vals):
        e_vals = numpy.exp(vals - numpy.max(vals))
        return e_vals / e_vals.sum()

    def run(self, in_arr):
        # Row vector
        assert(in_arr.shape == (self.inputs,))
        processed = in_arr
        # Iterate over input and hidden layers
        for w in self.weights[0:-1]:
            processed = self.node(processed.dot(w))
        # Output layer
        return self.output_node(processed.dot(self.weights[-1]))

    def node(self, vals):
        return self._sigmoid(vals)

    def output_node(self, vals):
        return self._softmax(vals)

    def write_net(self, output_stream):
        pass

    def read_net(self, input_stream):
        pass
