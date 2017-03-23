__author__ = 'chuntingzhou'
from emolga.layers.recurrent import *
import numpy as np


class DenseN(Layer):
    def __init__(self, input_dims, output_dim, init='glorot_uniform', activation='tanh', name='DenseN', learn_bias=True):
        super(DenseN, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.linear = (activation == 'linear')
        self.W = []
        # self.input = T.matrix()
        for i in range(0, len(input_dims)):
            self.W.append(self.init((self.input_dims[i], self.output_dim)))

        self.b = shared_zeros(self.output_dim)

        self.learn_bias = learn_bias
        if self.learn_bias:
            self.params = self.W + [self.b]
        else:
            self.params = self.W

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        for i in range(0, len(self.W)):
            self.W[i].name = "%s_W_%d" % (name, i)
        self.b.name = '%s_b' % name

    def __call__(self, X):
        # each X[i]: (batch_size, input_dims[i])
        # Add one zero vector to each example's labels
        # batch_size = X[0].shape[0]
        labels = [T.argmax(X[i], axis=1) for i in range(0, len(X))]
        mask = [T.switch(l, 1.0, 0.0) for l in labels]
        adds = [T.dot(X[i], self.W[i]) * mask[i][:, None] for i in range(0, len(X))]
        output = self.activation(sum(adds) + self.b)
        # adds.append(T.cast(T.alloc(0.0, adds[0].shape[0], adds[0].shape[1]), dtype=theano.config.floatX))
        cxt = T.as_tensor_variable(adds).dimshuffle(1, 0, 2) # (batch_size, max_len, dim)
        # cxt = concatenate(adds, axis=1).reshape((batch_size, len(X), self.output_dim))  # (batch_size, max_len, output_dim) here max_len = #labels
        # cxt_mask = T.ones((batch_size, len(X)+1), dtype=theano.config.floatX)
        labels = T.as_tensor_variable(labels).dimshuffle(1, 0) # batch_size, max_len
        cxt_mask = T.switch(labels, 1.0, 0.0)
        return output, cxt, cxt_mask


class MLP(Layer):
    def __init__(self, hdims, init='glorot_uniform', activation='tanh', name='MLP', learn_bias=True):
        super(MLP, self).__init__()
        self.init = initializations.get(init)
        self.activations= activations.get(activation)
        self.hdims = hdims
        self.W = []
        self.b = []
        for i in range(0, len(hdims)-1):
            self.W.append(self.init((self.hdims[i], self.hdims[i+1])))
            self.b.append(shared_zeros(self.hdims[i+1]))

        self.learn_bias = learn_bias
        if learn_bias:
            self.params = self.W + self.b
        else:
            self.params = self.W

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        for i in range(0, len(self.b)):
            self.W[i].name = "%s_W_%d" % (name, i)
            self.b[i].name = "%s_b_%d" % (name, i)

    def __call__(self, X):
        for i in range(0, len(self.b)):
            X = T.dot(X, self.W[i]) + self.b[i]
        return X


class Log_py_prior(Layer):
    def __init__(self, label_list, uniform_y=True, name='y_priors'):
        super(Log_py_prior, self).__init__()
        self.priors = []
        self.uniform_y = uniform_y
        self.name = name
        for c in range(0, len(label_list)):
            self.priors.append(shared_zeros((1, label_list[c])))
            self.priors[-1].name = '%s_%d' % (self.name, c)
        self.params = self.priors

    # X[i] is the ground truth labels (1-of-K) of class i: (batch_size, #labels)
    def __call__(self, X):
        class_num = len(X)
        batch_size = X[0].shape[0]
        output = tensor.zeros((batch_size, 1))
        for c in range(0, class_num):
            if self.uniform_y:
                self.priors[c] *= 0
            py = T.nnet.softmax(T.dot(tensor.ones((batch_size, 1)), self.priors[c])) # (batch_size, #labels)
            # if no such class exits for a sample, then the row are all 0,
            # therefore the crossentropy is 0 for this sample class
            output = output + T.nnet.categorical_crossentropy(py, X[c]).reshape((batch_size, 1))
        return output


class Dense3(Layer):
    def __init__(self, input_dim1, input_dim2, input_dim3, output_dim, init='glorot_uniform', activation='tanh', name='Dense3', learn_bias=True):

        super(Dense3, self).__init__()
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.input_dim3 = input_dim3
        self.output_dim = output_dim
        self.linear = (activation == 'linear')

        # self.input = T.matrix()

        self.W1 = self.init((self.input_dim1, self.output_dim))
        self.W2 = self.init((self.input_dim2, self.output_dim))
        self.W3 = self.init((self.input_dim3, self.output_dim))
        self.b  = shared_zeros((self.output_dim))

        self.learn_bias = learn_bias
        if self.learn_bias:
            self.params = [self.W1, self.W2, self.W3, self.b]
        else:
            self.params = [self.W1, self.W2, self.W3]

        if name is not None:
            self.set_name(name)

    def set_name(self, name):
        self.W1.name = '%s_W1' % name
        self.W2.name = '%s_W2' % name
        self.W3.name = '%s_W3' % name
        self.b.name = '%s_b' % name

    def __call__(self, X1, X2, X3):
        output = self.activation(T.dot(X1, self.W1) + T.dot(X2, self.W2) + T.dot(X3, self.W3) + self.b)
        return output


class q_y_x(Layer):
    def __init__(self, input_dim, label_list, init='glorot_uniform', activation=None, name='q_y_x', learn_bias=True):
        super(q_y_x, self).__init__()
        self.input_dim = input_dim
        self.label_list = label_list
        # no mlp transformation for x representation; directly predict y with x
        self.W = []
        self.b = []
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        for i in range(0, len(label_list)):
            self.W.append(self.init((input_dim, label_list[i])))
            self.b.append(shared_zeros(label_list[i]))
            self.W[-1].name = "%s_W_%d" % (name, i)
            self.b[-1].name = "%s_b_%d" % (name, i)
        if learn_bias:
            self.params = self.W + self.b
        else:
            self.params = self.W

    def __call__(self, X, labeled=False, Y=None):
        # X: (batch_size, enc_dim) Y[i]: (batch_size, #labels)
        batch_size = X.shape[0]
        out_logpy = tensor.zeros((batch_size, 1))
        py = []
        logits = []
        #prediction = []
        for i in range(0, len(self.label_list)):
            logit = T.dot(X, self.W[i]) + self.b[i]
            logits.append(logit)
            py.append(T.nnet.softmax(logit)) # (batch_size, #label)
            #prediction.append(tensor.as_tensor_variable(tensor.argmax(py[-1], axis=1)))
            if Y is not None:
                out_logpy = out_logpy + T.nnet.categorical_crossentropy(py[-1], Y[i]).reshape((batch_size, 1))
        if labeled:
            return out_logpy, py #, prediction
        else:
            # to avoid repeated computations: since for each batch,
            # we need to enumerate all label combinations, but share the same set of softmax.
            # The take-out crossentropy is done outside
            return py, logits #, prediction