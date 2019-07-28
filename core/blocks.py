from collections import Iterable

import tensorflow as tf
from tensorflow.python.keras import layers


def get_activation(activation):
    if activation is None:
        return None
    elif isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif issubclass(activation, tf.keras.layers.Layer):
        act_layer = activation()
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % activation)
    return act_layer


class DNN(tf.keras.Model):
    """
        Deep Neural Network
    """

    def __init__(self,
                 units,
                 use_bias=True,
                 use_bn=False,
                 dropout=0,
                 activations=None,
                 kernel_initializers='glorot_uniform',
                 bias_initializers='zeros',
                 kernel_regularizers=tf.keras.regularizers.l2(1e-5),
                 bias_regularizers=None,
                 **kwargs):
        """
        :param units:
            An iterable of hidden layers' neural units' number, its length is the depth of the DNN.
        :param use_bias:
            Iterable/Boolean.
            If this is not iterable, every layer of the DNN will have the same param, the same below.
        :param activations:
            Iterable/String/TF activation class
        :param kernel_initializers:
            Iterable/String/TF initializer class
        :param bias_initializers:
            Iterable/String/TF initializer class
        :param kernel_regularizers:
            Iterable/String/TF regularizer class
        :param bias_regularizers:
            Iterable/String/TF regularizer class
        """

        super(DNN, self).__init__(**kwargs)

        self.units = units
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout = dropout
        self.activations = activations
        self.kernel_initializers = kernel_initializers
        self.bias_initializers = bias_initializers
        self.kernel_regularizers = kernel_regularizers
        self.bias_regularizers = bias_regularizers

        if not isinstance(self.use_bias, Iterable):
            self.use_bias = [self.use_bias] * len(self.units)

        if not isinstance(self.use_bn, Iterable):
            self.use_bn = [self.use_bn] * len(self.units)

        if not isinstance(self.dropout, Iterable):
            self.dropout = [self.dropout] * len(self.units)

        if not isinstance(self.activations, Iterable):
            self.activations = [self.activations] * len(self.units)

        if isinstance(self.kernel_initializers, str) or not isinstance(self.kernel_initializers, Iterable):
            self.kernel_initializers = [self.kernel_initializers] * len(self.units)

        if isinstance(self.bias_initializers, str) or not isinstance(self.bias_initializers, Iterable):
            self.bias_initializers = [self.bias_initializers] * len(self.units)

        if isinstance(self.kernel_regularizers, str) or not isinstance(self.kernel_regularizers, Iterable):
            self.kernel_regularizers = [self.kernel_regularizers] * len(self.units)

        if isinstance(self.bias_regularizers, str) or not isinstance(self.bias_regularizers, Iterable):
            self.bias_regularizers = [self.bias_regularizers] * len(self.units)

        self.mlp = tf.keras.Sequential()
        for i in range(len(self.units)):
            self.mlp.add(layers.Dense(
                units=self.units[i],
                activation=self.activations[i],
                use_bias=self.use_bias[i],
                kernel_initializer=self.kernel_initializers[i],
                bias_initializer=self.bias_initializers[i],
                kernel_regularizer=self.kernel_regularizers[i],
                bias_regularizer=self.bias_regularizers[i]
            ))
            if self.dropout[i] > 0:
                self.mlp.add(layers.Dropout(self.dropout[i]))
            if self.use_bn[i]:
                self.mlp.add(layers.BatchNormalization())

    def call(self, inputs, **kwargs):

        output = self.mlp(inputs)

        return output


class FM(tf.keras.Model):
    """
        Factorization Machine Block
        compute cross features (order-2) and return their sum (without linear term)
    """

    def __init__(self, **kwargs):

        super(FM, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """
        :param inputs:
            list of 2D tensor with shape [batch_size, number_of_features, embedding_size]
            all the features should be embedded and have the same embedding size
        :return:
            2D tensor with shape [batch_size, 1]
            sum of all cross features
        """

        inputs_3d = tf.stack(inputs, axis=1)

        # (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2 * ab, we need the cross feature "ab"
        square_of_sum = tf.square(tf.reduce_sum(inputs_3d, axis=1, keepdims=False))
        sum_of_square = tf.reduce_sum(tf.square(inputs_3d), axis=1, keepdims=False)
        outputs = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)

        return outputs


class InnerProduct(tf.keras.Model):

    def __init__(self, **kwargs):

        super(InnerProduct, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):

        inner_products_list = list()

        for i in range(len(inputs) - 1):
            for j in range(i + 1, len(inputs)):
                inner_products_list.append(tf.reduce_sum(tf.multiply(inputs[i], inputs[j]), axis=1, keepdims=True))

        inner_product_layer = tf.concat(inner_products_list, axis=1)

        return inner_product_layer


class OuterProduct(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 **kwargs):

        super(OuterProduct, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def call(self, inputs, **kwargs):

        outer_products_list = list()

        for i in range(len(inputs) - 1):
            for j in range(i + 1, len(inputs)):
                inp_i = tf.expand_dims(inputs[i], axis=1)
                inp_j = tf.expand_dims(inputs[j], axis=-1)
                kernel = self.add_weight(shape=(inp_i.shape[2], inp_j.shape[1]),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         trainable=True)
                product = tf.reduce_sum(tf.matmul(tf.matmul(inp_i, kernel), inp_j), axis=-1, keepdims=False)
                outer_products_list.append(product)

        outer_product_layer = tf.concat(outer_products_list, axis=1)

        return outer_product_layer


class CrossNetwork(tf.keras.Model):

    def __init__(self,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                 **kwargs):

        super(CrossNetwork, self).__init__(**kwargs)

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer

    def call(self, inputs, layers_num=3, require_logit=True, **kwargs):

        x0 = tf.expand_dims(tf.concat(inputs, axis=1), axis=-1)
        x = tf.transpose(x0, [0, 2, 1])

        for i in range(layers_num):
            kernel = self.add_weight(shape=(x0.shape[1], 1),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)
            bias = self.add_weight(shape=(x0.shape[1], 1),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   trainable=True)
            x = tf.matmul(tf.matmul(x0, x), kernel) + bias + tf.transpose(x, [0, 2, 1])
            x = tf.transpose(x, [0, 2, 1])

        x = tf.squeeze(x, axis=1)
        if require_logit:
            kernel = self.add_weight(shape=(x0.shape[1], 1),
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     trainable=True)
            x = tf.matmul(x, kernel)

        return x
