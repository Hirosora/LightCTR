import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN


def CCPM(
        feature_metas,
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fixed_embedding_dim=32,
        cnn_filters=(4, 4, 2),
        cnn_kernel_widths=(5, 5, 3),
        dnn_hidden_units=(128, 64, 1),
        dnn_activations=('relu', 'relu', None),
        dnn_use_bias=True,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_kernel_initializers='glorot_uniform',
        dnn_bias_initializers='zeros',
        dnn_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizers=None,
        name='CCPM'):

    assert isinstance(feature_metas, FeatureMetas)
    assert fixed_embedding_dim is not None

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        inputs = features.get_stacked_feature(embedding_group='embedding',
                                              fixed_embedding_dim=fixed_embedding_dim,
                                              embedding_initializer=embedding_initializer,
                                              embedding_regularizer=embedding_regularizer,
                                              slots_filter=None,
                                              list_sparse_embedding_aggregater='mean')
        inputs = tf.expand_dims(inputs, axis=-1)

        l = len(cnn_filters)
        n = int(inputs.shape[1])
        for i, pack in enumerate(zip(cnn_filters, cnn_kernel_widths)):
            filter_num, width = pack
            inputs = tf.keras.layers.Conv2D(
                filters=filter_num,
                kernel_size=(width, 1),
                strides=(1, 1),
                padding='same',
                activation='relu',
                use_bias=True
            )(inputs)

            idx = i + 1
            p = 3 if idx == l else max(1, int(n * (1 - (idx / l) ** (l - idx))))
            inputs = tf.math.top_k(input=tf.transpose(inputs, [0, 3, 2, 1]), k=p)[0]
            inputs = tf.transpose(inputs, [0, 3, 2, 1])

        inputs = tf.keras.layers.Flatten()(inputs)
        output = DNN(
            units=dnn_hidden_units,
            use_bias=dnn_use_bias,
            activations=dnn_activations,
            use_bn=dnn_use_bn,
            dropout=dnn_dropout,
            kernel_initializers=dnn_kernel_initializers,
            bias_initializers=dnn_bias_initializers,
            kernel_regularizers=dnn_kernel_regularizers,
            bias_regularizers=dnn_bias_regularizers
        )(inputs)

        output = tf.keras.activations.sigmoid(output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
