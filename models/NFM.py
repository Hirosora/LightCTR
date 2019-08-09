import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN, FM
from core.utils import group_embedded_by_dim


def NFM(
        feature_metas,
        linear_slots,
        fm_slots,
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fm_fixed_embedding_dim=None,
        linear_use_bias=True,
        linear_kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4, seed=1024),
        linear_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        dnn_hidden_units=(128, 64, 1),
        dnn_activations=('relu', 'relu', None),
        dnn_use_bias=True,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_kernel_initializers='glorot_uniform',
        dnn_bias_initializers='zeros',
        dnn_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizers=None,
        name='NFM'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        # Linear Part
        with tf.name_scope('Linear'):
            linear_output = features.get_linear_logit(use_bias=linear_use_bias,
                                                      kernel_initializer=linear_kernel_initializer,
                                                      kernel_regularizer=linear_kernel_regularizer,
                                                      embedding_group='dot_embedding',
                                                      slots_filter=linear_slots)

        # Bi-interaction
        with tf.name_scope('BiInteraction'):
            fm_embedded_dict = features.get_embedded_dict(group_name='embedding',
                                                          fixed_embedding_dim=fm_fixed_embedding_dim,
                                                          embedding_initializer=embedding_initializer,
                                                          embedding_regularizer=embedding_regularizer,
                                                          slots_filter=fm_slots)
            fm_dim_groups = group_embedded_by_dim(fm_embedded_dict)
            fms = [FM()(group, require_logit=False) for group in fm_dim_groups.values() if len(group) > 1]
            dnn_inputs = tf.concat(fms, axis=1)
            dnn_output = DNN(
                units=dnn_hidden_units,
                use_bias=dnn_use_bias,
                activations=dnn_activations,
                use_bn=dnn_use_bn,
                dropout=dnn_dropout,
                kernel_initializers=dnn_kernel_initializers,
                bias_initializers=dnn_bias_initializers,
                kernel_regularizers=dnn_kernel_regularizers,
                bias_regularizers=dnn_bias_regularizers
            )(dnn_inputs)

        # Output
        output = tf.add_n([linear_output, dnn_output])
        output = tf.keras.activations.sigmoid(output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
