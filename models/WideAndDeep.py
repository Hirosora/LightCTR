import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN


def WideAndDeep(
        feature_metas,
        wide_slots,
        deep_slots,
        embedding_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1e-4),
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        wide_use_bias=True,
        wide_kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-4, seed=1024),
        wide_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        deep_fixed_embedding_dim=None,
        deep_hidden_units=(128, 64, 1),
        deep_activations=('relu', 'relu', None),
        deep_use_bias=True,
        deep_use_bn=False,
        deep_dropout=0,
        deep_kernel_initializers='glorot_uniform',
        deep_bias_initializers='zeros',
        deep_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        deep_bias_regularizers=None,
        name='Wide&Deep'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        # Wide Part
        with tf.name_scope('Wide'):
            wide_output = features.get_linear_logit(embedding_group='dot_embedding',
                                                    use_bias=wide_use_bias,
                                                    kernel_initializer=wide_kernel_initializer,
                                                    kernel_regularizer=wide_kernel_regularizer,
                                                    slots_filter=wide_slots)

        # Deep Part
        with tf.name_scope('Deep'):
            deep_inputs = features.gen_concated_feature(embedding_group='embedding',
                                                        fixed_embedding_dim=deep_fixed_embedding_dim,
                                                        embedding_initializer=embedding_initializer,
                                                        embedding_regularizer=embedding_regularizer,
                                                        slots_filter=deep_slots)
            deep_output = DNN(
                units=deep_hidden_units,
                use_bias=deep_use_bias,
                activations=deep_activations,
                use_bn=deep_use_bn,
                dropout=deep_dropout,
                kernel_initializers=deep_kernel_initializers,
                bias_initializers=deep_bias_initializers,
                kernel_regularizers=deep_kernel_regularizers,
                bias_regularizers=deep_bias_regularizers
            )(deep_inputs)

        # Output
        output = tf.add_n([wide_output, deep_output])
        output = tf.keras.activations.sigmoid(output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
