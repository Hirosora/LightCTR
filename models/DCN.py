import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN, CrossNetwork


def DCN(
        feature_metas,
        cross_kernel_initializer='glorot_uniform',
        cross_kernel_regularizer=tf.keras.regularizers.l2(1e-5),
        cross_bias_initializer='zeros',
        cross_bias_regularizer=None,
        cross_layers_num=3,
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fixed_embedding_dim=None,
        dnn_hidden_units=(128, 64, 1),
        dnn_activations=('relu', 'relu', None),
        dnn_use_bias=True,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_kernel_initializers='glorot_uniform',
        dnn_bias_initializers='zeros',
        dnn_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizers=None,
        name='Deep&CrossNetwork'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        embedded_dict = features.get_embedded_dict(
            group_name='embedding',
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer,
            slots_filter=None
        )

        # Deep Part
        deep_inputs = features.gen_concated_feature(
            embedding_group='embedding',
            fixed_embedding_dim=fixed_embedding_dim,
            slots_filter=None
        )
        deep_output = DNN(
            units=dnn_hidden_units,
            use_bias=dnn_use_bias,
            activations=dnn_activations,
            use_bn=dnn_use_bn,
            dropout=dnn_dropout,
            kernel_initializers=dnn_kernel_initializers,
            bias_initializers=dnn_bias_initializers,
            kernel_regularizers=dnn_kernel_regularizers,
            bias_regularizers=dnn_bias_regularizers
        )(deep_inputs)

        # Cross Part
        cross_inputs = list(embedded_dict.values())
        cross_output = CrossNetwork(
            kernel_initializer=cross_kernel_initializer,
            kernel_regularizer=cross_kernel_regularizer,
            bias_initializer=cross_bias_initializer,
            bias_regularizer=cross_bias_regularizer
        )(cross_inputs, layers_num=cross_layers_num, require_logit=True)

        output = tf.keras.activations.sigmoid(deep_output + cross_output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
