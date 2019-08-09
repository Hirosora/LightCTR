import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN, BiInteraction, SENet
from core.utils import split_tensor


def FiBiNet(
        feature_metas,
        interaction_mode='all',
        interaction_mode_se='all',
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
        name='FiBiNet'):

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
        inputs = list(embedded_dict.values())
        interactions = BiInteraction(mode=interaction_mode)(inputs)

        inputs_se = SENet(axis=-1)(tf.stack(inputs, axis=1))
        interactions_se = BiInteraction(mode=interaction_mode_se)(split_tensor(inputs_se, axis=1))

        dnn_inputs = tf.concat([interactions, interactions_se], axis=1)

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
        output = tf.keras.activations.sigmoid(dnn_output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
