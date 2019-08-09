import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN, AutoIntInteraction
from core.utils import group_embedded_by_dim


def AutoInt(
        feature_metas,
        seed=2333,
        interaction_layer_num=3,
        attention_embedding_size=8,
        attention_heads=2,
        interaction_use_res=True,
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
        name='AutoInt'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        embedded_dict = features.get_embedded_dict(slots_filter=None,
                                                   fixed_embedding_dim=fixed_embedding_dim,
                                                   embedding_initializer=embedding_initializer,
                                                   embedding_regularizer=embedding_regularizer,
                                                   group_name='embedding')
        grouped_embedded = group_embedded_by_dim(embedded_dict)
        grouped_inputs = [tf.stack(group, axis=1) for group in grouped_embedded.values()]
        for _ in range(interaction_layer_num):
            for i in range(len(grouped_inputs)):
                grouped_inputs[i] = AutoIntInteraction(
                    att_embedding_size=attention_embedding_size,
                    heads=attention_heads,
                    use_res=interaction_use_res,
                    seed=seed
                )(grouped_inputs[i])

        dnn_inputs = tf.keras.layers.Flatten()(tf.concat(grouped_inputs, axis=2))
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
        )(dnn_inputs)

        output = tf.keras.activations.sigmoid(output)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
