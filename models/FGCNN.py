import tensorflow as tf

from core.features import FeatureMetas, Features
from core.blocks import DNN, InnerProduct, FGCNNlayer
from core.utils import split_tensor


def FGCNN(
        feature_metas,
        fg_filters=(14, 16, 18, 20),
        fg_widths=(7, 7, 7, 7),
        fg_pool_widths=(2, 2, 2, 2),
        fg_new_feat_filters=(3, 3, 3, 3),
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fixed_embedding_dim=8,
        dnn_hidden_units=(128, 64, 1),
        dnn_activations=('relu', 'relu', None),
        dnn_use_bias=True,
        dnn_use_bn=False,
        dnn_dropout=0,
        dnn_kernel_initializers='glorot_uniform',
        dnn_bias_initializers='zeros',
        dnn_kernel_regularizers=tf.keras.regularizers.l2(1e-5),
        dnn_bias_regularizers=None,
        name='FGCNN'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        raw_feats = features.get_stacked_feature(
            embedding_group='raw',
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer,
            slots_filter=None
        )

        fg_inputs = features.get_stacked_feature(
            embedding_group='fgcnn',
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer,
            slots_filter=None
        )
        fg_inputs = tf.expand_dims(fg_inputs, axis=-1)

        new_feats_list = list()
        for filters, width, pool, new_filters in zip(fg_filters, fg_widths, fg_pool_widths, fg_new_feat_filters):
            fg_inputs, new_feats = FGCNNlayer(
                filters=filters,
                kernel_width=width,
                pool_width=pool,
                new_feat_filters=new_filters
            )(fg_inputs)
            new_feats_list.append(new_feats)

        inputs = tf.concat(new_feats_list + [raw_feats], axis=1)
        inputs = split_tensor(inputs, axis=1)

        inputs_fm = InnerProduct(require_logit=False)(inputs)

        dnn_inputs = tf.concat(inputs + [inputs_fm], axis=1)
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
