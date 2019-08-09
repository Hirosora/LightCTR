import tensorflow as tf

from core.features import FeatureMetas, Features


def MLR(
        feature_metas,
        regions=10,
        embedding_initializer='glorot_uniform',
        embedding_regularizer=tf.keras.regularizers.l2(1e-5),
        fixed_embedding_dim=None,
        name='MLR'):

    assert isinstance(feature_metas, FeatureMetas)

    with tf.name_scope(name):

        features = Features(metas=feature_metas)

        inputs = features.gen_concated_feature(
            embedding_group='embedding',
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_initializer=embedding_initializer,
            embedding_regularizer=embedding_regularizer,
            slots_filter=None
        )

        region_values = tf.keras.layers.Dense(
            units=regions,
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2()
        )(inputs)
        region_values = tf.keras.activations.sigmoid(region_values)

        region_weights = tf.keras.layers.Dense(
            units=regions,
            kernel_initializer=tf.keras.initializers.RandomNormal(),
            kernel_regularizer=tf.keras.regularizers.l2()
        )(inputs)
        region_weights = tf.keras.layers.Softmax()(region_weights)

        output = tf.reduce_sum(tf.multiply(region_values, region_weights), axis=-1, keepdims=True)

        model = tf.keras.Model(inputs=features.get_inputs_list(), outputs=output)

        return model
