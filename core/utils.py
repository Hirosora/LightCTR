import tensorflow as tf

# from core.features import SparseFeature, DenseFeature, ListSparseFeature, FeatureMetas


def gen_inputs_dict_from_metas(metas):
    """
    Generate tf.keras.layers.Input according to feature meta info.

    :param metas: FeatureMetas object
    :return: Dict of tf.Input, {feature name: corresponding tf.Input}.
    """

    # assert isinstance(metas, FeatureMetas)

    inputs_dict = dict()

    for meta in metas.dense_features:
        # assert isinstance(meta, DenseFeature)
        inputs_dict[meta.name] = tf.keras.layers.Input(shape=(meta.dim, ), name=meta.name, dtype=meta.dtype)
    for meta in metas.sparse_features:
        # assert isinstance(meta, SparseFeature)
        inputs_dict[meta.name] = tf.keras.layers.Input(shape=(1, ), name=meta.name, dtype=meta.dtype)
    for meta in metas.list_sparse_features:
        # assert isinstance(meta, ListSparseFeature)
        inputs_dict[meta.name] = tf.keras.layers.Input(shape=(meta.max_length, ), name=meta.name, dtype=meta.dtype)

    return inputs_dict


def get_embedded(feat, one_hot_dim, embedding_dim, embedding_name,
                 hash=False,
                 embedding_initializer='glorot_uniform',
                 embedding_regularizer=tf.keras.regularizers.l2(1e-5)):
    """
    Build embedding layer for sparse features according to their meta info.

    :param feat: Tensor.
    :param one_hot_dim: Integer. Dimension of the feature in its one-hot encoded form.
    :param embedding_dim: Integer. Dimension to which you want the feature to be embedded.
    :param embedding_name: String. Name of the embedding layer.
    :param embedding_initializer:
    :param embedding_regularizer:
    :return: 2D Tensor. Embedded feature. [batch_size, embedding_dim]
    """

    embedding = tf.keras.layers.Embedding(
        input_dim=one_hot_dim,
        output_dim=embedding_dim,
        embeddings_initializer=embedding_initializer,
        embeddings_regularizer=embedding_regularizer,
        name=embedding_name
    )
    if hash:
        embedded = embedding(tf.math.mod(feat, one_hot_dim))
    else:
        embedded = embedding(feat)

    if embedded.shape[1] == 1:
        embedded = tf.squeeze(embedded, axis=1)

    return embedded


def get_dense_embedded(feat, embedding_dim, embedding_name,
                       embedding_initializer='glorot_uniform',
                       embedding_regularizer=tf.keras.regularizers.l2(1e-5)):

    embedded = tf.keras.layers.Dense(
        embedding_dim,
        use_bias=False,
        kernel_initializer=embedding_initializer,
        kernel_regularizer=embedding_regularizer,
        name=embedding_name
    )(feat)

    return embedded


def group_embedded_by_dim(embedded_dict):
    """
    Group a embedded features' dict according to embedding dimension.

    :param embedded_dict: Dict of embedded sparse features {name: embedded_feature}
    :return: Dict of grouped embedded features {embedding_dim: [embedded_features]}
    """

    groups = dict()
    for embedded in embedded_dict.values():
        if embedded.shape[-1] not in groups.keys():
            groups[embedded.shape[-1]] = [embedded]
        else:
            groups[embedded.shape[-1]].append(embedded)

    return groups


def split_tensor(inputs, axis=0):
    """
    split a tensor along an axis into a list of tensor
    the splited axis will be squeezed

    :param inputs: Tensor (n dimension)
    :param axis: Integer
    :return: List of inputs[axis] tensors (n - 1 dimension).
    """

    l = tf.split(inputs, inputs.shape[axis], axis=axis)
    l = [tf.squeeze(element, axis=axis) for element in l]

    return l
