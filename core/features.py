from collections import namedtuple

import tensorflow as tf


SparseFeature = namedtuple('SparseFeature',
                           ['name', 'one_hot_dim', 'embedding_dim', 'dtype'])

DenseFeature = namedtuple('DenseFeature',
                          ['name', 'dim', 'dtype'])


class FeatureType(object):
    DenseFeature = 1
    SparseFeature = 2
    UnknownFeature = -1


class FeatureMetas(object):
    """
    Meta information of all features.
    """

    def __init__(self):

        self.sparse_features = list()
        self.dense_features = list()
        self.meta_dict = dict()
        self.dense_feats_slots = list()
        self.sparse_feats_slots = list()
        self.all_feats_slots = list()

    def get_feature_type(self, name):

        if name not in self.meta_dict.keys():
            return FeatureType.UnknownFeature
        elif isinstance(self.meta_dict[name], SparseFeature):
            return FeatureType.SparseFeature
        elif isinstance(self.meta_dict[name], DenseFeature):
            return FeatureType.DenseFeature
        else:
            return FeatureType.UnknownFeature

    def add_sparse_feature(self, name, one_hot_dim, embedding_dim=32, dtype='int32'):
        """
        Add a sparse feature.

        :param name: Feature name
        :param one_hot_dim: Dimension of the feature in its one-hot encoded form
        :param embedding_dim: Dimension to which you want the feature to be embedded
        :param dtype: Data type, 'int32' as default
        :return:
        """

        self.sparse_features.append(SparseFeature(name=name, dtype=dtype,
                                                  one_hot_dim=one_hot_dim, embedding_dim=embedding_dim))
        self.meta_dict[name] = self.sparse_features[-1]
        self.sparse_feats_slots.append(name)
        self.all_feats_slots.append(name)

    def add_dense_feature(self, name, dim, dtype='float32'):
        """
        Add a dense feature.

        :param name: Feature name
        :param dim: Dimension of the feature
        :param dtype: Data type, 'float32' as default
        :return:
        """

        self.dense_features.append(DenseFeature(name=name, dim=dim, dtype=dtype))
        self.meta_dict[name] = self.dense_features[-1]
        self.dense_feats_slots.append(name)
        self.all_feats_slots.append(name)


# TODO Hash
def get_embedded(feat, one_hot_dim, embedding_dim, embedding_name,
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
    embedded = tf.squeeze(embedding(feat), axis=1)

    return embedded


def group_embedded_by_dim(embedded_dict):
    """
    Group a embedded features' dict according to embedding dimension.

    :param embedded_dict: Dict of embedded sparse features {name: 3D_embedded_feature}
    :return: Dict of grouped embedded features {embedding_dim: [3D_embedded_features]}
    """

    groups = dict()
    for embedded in embedded_dict.values():
        if embedded.shape[-1] not in groups.keys():
            groups[embedded.shape[-1]] = [embedded]
        else:
            groups[embedded.shape[-1]].append(embedded)

    return groups


class Features(object):
    """
    Manage all features
    """

    def __init__(self, metas):
        """
        Construct all features according to FeatureMetas

        :param metas: FeatureMetas object
        """
        assert isinstance(metas, FeatureMetas)

        self.metas = metas
        # Lists of feature names
        self.sparse_feats_slots, self.dense_feats_slots, self.all_feats_slots = \
            metas.sparse_feats_slots, metas.dense_feats_slots, metas.all_feats_slots

        # Dict of meta infos {name: meta}
        self.meta_dict = metas.meta_dict
        # Dict of tf.keras.layers.Input {name: Input}
        self.inputs_dict = self.gen_inputs_dict_from_metas(self.metas)

        self.embedded_groups = dict()

    @classmethod
    def gen_inputs_dict_from_metas(cls, metas):
        """
        Generate tf.keras.layers.Input according to feature meta info.

        :param metas: FeatureMetas object
        :return: Dict of tf.Input, {feature name: corresponding tf.Input}.
        """

        inputs_dict = dict()

        for meta in metas.dense_features:
            inputs_dict[meta.name] = tf.keras.layers.Input(shape=(meta.dim,), name=meta.name, dtype=meta.dtype)
        for meta in metas.sparse_features:
            inputs_dict[meta.name] = tf.keras.layers.Input(shape=(1, ), name=meta.name, dtype=meta.dtype)

        return inputs_dict

    # def gen_embedded_sparse_feature(self,
    #                                 fixed_embedding_dim=None,
    #                                 embedding_initializer='glorot_uniform',
    #                                 embedding_regularizer=tf.keras.regularizers.l2(1e-5),
    #                                 slots_filter=None):
    #     """
    #     Generate embedded dict for selected features.
    #
    #     :param fixed_embedding_dim: Integer. If this is not None, then every feature will be embedded to the same
    #         dimension, else they will be embedded to their default embedding dimension (configured in their meta info)
    #     :param embedding_initializer: Initializer for embedding
    #     :param embedding_regularizer: Regularizer for embedding
    #     :param slots_filter: List of selected slots' name
    #     :return: Dict of embedded sparse features {name: embedded_sparse_feature}
    #     """
    #
    #     embedded_dict = dict()
    #
    #     if slots_filter is None:
    #         slots_filter = self.sparse_feats_slots
    #
    #     slots_filter = list(filter(lambda slot_name: slot_name in self.sparse_feats_slots, slots_filter))
    #
    #     for slot_name in slots_filter:
    #         meta = self.meta_dict[slot_name]
    #         embedding_dim = fixed_embedding_dim if fixed_embedding_dim is not None else meta.embedding_dim
    #         embedding_name = meta.name + '_d' + str(meta.one_hot_dim) + '_to_d' + str(embedding_dim)
    #         embedded_dict[slot_name] = get_embedded(
    #             feat=self.inputs_dict[slot_name],
    #             one_hot_dim=meta.one_hot_dim,
    #             embedding_dim=embedding_dim,
    #             embedding_name=embedding_name,
    #             embedding_regularizer=embedding_regularizer,
    #             embedding_initializer=embedding_initializer
    #         )
    #
    #     return embedded_dict

    def gen_concated_feature(self,
                             embedding_group='default_group',
                             fixed_embedding_dim=None,
                             embedding_initializer='glorot_uniform',
                             embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                             slots_filter=None):
        """
        Generate a concated tensor of all the selected features. Selected sparse features will be embedded.

        :param embedding_group: String. Embedding group
        :param fixed_embedding_dim: Integer. If this is not None, then every feature will be embedded to the same
            dimension, else they will be embedded to their default embedding dimension (configured in their meta info)
        :param embedding_initializer: Initializer for embedding
        :param embedding_regularizer: Regularizer for embedding
        :param slots_filter: Dict of embedded sparse features {name: embedded_sparse_feature}
        :return: 1) Tensor. Concated features.
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        slots_filter = list(filter(lambda slot_name: slot_name in self.all_feats_slots, slots_filter))

        assert len(slots_filter) > 0

        embedded_dict = self.get_embedded_dict(
            slots_filter=slots_filter,
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_regularizer=embedding_regularizer,
            embedding_initializer=embedding_initializer,
            group_name=embedding_group
        )

        dense_part = list()
        sparse_part = list()

        for slot_name in slots_filter:
            if slot_name in self.dense_feats_slots:
                dense_part.append(self.inputs_dict[slot_name])
            elif slot_name in embedded_dict.keys():
                sparse_part.append(embedded_dict[slot_name])

        concated = tf.concat(dense_part + sparse_part, axis=1, name='concat')

        return concated

    def get_linear_logit(self,
                         use_bias=True,
                         embedding_group='dot_embedding',
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                         slots_filter=None):
        """
        Get linear logit from selected features. Selected sparse feature will be regarded as one-hot encoded.

        :param use_bias: Boolean
        :param kernel_initializer: Initializer
        :param kernel_regularizer: Regularizer
        :param slots_filter: List of selected slots' names
        :return:
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        dense_slots_filter = list(filter(lambda slot_name: slot_name in self.dense_feats_slots, slots_filter))
        sparse_slots_filter = list(filter(lambda slot_name: slot_name in self.sparse_feats_slots, slots_filter))

        assert len(dense_slots_filter) > 0 or len(sparse_slots_filter) > 0

        logits = list()

        if len(dense_slots_filter) > 0:
            concated_dense = self.gen_concated_feature(slots_filter=dense_slots_filter)
            dense_logit = tf.keras.layers.Dense(units=1,
                                                activation=None,
                                                use_bias=False,
                                                kernel_initializer=kernel_initializer,
                                                kernel_regularizer=kernel_regularizer)(concated_dense)
            logits.append(dense_logit)

        if len(sparse_slots_filter) > 0:
            concated_sparse = self.gen_concated_feature(embedding_group=embedding_group,
                                                        fixed_embedding_dim=1,
                                                        embedding_initializer=kernel_initializer,
                                                        embedding_regularizer=kernel_regularizer,
                                                        slots_filter=sparse_slots_filter)
            sparse_logit = tf.reduce_sum(concated_sparse, axis=1, keepdims=True)
            logits.append(sparse_logit)

        logits = tf.add_n(logits)
        if use_bias:
            bias = tf.Variable(0.0, trainable=True)
            logits = logits + bias

        return logits

    def get_inputs_list(self):
        """
        Get a list of all Inputs

        :return: List of all features' tf.keras.layers.Input
        """

        return list(self.inputs_dict.values())

    def get_embedded(self,
                     slot_name,
                     fixed_embedding_dim=None,
                     embedding_initializer='glorot_uniform',
                     embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                     group_name='default_group'):

        assert slot_name in self.sparse_feats_slots

        if fixed_embedding_dim is not None:
            group_name = group_name + '_' + str(fixed_embedding_dim)

        # if the group doesn't exist, create one
        if group_name not in self.embedded_groups:
            self.embedded_groups[group_name] = dict()
        group = self.embedded_groups[group_name]

        # if the slot is not in this group, make a new embedding for it.
        if slot_name not in group:
            feat_input = self.inputs_dict[slot_name]
            feat_meta = self.meta_dict[slot_name]
            embedding_dim = fixed_embedding_dim if fixed_embedding_dim is not None else feat_meta.embedding_dim
            embedding_name = \
                group_name + '/' + feat_meta.name + '_d' + str(feat_meta.one_hot_dim) + '_to_d' + str(embedding_dim)
            group[slot_name] = get_embedded(
                feat=feat_input,
                one_hot_dim=feat_meta.one_hot_dim,
                embedding_dim=embedding_dim,
                embedding_name=embedding_name,
                embedding_initializer=embedding_initializer,
                embedding_regularizer=embedding_regularizer
            )

        return group[slot_name]

    def get_embedded_dict(self,
                          slots_filter=None,
                          fixed_embedding_dim=None,
                          embedding_initializer='glorot_uniform',
                          embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                          group_name='default_group'):

        if slots_filter is None:
            slots_filter = self.sparse_feats_slots

        slots_filter = list(filter(lambda slot_name: slot_name in self.sparse_feats_slots, slots_filter))

        if fixed_embedding_dim is not None:
            group_name = group_name + '_' + str(fixed_embedding_dim)

        # if the group doesn't exist, create one
        if group_name not in self.embedded_groups:
            self.embedded_groups[group_name] = dict()
        group = self.embedded_groups[group_name]

        embedded_dict = dict()

        # if the slot is not in this group, make a new embedding for it.
        for slot_name in slots_filter:
            if slot_name not in group:
                feat_input = self.inputs_dict[slot_name]
                feat_meta = self.meta_dict[slot_name]
                embedding_dim = fixed_embedding_dim if fixed_embedding_dim is not None else feat_meta.embedding_dim
                embedding_name = \
                    group_name + '/' + feat_meta.name + '_d' + str(feat_meta.one_hot_dim) + '_to_d' + str(embedding_dim)
                group[slot_name] = get_embedded(
                    feat=feat_input,
                    one_hot_dim=feat_meta.one_hot_dim,
                    embedding_dim=embedding_dim,
                    embedding_name=embedding_name,
                    embedding_initializer=embedding_initializer,
                    embedding_regularizer=embedding_regularizer
                )
            embedded_dict[slot_name] = group[slot_name]

        return embedded_dict
