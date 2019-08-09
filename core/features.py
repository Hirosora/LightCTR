from collections import namedtuple

import tensorflow as tf

from core.utils import get_embedded, gen_inputs_dict_from_metas, get_dense_embedded


SparseFeature = namedtuple(
    typename='SparseFeature',
    field_names=['name', 'one_hot_dim', 'embedding_dim', 'hash', 'dtype']
)

DenseFeature = namedtuple(
    typename='DenseFeature',
    field_names=['name', 'dim', 'embedding_dim', 'dtype']
)

ListSparseFeature = namedtuple(
    typename='ListSparseFeature',
    field_names=['name', 'max_length', 'one_hot_dim', 'embedding_dim', 'hash', 'dtype']
)


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
        self.list_sparse_features = list()

        self.meta_dict = dict()

        self.dense_feats_slots = list()
        self.sparse_feats_slots = list()
        self.list_sparse_feats_slots = list()
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

    def add_sparse_feature(self, name, one_hot_dim, embedding_dim=32, hash=False, dtype='int32'):
        """
        Add a sparse feature.

        :param name: Feature name
        :param one_hot_dim: Dimension of the feature in its one-hot encoded form
        :param embedding_dim: Dimension to which you want the feature to be embedded
        :param dtype: Data type, 'int32' as default
        :return:
        """

        self.sparse_features.append(SparseFeature(name=name, dtype=dtype, hash=hash,
                                                  one_hot_dim=one_hot_dim, embedding_dim=embedding_dim))
        self.meta_dict[name] = self.sparse_features[-1]
        self.sparse_feats_slots.append(name)
        self.all_feats_slots.append(name)

    def add_dense_feature(self, name, dim, embedding_dim=32, dtype='float32'):
        """
        Add a dense feature.

        :param name: Feature name
        :param dim: Dimension of the feature
        :param dtype: Data type, 'float32' as default
        :return:
        """

        self.dense_features.append(DenseFeature(name=name, dim=dim, embedding_dim=embedding_dim, dtype=dtype))
        self.meta_dict[name] = self.dense_features[-1]
        self.dense_feats_slots.append(name)
        self.all_feats_slots.append(name)

    def add_list_sparse_feature(self, name, max_length, one_hot_dim, embedding_dim=32, hash=False, dtype='float32'):
        """
        Add a list sparse feature whose length is not fiexed

        :param max_length: Integer. Max length of the feature list
        :param name: Feature name
        :param one_hot_dim: Dimension of the feature in its one-hot encoded form
        :param embedding_dim: Dimension to which you want the feature to be embedded
        :param dtype: Data type, 'int32' as default
        :return:
        """

        self.list_sparse_features.append(ListSparseFeature(name=name, dtype=dtype, max_length=max_length, hash=hash,
                                                           one_hot_dim=one_hot_dim, embedding_dim=embedding_dim))
        self.meta_dict[name] = self.list_sparse_features[-1]
        self.list_sparse_feats_slots.append(name)
        self.all_feats_slots.append(name)


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
        self.list_sparse_feats_slots = metas.list_sparse_feats_slots

        # Dict of meta infos {name: meta}
        self.meta_dict = metas.meta_dict
        # Dict of tf.keras.layers.Input {name: Input}
        self.inputs_dict = gen_inputs_dict_from_metas(self.metas)

        self.embedded_groups = dict()

    def get_inputs_list(self):
        """
        Get a list of all Inputs

        :return: List of all features' tf.keras.layers.Input
        """

        return list(self.inputs_dict.values())

    def gen_concated_feature(self,
                             embedding_group='default_group',
                             fixed_embedding_dim=None,
                             embedding_initializer='glorot_uniform',
                             embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                             slots_filter=None,
                             list_sparse_embedding_aggregater='mean'):
        """
        Generate a concated tensor of all the selected features. Selected sparse features will be embedded.

        :param embedding_group: String. Embedding group
        :param fixed_embedding_dim: Integer. If this is not None, then every feature will be embedded to the same
            dimension, else they will be embedded to their default embedding dimension (configured in their meta info)
        :param embedding_initializer: Initializer for embedding
        :param embedding_regularizer: Regularizer for embedding
        :param slots_filter: Dict of embedded sparse features {name: embedded_sparse_feature}
        :param list_sparse_embedding_aggregater: String. 'mean' or 'sum' or None, method to aggregate list feature
        :return: Tensor [batch_size, concated_size]. Concated features.
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        slots_filter = list(filter(lambda slot_name: slot_name in self.all_feats_slots, slots_filter))
        sparse_slots_filter = [slot_name for slot_name in slots_filter if slot_name in self.sparse_feats_slots]

        assert len(slots_filter) > 0

        embedded_dict = self.get_embedded_dict(
            slots_filter=sparse_slots_filter,
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_regularizer=embedding_regularizer,
            embedding_initializer=embedding_initializer,
            group_name=embedding_group,
            aggregater=list_sparse_embedding_aggregater
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

    def get_stacked_feature(self,
                            embedding_group='default_group',
                            fixed_embedding_dim=32,
                            embedding_initializer='glorot_uniform',
                            embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                            slots_filter=None,
                            list_sparse_embedding_aggregater='mean'):
        """
        Generate a stacked tensor of all the selected features. All features will be embedded to a same dimension.

        :param embedding_group: String. Embedding group
        :param fixed_embedding_dim: Integer. If this is not None, then every feature will be embedded to the same
            dimension, else they will be embedded to their default embedding dimension (configured in their meta info)
        :param embedding_initializer: Initializer for embedding
        :param embedding_regularizer: Regularizer for embedding
        :param slots_filter: Dict of embedded sparse features {name: embedded_sparse_feature}
        :param list_sparse_embedding_aggregater: String. 'mean' or 'sum' or None, method to aggregate list feature
        :return: Tensor [batch_size, features_num, embedding_size]. Stacked features.
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        slots_filter = list(filter(lambda slot_name: slot_name in self.all_feats_slots, slots_filter))

        assert len(slots_filter) > 0
        assert fixed_embedding_dim is not None

        embedded_dict = self.get_embedded_dict(
            slots_filter=slots_filter,
            fixed_embedding_dim=fixed_embedding_dim,
            embedding_regularizer=embedding_regularizer,
            embedding_initializer=embedding_initializer,
            group_name=embedding_group,
            aggregater=list_sparse_embedding_aggregater
        )

        embedded_list = list()
        for embedded in embedded_dict.values():
            if len(embedded.shape) == 3:
                embedded_list.append(embedded)
            else:
                embedded_list.append(tf.expand_dims(embedded, axis=1))

        return tf.concat(embedded_list, axis=1)

    def get_linear_logit(self,
                         use_bias=True,
                         embedding_group='dot_embedding',
                         kernel_initializer='glorot_uniform',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                         slots_filter=None,
                         list_sparse_embedding_aggregater='mean'):
        """
        Get linear logit from selected features. Selected sparse feature will be regarded as one-hot encoded.

        :param use_bias: Boolean
        :param embedding_group: String. Name of embedding group
        :param kernel_initializer: Initializer
        :param kernel_regularizer: Regularizer
        :param slots_filter: List of selected slots' names
        :return:
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        dense_slots_filter = list(filter(lambda slot_name: slot_name in self.dense_feats_slots, slots_filter))
        sparse_slots_filter = list(filter(lambda slot_name: slot_name in self.sparse_feats_slots + self.list_sparse_feats_slots, slots_filter))

        assert len(dense_slots_filter) > 0 or len(sparse_slots_filter) > 0

        logits = list()

        if len(dense_slots_filter) > 0:
            concated_dense = self.gen_concated_feature(slots_filter=dense_slots_filter)
            dense_logit = tf.keras.layers.Dense(
                units=1,
                activation=None,
                use_bias=False,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer
            )(concated_dense)
            logits.append(dense_logit)

        if len(sparse_slots_filter) > 0:
            concated_sparse = self.gen_concated_feature(
                embedding_group=embedding_group,
                fixed_embedding_dim=1,
                embedding_initializer=kernel_initializer,
                embedding_regularizer=kernel_regularizer,
                slots_filter=sparse_slots_filter,
                list_sparse_embedding_aggregater=list_sparse_embedding_aggregater
            )
            sparse_logit = tf.reduce_sum(concated_sparse, axis=1, keepdims=True)
            logits.append(sparse_logit)

        logits = tf.add_n(logits)
        if use_bias:
            bias = tf.Variable(0.0, trainable=True)
            logits = logits + bias

        return logits

    def get_embedded(self,
                     slot_name,
                     fixed_embedding_dim=None,
                     embedding_initializer='glorot_uniform',
                     embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                     group_name='default_group',
                     aggregater='mean'):
        """
        Get a single embedded feature using the specified embedding matrix (according to the group name)

        :param slot_name: String. Name of the feature
        :param fixed_embedding_dim:
            Integer. Dimension to which the feature should be embedded, if None, it will be
            embedded to its default embedding dimension
        :param embedding_initializer: Initializer
        :param embedding_regularizer: Regularizer
        :param group_name:
            String. Name of embedding group, each embedding group relate to an independent embedding matrix.
            Note that if fixed_embedding_dim if given, a suffix will be appended to the group name.
            For example, if the group name is "group" and the fixed_embedding_dim is "32" then the real group name
            will become "group_32"
        :return: Tensor. The embedded feature
        """

        assert slot_name in self.all_feats_slots

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
            if slot_name in self.dense_feats_slots:
                group[slot_name] = get_dense_embedded(
                    feat=feat_input,
                    embedding_dim=embedding_dim,
                    embedding_name=embedding_name,
                    embedding_initializer=embedding_initializer,
                    embedding_regularizer=embedding_regularizer
                )
            else:
                group[slot_name] = get_embedded(
                    feat=feat_input,
                    hash=feat_meta.hash,
                    one_hot_dim=feat_meta.one_hot_dim,
                    embedding_dim=embedding_dim,
                    embedding_name=embedding_name,
                    embedding_initializer=embedding_initializer,
                    embedding_regularizer=embedding_regularizer
                )

        ret = group[slot_name]
        if len(ret.shape) == 3:
            if aggregater == 'mean':
                ret = tf.reduce_mean(ret, axis=1, keepdims=False)
            elif aggregater == 'sum':
                ret = tf.reduce_sum(ret, axis=1, keepdims=False)
        return ret

    def get_embedded_dict(self,
                          slots_filter=None,
                          fixed_embedding_dim=None,
                          embedding_initializer='glorot_uniform',
                          embedding_regularizer=tf.keras.regularizers.l2(1e-5),
                          group_name='default_group',
                          aggregater='mean'):
        """
        Get a embedded dict for a list of features using your specified embedding matrix (according to the group name)

        :param slots_filter:
            List of features' names. If None, all features' embedding will be returned
        :param fixed_embedding_dim:
            Integer. Dimension to which the feature should be embedded, if None, it will be
            embedded to its default embedding dimension
        :param embedding_initializer: Initializer
        :param embedding_regularizer: Regularizer
        :param group_name:
            String. Name of embedding group, each embedding group relate to an independent embedding matrix.
            Note that if fixed_embedding_dim if given, a suffix will be appended to the group name.
            For example, if the group name is "group" and the fixed_embedding_dim is "32" then the real group name
            will become "group_32"
        :return: Dict {name: embedded_feature}
        """

        if slots_filter is None:
            slots_filter = self.all_feats_slots

        slots_filter = list(filter(lambda slot_name: slot_name in self.all_feats_slots, slots_filter))

        embedded_dict = dict()

        if fixed_embedding_dim is not None:
            group_name = group_name + '_' + str(fixed_embedding_dim)

        # if the group doesn't exist, create one
        if group_name not in self.embedded_groups:
            self.embedded_groups[group_name] = dict()
        group = self.embedded_groups[group_name]

        # if the slot is not in this group, make a new embedding for it.
        for slot_name in slots_filter:
            if slot_name not in group:
                feat_input = self.inputs_dict[slot_name]
                feat_meta = self.meta_dict[slot_name]
                embedding_dim = fixed_embedding_dim if fixed_embedding_dim is not None else feat_meta.embedding_dim
                embedding_name = \
                    group_name + '/' + feat_meta.name + '_d' + str(feat_meta.one_hot_dim) + '_to_d' + str(embedding_dim)
                if slot_name in self.dense_feats_slots:
                    group[slot_name] = get_dense_embedded(
                        feat=feat_input,
                        embedding_dim=embedding_dim,
                        embedding_name=embedding_name,
                        embedding_initializer=embedding_initializer,
                        embedding_regularizer=embedding_regularizer
                    )
                else:
                    group[slot_name] = get_embedded(
                        feat=feat_input,
                        hash=feat_meta.hash,
                        one_hot_dim=feat_meta.one_hot_dim,
                        embedding_dim=embedding_dim,
                        embedding_name=embedding_name,
                        embedding_initializer=embedding_initializer,
                        embedding_regularizer=embedding_regularizer
                    )
            embedded_dict[slot_name] = group[slot_name]
            if len(embedded_dict[slot_name].shape) == 3:
                if aggregater == 'mean':
                    embedded_dict[slot_name] = tf.reduce_mean(embedded_dict[slot_name], axis=1, keepdims=False)
                elif aggregater == 'sum':
                    embedded_dict[slot_name] = tf.reduce_sum(embedded_dict[slot_name], axis=1, keepdims=False)

        return embedded_dict
