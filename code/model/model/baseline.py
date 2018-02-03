#encoding: utf-8
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras import initializers
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Flatten, Reshape, RepeatVector, Permute, Dropout, Masking, Add, dot, concatenate, Lambda, Layer, Multiply, multiply, BatchNormalization
from keras.layers import LSTM, GRU, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.regularizers import l1, l2, l1_l2
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


class BaselineConfig(object):
    """baseline 配置参数"""

    embedding_dim = 200  # 词向量维度
    seq_length = 50  # 序列长度
    image_extract_mode = 'vgg16'
    num_classes = 3  # 类别数

    num_filters = 128  # 卷积核数目
    filter_sizes = [3, 4, 5]  # 卷积核尺寸
    filter_num_total = num_filters * len(filter_sizes)
    vocab_size = 10000  # 词汇表

    mem_size = 512  # memory size
    n_hop = 1  # hops of deep memory network

    hidden_rnn_dim = 50
    hidden_dim = 35  # 隐层神经元数量

    dropout_keep_prob = 0.5  # dropout of dense layer
    dropout_embedding_prob = 0.35  # embedding layer之后的drop
    rate_drop_gru = 0.15 + np.random.rand() * 0.25
    rate_drop_dense = 0.15 + np.random.rand() * 0.25

    learning_rate = 1e-2  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    embedding_paras = None  # 预训练的词向量

    if image_extract_mode == 'inceptionv3':
        image_feature_dim = (3, 3, 2048)
    elif image_extract_mode == 'vgg16':
        image_feature_dim = (4, 4, 512)
    else:
        image_feature_dim = (14, 14, 512)


    if image_extract_mode == 'inceptionv3':
        image_feature_dim = (3, 3, 2048)
    elif image_extract_mode == 'vgg16':
        image_feature_dim = (4, 4, 512)
    else:
        image_feature_dim = (14, 14, 512)



class Baseline(object):
    def __init__(self, config):
        self.config = config
        self.model()

    def _image_encoding(self, image_feature):
        image_input_encoder = Dense(units=self.config.hidden_rnn_dim, activity_regularizer=l2(l=0.01), name= 'img_dense')(image_feature)
        return image_feature

    def _text_encoding(self, sequence_input):

        self.embedding_matrix = np.zeros((self.config.vocab_size, self.config.embedding_dim))

        if self.config.embedding_paras != None:
            embeddings_index = self.config.embedding_paras[0]
            word_index = self.config.embedding_paras[1]
            for word, i in word_index.items():
                if i >= self.config.vocab_size:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    self.embedding_matrix[i] = embedding_vector

        text_embedding_layer_a = Embedding(input_dim=self.config.vocab_size,
                                           output_dim=self.config.embedding_dim,
                                           input_length=self.config.seq_length,
                                           weights=[self.embedding_matrix],
                                           #embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(l=0.01),
                                           trainable=True)

        sequence_input_encoder = text_embedding_layer_a(sequence_input)
        if self.config.is_train == True:
            sequence_input_encoder = Dropout(self.config.dropout_embedding_prob)(sequence_input_encoder)
        # output: (batch_size, seq_length, embedding_dim)

        #GRU
        sequence_input_encoder_gru = GRU(self.config.hidden_rnn_dim,
                                         dropout=self.config.rate_dropout_dense,
                                         recurrent_dropout=self.config.rate_dropout_gru)(sequence_input_encoder)

        #CNN
        self.kernels = []
        output = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            print('------ ', i, ' -------')
            #sequence_input_encoder = Lambda(self._expand_dim)(sequence_input_encoder)
            print(sequence_input_encoder.shape)
            conv = Conv1D(self.config.num_filters,
                       filter_size,
                       activation='relu',
                       kernel_initializer=initializers.TruncatedNormal(stddev=0.1),
                       bias_initializer=initializers.Constant(value=0.1),
                       name='text-cnn-layer-'+str(i+1))(sequence_input_encoder)
            print(conv.shape)
            pooled = MaxPooling1D(pool_size=self.config.seq_length-filter_size+1,
                                strides=2,
                                padding='valid',
                                name='text-pool-layer-'+str(i+1))(conv)
            print(pooled.shape)
            output.append(pooled)

        pooled_reshape = Flatten()(concatenate(output, axis=1))
        sequence_input_encoder_cnn = Dropout(self.config.dropout_keep_prob)(pooled_reshape)

        #Average
        #sequence_input_encoder = Lambda(self._reduce_mean, name='sequence_embed')(sequence_input_encoder)#self.reduce_sum_layer(sequence_input_encoder) #tf.reduce_sum(sequence_input_encoder, 2)
        # output: (batch_size, seq_length)
        #sequence_input_encoder = BatchNormalization()(sequence_input_encoder)

        return sequence_input_encoder_gru

    def _embedding(self):
        """text embedding, image region feature representation and image feature embeding"""
        self.sequence_input = Input((self.config.seq_length,), name='sequence_input')
        self.image_input = Input(self.config.image_feature_dim, name='image_feature_input')

        #text(sequence)embedding for input
        sequence_encoder = self._text_encoding(self.sequence_input)

        image_input_feature = Flatten()(self.image_input)

        return sequence_encoder, image_input_feature

    def model(self):
        sequence_encoder, image_input_feature = self._embedding()
        image_encoder = self._image_encoding(image_input_feature)
        self.logits = concatenate(sequence_encoder, image_encoder)

        self.preds = Dense(self.config.num_classes, activation='softmax')(self.logits)
        self.model = Model(inputs=[self.sequence_input, self.image_input], outputs=self.preds)

        self.optimizer = optimizers.Adam(self.config.learning_rate)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['acc'], )  # additional_metrics.recall, additional_metrics.f1])

        print(self.model.summary())

