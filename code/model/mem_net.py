#encoding: utf-8
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Flatten, Reshape, Permute, Dropout, Masking, Add, dot, concatenate, Lambda, Layer, Multiply, multiply
from keras.layers import LSTM, GRU, Conv1D, Conv2D
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
sess = tf.Session(config=config)
K.set_session(sess)

class MeMConfig(object):
    """memory network配置参数"""

    embedding_dim = 64      # 词向量维度
    seq_length = 35        # 序列长度
    image_extract_mode = 'inceptionv3'
    num_classes = 4        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸
    vocab_size = 10000       # 词汇表


    mem_size = 2048         #memory size
    n_hop = 1               #hops of deep memory network

    hidden_dim = 32        # GRU层神经元数量

    dropout_keep_prob = 0.5 # dropout
    dropout_embedding_prob = 0.25 #embedding layer之后的drop
    learning_rate = 1e-4   # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 100        # 总迭代轮次

    pretrain = True          #预训练的词向量


    if image_extract_mode == 'inceptionv3':
        image_feature_dim = (3, 3, 2048)
    elif image_extract_mode == 'vgg16':
        image_feature_dim = (4, 4, 512)
    else:
        image_feature_dim = (14, 14, 512)

# class ExpandLayer(Layer):
#
#     def __init__(self, **kwargs):
#         #self.output_dim = output_dim
#         super(ExpandLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(ExpandLayer, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x):
#         return K.expand_dims(x, -1)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)
#
# ExpandLayer = ExpandLayer()
#
# class SumLayer(Layer):
#
#     def __init__(self, **kwargs):
#         #self.output_dim = output_dim
#         super(SumLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         # Create a trainable weight variable for this layer.
#         self.kernel = self.add_weight(name='kernel',
#                                       shape=(input_shape[1], self.output_dim),
#                                       initializer='uniform',
#                                       trainable=True)
#         super(SumLayer, self).build(input_shape)  # Be sure to call this somewhere!
#
#     def call(self, x):
#         return K.sum(x, 2)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.output_dim)



class MemNet(object):
    """deep memory network on multimodal sentiment classification task"""
    def __init__(self, config):
        self.config = MeMConfig()
        self.model()

    def _image_encoding(self, image_feature, shared_linear):
        # image_input_encoder = []
        # for index in range(image_feature.shape[1]):
        #     memory = shared_linear(image_feature[:, index, :])
        #     image_input_encoder.append(memory)
        # tf.transpose(tf.convert_to_tensor(image_input_encoder), [1, 0, 2])
        image_input_encoder = TimeDistributed(shared_linear)(image_feature)
        return image_input_encoder


    def _text_encoding(self, sequence_input):
        text_embedding_layer_a = Embedding(input_dim=self.config.vocab_size,
                                           output_dim=self.config.embedding_dim,
                                           input_length=self.config.seq_length,
                                           trainable=True)

        sequence_input_encoder = text_embedding_layer_a(sequence_input)
        sequence_input_encoder = Dropout(self.config.dropout_embedding_prob)(sequence_input_encoder)
        # output: (batch_size, seq_length, embedding_dim)

        sequence_input_encoder = Lambda(self._reduce_sum, name='sequence_embed')(sequence_input_encoder)#self.reduce_sum_layer(sequence_input_encoder) #tf.reduce_sum(sequence_input_encoder, 2)
        # output: (batch_size, seq_length)
        return sequence_input_encoder

    def _embedding(self):
        """text embedding, image region feature representation and image feature embeding"""
        self.sequence_input = Input((self.config.seq_length,), name='sequence_input')
        self.image_input = Input(self.config.image_feature_dim, name='image_feature_input')

        #text(sequence)embedding for input
        sequence_encoder = self._text_encoding(self.sequence_input)

        image_input_feature = Reshape((self.config.image_feature_dim[2], self.config.image_feature_dim[0] * self.config.image_feature_dim[1]))(self.image_input)
        # output: (batch_size, mem_size(512), 3 * 3)

        return sequence_encoder, image_input_feature



    def _build_vars(self):
        self.C = []
        for hopn in range(self.config.n_hop):
            self.C.append(Dense(units=self.config.seq_length, name= 'hop_' + str(hopn) + '_shared_linear_layer'))



    def _inference(self):
        self._build_vars()
        sequence_encoder, image_input_feature = self._embedding()

        sequence_encoder = [sequence_encoder]


        for hnop in range(self.config.n_hop):
            if hnop == 1:
                memory_encoder = self._image_encoding(image_input_feature, Dense(units=self.config.seq_length, name='hop_' + str(hnop) + '_shared_linear_layer'))
            else:
                memory_encoder = self._image_encoding(image_input_feature, self.C[hnop - 1])

            #get inner product of text embedding and image feature embedding
            sequence_encoder_temp = Permute((2, 1), name='sequence_encoder_temp')(Lambda(self._expand_dim)(sequence_encoder[-1]))#(self.expand_dim_layer(sequence_encoder[-1]))#tf.transpose(tf.expand_dims(sequence_encoder[-1], -1), [0, 2, 1])
            # output: (batch_size, 1, seq_length)


            dotted = Lambda(self._reduce_sum, name='dotted')(memory_encoder * sequence_encoder_temp)#self.reduce_sum_layer(memory_encoder * sequence_encoder_temp)#tf.reduce_sum(memory_encoder  * sequence_encoder_temp, 2)
            # outputs: (batch_size, mem_size)
            self.dotted = Lambda(lambda x: x[:, 0, :])(dot([sequence_encoder_temp, memory_encoder], axes=(2, 2), name='match'))
            #softmax weights
            probs = Activation(activation='softmax')(self.dotted)#tf.nn.softmax(dotted)
            # outputs: (batch_size, mem_size)
            probs_temp = Permute((2, 1), name='probs_temp')(Lambda(self._expand_dim)(probs))#(self.expand_dim_layer(probs))#tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
            # outputs: (batch_size, 1, mem_size)

            memory_output_encoder = self._image_encoding(image_input_feature, self.C[hnop])

            memory_output_encoder_temp = Permute((2, 1), name='memory_output_encoder_temp')(memory_output_encoder)


            o_k = Lambda(self._reduce_sum, name='o_k')(multiply([memory_output_encoder_temp, probs_temp]))#(memory_output_encoder_temp * probs_temp)#self.reduce_sum_layer(memory_output_encoder_temp * probs_temp)#tf.reduce_sum(memory_output_encoder_temp * probs_temp, 2)

            u_k = concatenate([o_k, sequence_encoder[-1]])

            #u_k = GRU(self.config.hidden_dim)(u_k)
            u_k= Dense(self.config.hidden_dim)(u_k)

            sequence_encoder.append(u_k)

        return u_k#tf.matmul(u_k, tf.transpose(self.C[-1], [1, 0]))

    def model(self):
        #self._build_vars()
        self.labels = Input((self.config.num_classes,), name='label')
        self.logits = self._inference()
        self.preds = Dense(self.config.num_classes, activation='softmax')(self.logits)
        print(self.preds)
        model = Model(inputs=[self.sequence_input, self.image_input], outputs=self.preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])
        print(model.summary())
        # self.loss = tf.reduce_mean(categorical_crossentropy(self.labels, self.preds))
        # self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss)

    def _expand_dim(self, x):
        return K.expand_dims(x, -1)

    def _reduce_sum(self, x):
        return K.sum(x, 2)

    def train(self):
        pass


    def test(self):
        pass

if __name__ == '__main__':
    config = MeMConfig()
    net = MemNet(config)
    net.train()