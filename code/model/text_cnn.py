import sys
from gensim.models import KeyedVectors
from keras.callbacks import *
from keras import initializers
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import *
from keras.models import *
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import *
from keras.utils.np_utils import *

sys.path.append('../')
from helper.data_helper import *
# import helper.dataHelper

base_dir = '../reddit/text'
vocab_dir = os.path.join(base_dir, 'text.vocab.txt')

class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 200      # 词向量维度
    seq_length = 30        # 序列长度
    num_classes = 3        # 类别数
    num_filters = 128        # 卷积核数目
    kernel_size = 3         # 卷积核尺寸
    vocab_size = 10000       # 词汇表

    hidden_dim = 128        # 全连接层神经元

    dropout_keep_prob = 0.5 # dropout保留比例
    learning_rate = 1e-3    # 学习率

    batch_size = 64         # 每批训练大小
    num_epochs = 100        # 总迭代轮次

    print_per_batch = 100    # 每多少轮输出一次结果
    save_per_batch = 10      # 每多少轮存入tensorboard

    pretrain = True



class TextCNN(object):
    def __init__(self, config):
         self.config = config
         self.embeddings_index = get_embeddings_index()
         self.word_index = read_vocab(vocab_dir)[1]
         self.pretrain = self.config.pretrain
         self.nb_words = self.config.vocab_size#min(self.config.vocab_size, len(self.word_index)) + 1
         self.model = self.model()

    def model(self):

        embedding_matrix = np.zeros((self.nb_words, self.config.embedding_dim))

        if self.pretrain == True:
            # for word, i in self.word_index.items():
            #     # if company_trie.find(word) == True:  #dictionary of the whole key name
            #     # 	embedding_matrix[i] = np.array([0.01] * 300)
            #     if word in self.word2vec:
            #         embedding_matrix[i] = self.word2vec[word]
            for word, i in self.word_index.items():
                #print(word, '\t', i)
                if i >= self.config.vocab_size:
                    continue
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                    #print(embedding_matrix[i])



        print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
        embedding_layer = Embedding(self.nb_words,
                                    self.config.embedding_dim,
                                    weights=[embedding_matrix],
                                    input_length=self.config.seq_length,
                                    trainable=False)

        sequence_input = Input((self.config.seq_length,))
        embedded_sequences = embedding_layer(sequence_input)

        x = Conv1D(self.config.num_filters,
                   self.config.kernel_size,
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros')(embedded_sequences)
        x = MaxPool1D(2)(x)

        x = Conv1D(self.config.num_filters,
                   self.config.kernel_size,
                   kernel_initializer='random_uniform',
                   bias_initializer='zeros')(x)
        x = MaxPool1D(2)(x)

        # x = Conv1D(self.config.num_filters, self.config.kernel_size, kernel_initializer='random_uniform', bias_initializer='zeros')(x)
        # x = MaxPool1D(2)(x)

        x = GlobalMaxPooling1D()(x)

        x = Dense(self.config.hidden_dim,
                  activation='relu',
                  kernel_initializer=initializers.random_normal(stddev=0.01))(x)

        preds = Dense(self.config.num_classes,
                      activation='softmax')(x)

        model = Model(inputs=sequence_input,
                      outputs=preds)

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        #print(model.summary())
        return model

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TCNNConfig()
    model = TextCNN(config).model
