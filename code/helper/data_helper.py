import os
import sys
import re
import tensorflow.contrib.keras as kr
import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

MAX_NB_WORDS = 10000

base_dir = '../'
glove_dir = os.path.join(base_dir, 'glove.6B')

wordnet_lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    res = []
    text = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", text)
    for word in text:
        res.append(wordnet_lemmatizer.lemmatize(word))
    return res

def get_embeddings_index():
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    return embeddings_index

def open_file(filename, mode='r'):
    """
    Commonly used file reader, change this to switch between python2 and python3.
    mode: 'r' or 'w' for read or write
    """
    return open(filename, mode, encoding='utf-8', errors='ignore')

def read_file(filename):
    """读取文件数据"""
    contents, labels, image_path = [], [],[]
    cnt = 0

    with open_file(filename) as f:
        cnt = 0
        for line in f:
            cnt += 1
            try:
                label, content, path = line.strip().split('\t')[1:4]
                content = text_to_word_sequence(content)#lemmatize(content)
                #print(content)
                contents.append(content)#(list(content))
                labels.append(label)
                image_path.append(path)
            except:
               pass

    return contents, labels, image_path

def build_vocab(train_dir, vocab_dir, vocab_size=10000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)[:2]

    all_data = []
    for content in data_train:
        #print(content)
        all_data.extend(content)

    #print(all_data)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)

    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')

def read_vocab(vocab_dir):
    """读取词汇表"""
    words = open_file(vocab_dir).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id

def read_category():
    """读取分类目录，固定"""
    categories = ['creepy', 'gore', 'happy', 'rage']
    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id

def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)

def process_file(filename, word_to_id, cat_to_id, max_length=100):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)[:2]

    data_id, label_id = [], []

    # tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    # tokenizer.fit_on_texts(contents)
    # data_id = tokenizer.texts_to_sequences(contents)

    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id)  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]



# if __name__ == '__main__':
#     base_dir = '../data/title'
#     categories, cat_to_id = read_category()
#     print(categories, cat_to_id)
#     #x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, 25)
#     build_vocab(train_dir='../../reddit/text/text.train.txt', vocab_dir='../../reddit/text/text.vocab.txt', vocab_size=10000)
#     # process_file('../../reddit/text/text.train.txt')
#     # print(content[:5])
