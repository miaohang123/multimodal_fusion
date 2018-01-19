import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from sklearn import metrics
from datetime import timedelta

from keras.callbacks import EarlyStopping

from model.text_cnn import TCNNConfig, TextCNN
from helper.data_helper import *

base_dir = '../reddit/valid_data_path'
train_dir = os.path.join(base_dir, 'multimodal.train.txt')
test_dir = os.path.join(base_dir, 'multimodal.test.txt')
val_dir = os.path.join(base_dir, 'multimodal.val.txt')
vocab_dir = os.path.join(base_dir, 'text.vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')   # 最佳验证结果保存路径

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# def evaluate(sess, x_, y_):
#     """评估在某一数据上的准确率和损失"""
#     data_len = len(x_)
#     batch_eval = batch_iter(x_, y_, 128)
#     total_loss = 0.0
#     total_acc = 0.0
#     for x_batch, y_batch in batch_eval:
#         batch_len = len(x_batch)
#         feed_dict = feed_data(x_batch, y_batch, 1.0)
#         loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
#         total_loss += loss * batch_len
#         total_acc += acc * batch_len
#
#     return total_loss / data_len, total_acc / data_len
#
#
# def test():
#     print("Loading test data...")
#     start_time = time.time()
#     x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)
#
#     print('Testing...')
#     loss_test, acc_test = evaluate(session, x_test, y_test)
#     msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
#     print(msg.format(loss_test, acc_test))
#
#     batch_size = 128
#     data_len = len(x_test)
#     num_batch = int((data_len - 1) / batch_size) + 1
#
#     y_test_cls = np.argmax(y_test, 1)
#     y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32) # 保存预测结果
#     for i in range(num_batch):   # 逐批次处理
#         start_id = i * batch_size
#         end_id = min((i + 1) * batch_size, data_len)
#         feed_dict = {
#             model.input_x: x_test[start_id:end_id],
#             model.keep_prob: 1.0
#         }
#         y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
#
#     # 评估
#     print("Precision, Recall and F1-Score...")
#     print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))
#
#     # 混淆矩阵
#     print("Confusion Matrix...")
#     cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
#     print(cm)
#
#     time_dif = get_time_dif(start_time)
#     print("Time usage:", time_dif)


class Text(object):
    def __init__(self, model):
        self.model = model
        self.train()
        self.save()

    def train(self):
        # print("Configuring TensorBoard and Saver...")
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        print("Loading training and validation data...")
        # 载入训练集与验证集
        start_time = time.time()
        x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
        x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
        print("train data size: ", len(x_train))
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        print('Training and evaluating...')
        start_time = time.time()
        # total_batch = 0              # 总批次
        # best_acc_val = 0.0           # 最佳验证集准确率
        # last_improved = 0            # 记录上一次提升批次
        # require_improvement = 1000   # 如果超过1000轮未提升，提前结束训练
        earystop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
        self.model.fit(x_train, y_train,
                  batch_size=128,
                  epochs=200,
                  validation_data=(x_val, y_val),
                  callbacks=[earystop])

    def save(self):
        if os.path.exists('weights') == False:
            os.mkdir('weights')
        if os.path.exists('model') == False:
            os.mkdir('model')
        json_string = model.to_json()

        json.dump(json_string, open('model/textCNN.json', 'a'))  # the json file of the model
        self.model.save_weights('weights/textCNN_weights.h5')  # the weights file of the model

# class Image(object):
#     def __init__(self, model):
#         self.model = model
#
#     def train(self):
#         pass
#
#     def save(self):
#         pass



if __name__ == '__main__':
    # if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
    #     raise ValueError("""usage: python run_cnn.py [train / test]""")

    #text
    print('Configuring CNN model...')
    config = TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    # print(categories)
    config.vocab_size = len(words)
    model = TextCNN(config).model

    text_pipe_line = Text(model)

    #image


    #multimodal
