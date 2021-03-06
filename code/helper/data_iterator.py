import os
import sys
import cv2
import numpy as np
import pandas as pd

from helper import data_helper

image_width = 150
image_height = 150

#base_dir = '../reddit/text'
#vocab_dir = os.path.join(base_dir, 'text.vocab.txt')

# if os.path.exists(vocab_dir) == False:
#     data_helper.build_

class DataIterator:
    def __init__(self, dataset, datapath, image_feature_path):
        base_dir = '../' + dataset + '/valid_data_path'
        vocab_dir = os.path.join(base_dir, 'text.vocab.txt')
        self.categories, self.cat_to_id = data_helper.read_category(dataset)
        self.words, self.word_to_id = data_helper.read_vocab(vocab_dir)
        
        self.text, self.label = data_helper.process_file(datapath, self.word_to_id, self.cat_to_id)
        self.image_path_list = data_helper.read_file(datapath)[2]
        self.image = np.load(image_feature_path)[:, 0, :, :]

        # for root, sub_folder, file_list in os.walk(data_dir):
        #     for file_path in file_list:
        #         image_name = os.path.join(root,file_path)
        #         self.image_names.append(image_name)
        #         im = cv2.imread(image_name,0).astype(np.float32)/255.
        #         im = cv2.resize(im,(image_width,image_height))
        #         im = im.swapaxes(0,1)
        #         self.image.append(np.array(im))
        #         self.labels.append(code)


    # def input_index_generate_batch(self,index=None):
    #     if index:
    #         image_batch=[self.image[i] for i in index]
    #         label_batch=[self.labels[i] for i in index]
    #     else:
    #         # get the whole data as input
    #         image_batch=self.image
    #         label_batch=self.labels
    #
    #     def get_input_lens(sequences):
    #         lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    #         return sequences,lengths
    #
    #     batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))
    #
    #
    #     return batch_inputs,batch_seq_len,label_batch




if __name__ == '__main__':
    pass
