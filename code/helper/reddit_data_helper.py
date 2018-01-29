import os
import requests
import re
import time
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from lxml import html
from time import sleep
from bs4 import BeautifulSoup

suffix = ['png', 'jpg', 'jpeg']
categories = ['creepy', 'gore', 'happy', 'rage']

class Spider:
    def __init__(self, base_url_list, id_list, headers, save_dir):
        self.base_url_list = base_url_list
        self.id_list = id_list
        self.headers = headers
        self.save_dir = save_dir
        self.get_content()

    def get_content(self):
        cnt = 1
        exist_list = []
        for item in os.listdir(self.save_dir):
            exist_list.append(item.split('.')[0])
        print(len(exist_list))
        for base_url ,id in zip(self.base_url_list, self.id_list):
            print(id)
            if id in exist_list:
                continue
            suffix_flag = self.judgeSuffix(base_url)
            try:
                r = requests.get(base_url, headers=self.headers)
            except:
                continue
            #if suffix_flag == True:
            if suffix_flag == True:
                try:
                    image = Image.open(BytesIO(r.content))
                    save_path = os.path.join(self.save_dir, id + '.png')
                    print(save_path)
                    image.save(save_path)

                except:
                    print("suffix flag True but failed")

            else:
                try:
                    image = Image.open(BytesIO(r.content))
                    save_path = os.path.join(self.save_dir, id + '.png')
                    print(save_path)
                    image.save(save_path)
                except:
                    sp = BeautifulSoup(r.content)
                    image_url = sp.find('img')['src']
                    if image_url[:2] == "//":
                        image_url = "https:" + image_url
                    print(image_url)
                    try:
                        image = Image.open(BytesIO(requests.get(image_url, headers=self.headers).content))
                        save_path = os.path.join(self.save_dir, id + '.jpg')
                        image.save(save_path)
                    except:
                        print("cnt: ", id, " ", base_url)
            cnt += 1

        print("该文件共: ", len(self.base_url_list), "个要爬的url, 实际爬到的图片数量为: ", cnt)

    def judgeSuffix(self, url):
        if url.split('.')[-1] in suffix:
            return True
        else:
            return False

def arrangeImage(save_dir):
    data_dir = '../../reddit'
    for file in os.listdir(data_dir):
        data_path = os.path.join(data_dir, file)
        if file[0] != 'p':
            continue
        data = pd.read_csv(data_path)
        id_data = data.loc[:, 'id']
        id_list = np.array(id_data).tolist()
        print(id_data.head())
        category = file.split('.')[0].split('_')[1]
        target_path = os.path.join(save_dir, category)
        if os.path.exists(target_path) == False:
            os.mkdir(target_path)

        for image in os.listdir(save_dir):
            print(image.split('.')[0])
            if image.split('.')[0] in id_list:
                shutil.copy(os.path.join(save_dir, image), target_path)

def processText(data_dir, save_dir):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    train_file = os.path.join(save_dir, 'text.train.txt')
    test_file = os.path.join(save_dir, 'text.test.txt')
    val_file = os.path.join(save_dir, 'text.val.txt')

    for data_file in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data_file)
        if data_file[0] != 'p':
            continue
        print(data_path)
        data = pd.read_csv(data_path, sep=',')
        id = data.loc[:, 'id'].tolist()
        title = data.loc[:, 'title'].tolist()
        label = data_file.split('.')[0].split('_')[1]
        for i in range(len(title)):
            if i <= (int)(0.1 * len(title)):
                with open(test_file, 'a') as f:
                    f.write(id[i] + '\t' + label + '\t' + title[i].replace('\t', ' ') + '\n')
            elif i <= (int)(0.2 * len(title)):
                with open(val_file, 'a') as f:
                    f.write(id[i] + '\t' + label + '\t' + title[i].replace('\t', ' ') + '\n')
            else:
                with open(train_file, 'a') as f:
                    f.write(id[i] + '\t' + label + '\t' + title[i].replace('\t', ' ') + '\n')


def processSingleDirImage(image_dir, save_dir, text_path, mode):
    target_image_dir = os.path.join(save_dir, mode)
    if os.path.exists(target_image_dir) == False:
        os.mkdir(target_image_dir)
    id_list = []
    label_list = []
    with open(text_path, 'r') as f:
        for line in f:
            id, label = line.split('\t')[:2]
            id_list.append(id)
            label_list.append(label)
    for label in categories:
        if os.path.exists(os.path.join(target_image_dir, label)) == False:
            os.mkdir(os.path.join(target_image_dir, label))
    miss_cnt = 0
    for id, label in zip(id_list, label_list):
        #print(id, '\t', label)
        tar_img_dir = os.path.join(target_image_dir, label)
        tar_img_path = os.path.join(tar_img_dir, id+'.png')
        #print('tar_img_path: ', tar_img_path)
        origin_img_path = os.path.join(os.path.join(image_dir, label), id+'.png')
        if os.path.exists(origin_img_path) == False:
            origin_img_path = os.path.join(os.path.join(image_dir, label), id + '.jpg')
        if os.path.exists(origin_img_path) == False:
            print(id, '\t', label)
            miss_cnt += 1
            continue
        shutil.copy(origin_img_path, tar_img_dir)

    print('miss ', miss_cnt)

def get_FileSize(filePath):
    #filePath = unicode(filePath, 'utf8')
    fsize = os.path.getsize(filePath)
    #fsize = fsize / float(1024 * 1024)
    return round(fsize, 2)

def filterImage(image_dir):
    for sub_dir in os.listdir(image_dir):
        sub_dir = os.path.join(image_dir, sub_dir)
        if os.path.isdir(sub_dir) == False:
            continue
        for file in os.listdir(sub_dir):
            filepath = os.path.join(sub_dir, file)
            if os.path.isfile(filepath) == False:
                continue
            if get_FileSize(filepath) < 1024:
                print(filepath)
                os.remove(filepath)


def processImage(image_dir, save_dir, text_dir):
    #make train dir
    text_train_path = os.path.join(text_dir, 'text.train.txt')
    processSingleDirImage(image_dir, save_dir, text_path=text_train_path, mode='train')
    #make val dir
    text_val_path = os.path.join(text_dir, 'text.val.txt')
    processSingleDirImage(image_dir, save_dir, text_path=text_val_path, mode='val')
    #make test dir
    text_test_path = os.path.join(text_dir, 'text.test.txt')
    processSingleDirImage(image_dir, save_dir, text_path=text_test_path, mode='test')

def getMultiList(image_dir, text_path, save_path):
    """
    :param image_dir: 'new_image/train'
    :param text_path: 'text/text.train.txt'
    :return:
    """
    id_list = []
    label_list = []
    text_list = []

    with open(text_path, 'r') as f:
        for line in f:
            id_list.append(line.split('\t')[0])
            label_list.append(line.split('\t')[1])
            text_list.append(line.split('\t')[2].strip('\n'))

    miss_cnt = 0

    for id, label, text in zip(id_list, label_list, text_list):
        sub_image_dir = os.path.join(image_dir, label)

        origin_img_path = os.path.join(sub_image_dir, id + '.png')
        if os.path.exists(origin_img_path) == False:
            origin_img_path = os.path.join(os.path.join(image_dir, label), id + '.jpg')
        if os.path.exists(origin_img_path) == False:
            print(id, '\t', label)
            miss_cnt += 1
            continue

        with open(save_path, 'a') as f:
            f.write(id + '\t' + label + '\t' + text + '\t' + origin_img_path + '\n')

if __name__ == '__main__':
    headers = {
        'x-requested-with': 'XMLHttpRequest',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36',
        'cookie': '',
    }
    data_dir = '../../reddit'
    # all_num = 0
    # for data_file in os.listdir(data_dir):
    #     data_path = os.path.join(data_dir, data_file)
    #     if data_file[0] != 'p':
    #         continue
    #     print(data_path)
    #     print("=============")
    #     start =time.time()
    #     data = pd.read_csv(data_path)
    #     id_data = data.loc[:, 'id']
    #     url_data = data.loc[:,'url']
    #
    #     base_url_list = np.array(url_data).tolist()
    #     id_list = np.array(id_data).tolist()
    #
    #     save_dir = os.path.join('../../reddit', 'image')
    #
    #     if os.path.exists(save_dir) == False:
    #         print(save_dir)
    #         os.mkdir(save_dir)
    #
    #     spider = Spider(base_url_list=base_url_list, id_list = id_list, headers=headers, save_dir = save_dir)
    #     all_num += len(id_list)
    #     end = time.time()
    #     print("This file cost: ", (end - start) / 3600.0)

    #print("total data num: ", all_num)

    #arrangeImage(save_dir=os.path.join('../../reddit', 'image'))

    #processText(data_dir='../../reddit', save_dir='../../reddit/text')
    #processImage(image_dir='../../reddit/image', save_dir='../../reddit/new_image', text_dir='../../reddit/text')

    # base_dir = '../../reddit/new_image'
    # for sub_dir in os.listdir(base_dir):
    #     sub_dir = os.path.join(base_dir, sub_dir)
    #     if os.path.isdir(sub_dir) == False:
    #         continue
    #     filterImage(image_dir=sub_dir)
    getMultiList(image_dir='../../reddit/new_image/train', text_path='../../reddit/text/text.train.txt',
                 save_path='../../reddit/valid_data_path/multimodal.train.txt')
    getMultiList(image_dir='../../reddit/new_image/val', text_path='../../reddit/text/text.val.txt',
                 save_path='../../reddit/valid_data_path/multimodal.val.txt')
    getMultiList(image_dir='../../reddit/new_image/test', text_path='../../reddit/text/text.test.txt',
                 save_path='../../reddit/valid_data_path/multimodal.test.txt')





