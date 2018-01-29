import os
import requests
import ast
import re
import time
import shutil
import numpy as np
import pandas as pd
from PIL import Image

a = 'News Reality-TV Talk-Show Documentary Biography'.split(' ')
b = 'Crime War Horror Film-Noir Thriller Mystery Action Adventure'.split(' ')
c = 'Animation Music Musical Comedy Romance Fantasy Family'.split(' ')
genre_kinds = 3
genre_list = a + b + c
genre_a, genre_b, genre_c = a, b, c

print(genre_a)

def get_label(genre):
    count_genre = {}.fromkeys([0, 1, 2], 0)
    for item in genre:
        item = item.strip()
        if item in genre_a:
            count_genre[1] += 1
        elif item in genre_b:
            count_genre[0] += 1
        elif item in genre_c :
            count_genre[2] += 1
    label_g = max(count_genre.items(), key=lambda x: x[1])[0]
    return label_g


def get_file_lines(textpath='../../imdb/imdb_uniq.txt'):
    myfile = open(textpath, 'r')
    lines = len(myfile.readlines())
    return lines


def prepare(textpath='../../imdb/imdb_uniq.txt', save_dir='../../imdb/valid_data_path'):
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)
    cnt = get_file_lines(textpath=textpath)
    category_a_list = []
    category_b_list = []
    category_c_list = []
    train_file = os.path.join(save_dir, 'multimodal.train.txt')
    test_file = os.path.join(save_dir, 'multimodal.test.txt')
    val_file = os.path.join(save_dir, 'multimodal.val.txt')

    with open(textpath, 'r') as f:
        for line in f:
            #count_genre = {}.fromkeys([0, 1, 2], 0)
            movie = ast.literal_eval(line)

            id = re.match(r'./img/(\d+).jpg',movie['Poster']).group(1)
            plot = movie['Plot'].strip()
            img_path = '../../imdb' + movie['Poster'].strip()[1:]
            genre = movie['Genre']
            rating = movie['imdbRating']

            if plot == 'N/A' or genre == 'N/A':
                continue

            genre = genre.split(',')
            label_g = get_label(genre)
            if label_g == 0:
                category_b_list.append([id, 'negative', plot, img_path, rating])
            elif label_g == 1:
                category_a_list.append([id, 'neural', plot, img_path, rating])
            else:
                category_c_list.append([id, 'positive', plot, img_path, rating])

    category_a_list = sorted(category_a_list, key=lambda item: item[4], reverse=True)
    category_b_list = sorted(category_b_list, key=lambda item: item[4], reverse=True)
    category_c_list = sorted(category_c_list, key=lambda item: item[4], reverse=True)
    print(category_a_list[0])
    write_file(category_a_list, train_file, test_file, val_file)
    write_file(category_b_list, train_file, test_file, val_file)
    write_file(category_c_list, train_file, test_file, val_file)


def write_file(category_list, train_file, test_file, val_file):
    for i in range(len(category_list)):
        if i <= (int)(0.1 * len(category_list)):
            with open(test_file, 'a') as f:
                f.write(category_list[i][0] + '\t'
                        + category_list[i][1] + '\t'
                        + category_list[i][2].replace('\t', ' ') + '\t'
                        + category_list[i][3] + '\t'
                        + category_list[i][4] + '\n')

        elif i <= (int)(0.2 * len(category_list)):
            with open(val_file, 'a') as f:
                f.write(category_list[i][0] + '\t'
                        + category_list[i][1] + '\t'
                        + category_list[i][2].replace('\t', ' ') + '\t'
                        + category_list[i][3] + '\t'
                        + category_list[i][4] + '\n')
        else:
            with open(train_file, 'a') as f:
                f.write(category_list[i][0] + '\t'
                        + category_list[i][1] + '\t'
                        + category_list[i][2].replace('\t', ' ') + '\t'
                        + category_list[i][3] + '\t'
                        + category_list[i][4] + '\n')




if __name__ == '__main__':
    prepare()