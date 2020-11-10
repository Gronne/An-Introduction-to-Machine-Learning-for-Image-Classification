#Used to load the idx3_ubyte data format
import idx2numpy
#Used as the data object for images and labes
import numpy as np
#used to split data
import random
import math
from datetime import datetime




class Datasets:
    def load_mnist(base_path, folder_path):
        #Load training data
        mnist_image_train_name = 'train-images'
        mnist_label_train_name = 'train-labels'
        images_train_mnist = Images.Load.idx_as_numpy(base_path, folder_path, mnist_image_train_name)
        labels_train_mnist = Labels.Load.idx_as_numpy(base_path, folder_path, mnist_label_train_name)
        #Load test data
        mnist_image_test_name = 't10k-images'
        mnist_label_test_name = 't10k-labels'
        images_test_mnist = Images.Load.idx_as_numpy(base_path, folder_path, mnist_image_test_name)
        labels_test_mnist = Labels.Load.idx_as_numpy(base_path, folder_path, mnist_label_test_name)
        #Return data
        return images_train_mnist, labels_train_mnist, images_test_mnist, labels_test_mnist

    def load_orl(base_path, folder_path):
        #Load data
        orl_image_name = 'orl_data'
        orl_lable_name = 'orl_lbls'
        images_orl = Images.Load.txt_as_numpy(base_path, folder_path, orl_image_name)
        labels_orl = Labels.Load.txt_as_numpy(base_path, folder_path, orl_lable_name)
        #Return data
        return images_orl, labels_orl

    def split_dataset(data, labels, ratio = 0.5, random_seed = None):
        #Create random seed if nothing is specified
        if not isinstance(random_seed, type(None)): random.seed(random_seed)
        #split data into categories
        categories_data = { category: [] for category in set(labels)}
        for index, label in enumerate(labels): categories_data[label].append(data[index])
        #Scramble categories
        categories_data = Datasets._scramble_data(categories_data)
        #Split categories an recombine into train and test list
        train_list, test_list = Datasets._combine_cats_with_ratio(categories_data, ratio)
        #split lists into data and label lists again
        train_data_list, train_label_list = Datasets._split_into_data_label_lists(train_list)
        test_data_list, test_label_list = Datasets._split_into_data_label_lists(test_list)
        #Return data
        return np.array(train_data_list), np.array(train_label_list), np.array(test_data_list), np.array(test_label_list)
        
    def _scramble_data(categories_data):
        for category in categories_data: 
            categories_data[category] = sorted(categories_data[category], key = lambda x: random.random())
        return categories_data

    def _combine_cats_with_ratio(categories_data, ratio):
        train_list = []
        test_list = []
        for category in categories_data:
            #Find index to split at
            split_index = math.ceil(len(categories_data[category])*ratio)
            #Get training part
            train_data = categories_data[category][:split_index]
            train_list.append([{'class': category, 'data': data} for data in train_data])
            #Get test part
            test_data = categories_data[category][split_index:]
            test_list.append([{'class': category, 'data': data} for data in test_data])
        #Collaps list of lists, so every category is combined
        train_list = [data_point for cat_list in train_list for data_point in cat_list]
        test_list = [data_point for cat_list in test_list for data_point in cat_list]
        return train_list, test_list
    
    def _split_into_data_label_lists(data_cat_list):
        data_list = [data_cat['data'] for data_cat in data_cat_list]
        label_list = [data_cat['class'] for data_cat in data_cat_list]
        return data_list, label_list
        





class Images:
    class Load:
        def idx_as_numpy(base_path, path = "", name = ""):
            data_type = '.idx3-ubyte'
            full_path = base_path + path + name + data_type
            images = idx2numpy.convert_from_file(full_path)
            return images

        def txt_as_numpy(base_path, path = "", name = ""):
            data_type = '.txt'
            full_path = base_path + path + name + data_type
            images = np.loadtxt(full_path)
            images = images.reshape(30, 40, 400)
            images = images.transpose()
            return images
        



class Labels:
    class Load:
        def idx_as_numpy(base_path, path = "", name = ""):
            data_type = '.idx1-ubyte'
            full_path = base_path + path + name + data_type
            lables = idx2numpy.convert_from_file(full_path)
            return lables

        def txt_as_numpy(base_path, path = "", name = ""):
            data_type = '.txt'
            full_path = base_path + path + name + data_type
            labels = np.loadtxt(full_path)
            return labels