#Used to load the idx3_ubyte data format
import idx2numpy
#Used as the data object for images and labes
import numpy as np



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