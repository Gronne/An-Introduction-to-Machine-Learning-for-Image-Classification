#------------------------- Imports --------------------------
#Used for getting the MNIST and ORL dataset
from DataManipulation.DataLoaders import Datasets
#Used for tranforming data
from DataManipulation.DataTransformers import DataTransformers
#Used for visualising data
from DataManipulation.DataVisualisers import DataVisualiser
#Used to access the different algorithms
from Algorithms import NearestClassifiers, Perceptrons
#Used for getting the current directory (Main)
import pathlib
#Used as the dataobject for data
import numpy



#------------------------ Load Data -------------------------
#Get folder that Main is in
base_path = str(pathlib.Path(__file__).parent.absolute()) + '/'

#Load MNIST dataset
mnist_path = "../Data/MNIST/"
mnist_img_train, mnist_lbl_train, mnist_img_test, mnist_lbl_test = Datasets.load_mnist(base_path, mnist_path)

#Load ORL dataset
orl_path = "../Data/ORL_txt/"
orl_img, orl_lbl = Datasets.load_orl(base_path, orl_path)



#---------------------- Transform Data ----------------------
#Transform MNIST data
mnist_img_train_PCA = DataTransformers.PCA.tranform(mnist_img_train, 2)
mnist_img_test_PCA = DataTransformers.PCA.tranform(mnist_img_test, 2)

#Transform ORL Data
orl_img_PCA = DataTransformers.PCA.tranform(orl_img, 2)


# #---------------------- Visualize Data ----------------------
# #------ Images ------
# #MNIST Train
# DataVisualiser.plotImages(mnist_img_train[0:11], mnist_lbl_train[0:11])
# #MNIST Test
# DataVisualiser.plotImages(mnist_img_test[0:9], mnist_lbl_test[0:9])
# #ORL
# DataVisualiser.plotImages(orl_img[0:9], orl_lbl[0:9])

# #------ 2D Plot -----
# #MNIST Train
# DataVisualiser.plot2dData(mnist_img_train_PCA, labels=mnist_lbl_train)
# #MNIST Test
# DataVisualiser.plot2dData(mnist_img_test_PCA, labels=mnist_lbl_test)
# #ORL
# DataVisualiser.plot2dData(orl_img_PCA, labels=orl_lbl)





# #------------------------ Exercises ------------------------
# #Exercise 1 - Nearest Class centroid classifier
# #--------MNIST--------
# #Full dimensionality
# model_MNIST = NearestClassifiers.ClassCentroid.train(mnist_img_train, mnist_lbl_train)
# accuracy_MNIST = NearestClassifiers.ClassCentroid.test(model_MNIST, mnist_img_test, mnist_lbl_test)
# print(f"NCCC - MNIST Full Accuracy: {accuracy_MNIST}")
# #PCA applied
# model_MNIST_PCA = NearestClassifiers.ClassCentroid.train(mnist_img_train_PCA, mnist_lbl_train)
# accuracy_MNIST_PCA = NearestClassifiers.ClassCentroid.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test)
# print(f"NCCC - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

# #---------ORL---------
# #Full dimensionality
# model_ORL = NearestClassifiers.ClassCentroid.train(orl_img, orl_lbl)
# accuracy_ORL = NearestClassifiers.ClassCentroid.test(model_ORL, orl_img, orl_lbl)
# print(f"NCCC - ORL Full Accuracy: {accuracy_ORL}")
# #PCA applied
# model_ORL_PCA = NearestClassifiers.ClassCentroid.train(orl_img_PCA, orl_lbl)
# accuracy_ORL_PCA = NearestClassifiers.ClassCentroid.test(model_ORL_PCA, orl_img_PCA, orl_lbl)
# print(f"NCCC - ORL PCA Accuracy: {accuracy_ORL_PCA}")




# #Exercise 2 - Nearest sub-class centroid classifier using number of subclasses in set {2, 3, 5}
# sets = [2, 3, 5]
# for nr_sub_classes in sets:
#     properties = {'nr_clusters': nr_sub_classes}
#     #--------MNIST--------
#     #Full dimensionality
#     model_MNIST = NearestClassifiers.SubClassCentroid.train(mnist_img_train, mnist_lbl_train, properties)
#     accuracy_MNIST = NearestClassifiers.SubClassCentroid.test(model_MNIST, mnist_img_test, mnist_lbl_test)
#     print(f"NSCCC({nr_sub_classes}) - MNIST Full Accuracy: {accuracy_MNIST}")
#     #PCA applied
#     model_MNIST_PCA = NearestClassifiers.SubClassCentroid.train(mnist_img_train_PCA, mnist_lbl_train, properties)
#     accuracy_MNIST_PCA = NearestClassifiers.SubClassCentroid.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test)
#     print(f"NSCCC({nr_sub_classes}) - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

#     #---------ORL---------
#     #Full dimensionality
#     model_ORL = NearestClassifiers.SubClassCentroid.train(orl_img, orl_lbl, properties)
#     accuracy_ORL = NearestClassifiers.SubClassCentroid.test(model_ORL, orl_img, orl_lbl)
#     print(f"NSCCC({nr_sub_classes}) - ORL Full Accuracy: {accuracy_ORL}")
#     #PCA applied
#     model_ORL_PCA = NearestClassifiers.SubClassCentroid.train(orl_img_PCA, orl_lbl, properties)
#     accuracy_ORL_PCA = NearestClassifiers.SubClassCentroid.test(model_ORL_PCA, orl_img_PCA, orl_lbl)
#     print(f"NSCCC({nr_sub_classes}) - ORL PCA Accuracy: {accuracy_ORL_PCA}")




#Exercise 3 - Nearest Neighbor classifier
properties = {'neighbors': 100, 'density': 0.1} #Hyper parametre: Distance function, neighbors, density, density function, strength function
#--------MNIST--------
#Full dimensionality
model_MNIST = NearestClassifiers.Neighbor.train(mnist_img_train, mnist_lbl_train, properties)
accuracy_MNIST = NearestClassifiers.Neighbor.test(model_MNIST, mnist_img_test, mnist_lbl_test, properties)
print(f"NNC - MNIST Full Accuracy: {accuracy_MNIST}")
#PCA applied
model_MNIST_PCA = NearestClassifiers.Neighbor.train(mnist_img_train_PCA, mnist_lbl_train, properties)
accuracy_MNIST_PCA = NearestClassifiers.Neighbor.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test, properties)
print(f"NNC - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

#---------ORL---------
properties = {'neighbors': 1, 'density': 1}
#Full dimensionality
model_ORL = NearestClassifiers.Neighbor.train(orl_img, orl_lbl, properties)
accuracy_ORL = NearestClassifiers.Neighbor.test(model_ORL, orl_img, orl_lbl, properties)
print(f"NNC - ORL Full Accuracy: {accuracy_ORL}")
#PCA applied
model_ORL_PCA = NearestClassifiers.Neighbor.train(orl_img_PCA, orl_lbl, properties)
accuracy_ORL_PCA = NearestClassifiers.Neighbor.test(model_ORL_PCA, orl_img_PCA, orl_lbl, properties)
print(f"NNC - ORL PCA Accuracy: {accuracy_ORL_PCA}")





#Exercise 4 - Perceptron trained using Backpropagation




#Exercise 5 - Perceptron trained using MSE (least squares solution)

