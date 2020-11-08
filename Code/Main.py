#------------------------- Imports --------------------------
#Used for getting the MNIST and ORL dataset
from HelpFunctions.DataLoaders import Datasets
#Used for tranforming data
from HelpFunctions.DataTransformers import DataTransformers
#Used for visualising data
from HelpFunctions.DataVisualisers import DataVisualiser
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


#---------------------- Visualize Data ----------------------
#------ Images ------
#MNIST Train
DataVisualiser.plotImages(mnist_img_train[0:11], mnist_lbl_train[0:11])
#MNIST Test
DataVisualiser.plotImages(mnist_img_test[0:9], mnist_lbl_test[0:9])
#ORL
DataVisualiser.plotImages(orl_img[0:9], orl_lbl[0:9])

#------ 2D Plot -----
#MNIST Train
DataVisualiser.plot2dData(mnist_img_train_PCA, labels=mnist_lbl_train)
#MNIST Test
DataVisualiser.plot2dData(mnist_img_test_PCA, labels=mnist_lbl_test)
#ORL
DataVisualiser.plot2dData(orl_img_PCA, labels=orl_lbl)





#------------------------ Exercises ------------------------
#Exercise 1 - Nearest Class centroid classifier
#--------MNIST--------
#Full dimensionality
model_MNIST = NearestClassifiers.ClassCentroid.train(mnist_img_train, mnist_lbl_train)
accuracy_MNIST = NearestClassifiers.ClassCentroid.test(model_MNIST, mnist_img_test, mnist_lbl_test)
#PCA applied
model_MNIST_PCA = NearestClassifiers.ClassCentroid.train(mnist_img_train_PCA, mnist_lbl_train)
accuracy_MNIST_PCA = NearestClassifiers.ClassCentroid.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test)

#---------ORL---------
#Full dimensionality
model_ORL = NearestClassifiers.ClassCentroid.train(orl_img, orl_lbl)
accuracy_ORL = NearestClassifiers.ClassCentroid.test(model_ORL, orl_img, orl_lbl)
#PCA applied
model_ORL_PCA = NearestClassifiers.ClassCentroid.train(orl_img_PCA, orl_lbl)
accuracy_ORL_PCA = NearestClassifiers.ClassCentroid.test(model_ORL_PCA, orl_img_PCA, orl_lbl)




#Exercise 2 - Nearest sub-class centroid classifier using number of subclasses in set {2, 3, 5}
sets = [2, 3, 5]
#----MNIST----
model_MNIST = NearestClassifiers.SubClassCentroid.train(mnist_img_train, mnist_lbl_train, sets)
accuracy_MNIST = NearestClassifiers.SubClassCentroid.test(model_MNIST, mnist_img_test, mnist_lbl_test)
#-----ORL-----
model_ORL = NearestClassifiers.SubClassCentroid.train(orl_img, orl_lbl, sets)
accuracy_ORL = NearestClassifiers.SubClassCentroid.test(model_ORL, orl_img, orl_lbl)


#Exercise 3 - Nearest Neighbor classifier
neighbors = 3
#----MNIST----
model_MNIST = NearestClassifiers.SubClassCentroid.train(mnist_img_train, mnist_lbl_train, neighbors)
accuracy_MNIST = NearestClassifiers.SubClassCentroid.test(model_MNIST, mnist_img_test, mnist_lbl_test)
#-----ORL-----
model_ORL = NearestClassifiers.SubClassCentroid.train(orl_img, orl_lbl, neighbors)
accuracy_ORL = NearestClassifiers.SubClassCentroid.test(model_ORL, orl_img, orl_lbl)



#Exercise 4 - Perceptron trained using Backpropagation




#Exercise 5 - Perceptron trained using MSE (least squares solution)

