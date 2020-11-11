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
#Split ORL into training and test sets
split_ratio = 0.7
orl_img_train, orl_lbl_train, orl_img_test, orl_lbl_test = Datasets.split_dataset(orl_img, orl_lbl, split_ratio)


#---------------------- Transform Data ----------------------
#Transform MNIST data
mnist_img_train_PCA = DataTransformers.PCA.tranform(mnist_img_train, 2)
mnist_img_test_PCA = DataTransformers.PCA.tranform(mnist_img_test, 2)

#Transform ORL Data
orl_img_train_PCA = DataTransformers.PCA.tranform(orl_img_train, 2)
orl_img_test_PCA = DataTransformers.PCA.tranform(orl_img_test, 2)


# #---------------------- Visualize Data ----------------------
# #------ Images ------
# #MNIST Train
# DataVisualiser.plotImages(mnist_img_train[0:11], mnist_lbl_train[0:11])
# #MNIST Test
# DataVisualiser.plotImages(mnist_img_test[0:9], mnist_lbl_test[0:9])
# #ORL
# DataVisualiser.plotImages(orl_img_train[0:9], orl_lbl_train[0:9])

# #------ 2D Plot -----
# #MNIST Train
# DataVisualiser.plot2dData(mnist_img_train_PCA, labels=mnist_lbl_train)
# #MNIST Test
# DataVisualiser.plot2dData(mnist_img_test_PCA, labels=mnist_lbl_test)
# #ORL
# DataVisualiser.plot2dData(orl_img_train_PCA, labels=orl_lbl_train)





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
# model_ORL = NearestClassifiers.ClassCentroid.train(orl_img_train, orl_lbl_train)
# accuracy_ORL = NearestClassifiers.ClassCentroid.test(model_ORL, orl_img_test, orl_lbl_test)
# print(f"NCCC - ORL Full Accuracy: {accuracy_ORL}")
# #PCA applied
# model_ORL_PCA = NearestClassifiers.ClassCentroid.train(orl_img_train_PCA, orl_lbl_train)
# accuracy_ORL_PCA = NearestClassifiers.ClassCentroid.test(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test)
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
#     model_ORL = NearestClassifiers.SubClassCentroid.train(orl_img_train, orl_lbl_train, properties)
#     accuracy_ORL = NearestClassifiers.SubClassCentroid.test(model_ORL, orl_img_test, orl_lbl_test)
#     print(f"NSCCC({nr_sub_classes}) - ORL Full Accuracy: {accuracy_ORL}")
#     #PCA applied
#     model_ORL_PCA = NearestClassifiers.SubClassCentroid.train(orl_img_train_PCA, orl_lbl_train, properties)
#     accuracy_ORL_PCA = NearestClassifiers.SubClassCentroid.test(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test)
#     print(f"NSCCC({nr_sub_classes}) - ORL PCA Accuracy: {accuracy_ORL_PCA}")




# #Exercise 3 - Nearest Neighbor classifier
# properties = {'neighbors': 10, 'density': 0.001} #Hyper parametre: Distance function, neighbors, density, density function, strength function
# #--------MNIST--------
# #Full dimensionality
# model_MNIST = NearestClassifiers.Neighbor.train(mnist_img_train, mnist_lbl_train, properties)
# accuracy_MNIST = NearestClassifiers.Neighbor.test(model_MNIST, mnist_img_test, mnist_lbl_test, properties)
# print(f"NNC - MNIST Full Accuracy: {accuracy_MNIST}")
# #PCA applied
# model_MNIST_PCA = NearestClassifiers.Neighbor.train(mnist_img_train_PCA, mnist_lbl_train, properties)
# accuracy_MNIST_PCA = NearestClassifiers.Neighbor.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test, properties)
# print(f"NNC - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

# #---------ORL---------
# properties = {'neighbors': 1, 'density': 1}
# #Full dimensionality
# model_ORL = NearestClassifiers.Neighbor.train(orl_img_train, orl_lbl_train, properties)
# accuracy_ORL = NearestClassifiers.Neighbor.test(model_ORL, orl_img_test, orl_lbl_test, properties)
# print(f"NNC - ORL Full Accuracy: {accuracy_ORL}")
# #PCA applied
# model_ORL_PCA = NearestClassifiers.Neighbor.train(orl_img_train_PCA, orl_lbl_train, properties)
# accuracy_ORL_PCA = NearestClassifiers.Neighbor.test(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test, properties)
# print(f"NNC - ORL PCA Accuracy: {accuracy_ORL_PCA}")





#Exercise 4 - Perceptron trained using Backpropagation
properties = {'hidden_layers': [{'nodes': 8}, {'nodes': 16}, {'nodes': 32}], 'activation': 'sigmoid', 'epochs': 20, 'verbose': True} 
#--------MNIST--------
#Full dimensionality
model_MNIST = Perceptrons.Backpropagation.train(mnist_img_train, mnist_lbl_train, properties)
accuracy_MNIST = Perceptrons.Backpropagation.test(model_MNIST, mnist_img_test, mnist_lbl_test)
print(f"PB - MNIST Full Accuracy: {accuracy_MNIST}")
#PCA applied
model_MNIST_PCA = Perceptrons.Backpropagation.train(mnist_img_train_PCA, mnist_lbl_train, properties)
accuracy_MNIST_PCA = Perceptrons.Backpropagation.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test)
print(f"PB - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

#---------ORL---------
#Full dimensionality
model_ORL = Perceptrons.Backpropagation.train(orl_img_train, orl_lbl_train, properties)
accuracy_ORL = Perceptrons.Backpropagation.test(model_ORL, orl_img_test, orl_lbl_test)
print(f"PB - ORL Full Accuracy: {accuracy_ORL}")
#PCA applied
model_ORL_PCA = Perceptrons.Backpropagation.train(orl_img_train_PCA, orl_lbl_train, properties)
accuracy_ORL_PCA = Perceptrons.Backpropagation.test(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test)
print(f"PB - ORL PCA Accuracy: {accuracy_ORL_PCA}")





# #Exercise 5 - Perceptron trained using MSE (least squares solution)
# #--------MNIST--------
# #Full dimensionality
# model_MNIST = Perceptrons.LeastSquare.train(mnist_img_train, mnist_lbl_train)
# accuracy_MNIST = Perceptrons.LeastSquare.test(model_MNIST, mnist_img_test, mnist_lbl_test)
# print(f"PLS - MNIST Full Accuracy: {accuracy_MNIST}")
# #PCA applied
# model_MNIST_PCA = Perceptrons.LeastSquare.train(mnist_img_train_PCA, mnist_lbl_train)
# accuracy_MNIST_PCA = Perceptrons.LeastSquare.test(model_MNIST_PCA, mnist_img_test_PCA, mnist_lbl_test)
# print(f"PLS - MNIST PCA Accuracy: {accuracy_MNIST_PCA}")

# #---------ORL---------
# #Full dimensionality
# model_ORL = Perceptrons.LeastSquare.train(orl_img_train, orl_lbl_train)
# accuracy_ORL = Perceptrons.LeastSquare.test(model_ORL, orl_img_test, orl_lbl_test)
# print(f"PLS - ORL Full Accuracy: {accuracy_ORL}")
# #PCA applied
# model_ORL_PCA = Perceptrons.LeastSquare.train(orl_img_train_PCA, orl_lbl_train)
# accuracy_ORL_PCA = Perceptrons.LeastSquare.test(model_ORL_PCA, orl_img_test_PCA, orl_lbl_test)
# print(f"PLS - ORL PCA Accuracy: {accuracy_ORL_PCA}")
