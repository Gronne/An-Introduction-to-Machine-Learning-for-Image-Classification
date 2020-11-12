#Nearest Neighbor algorithms
from Exercise1.exercise1 import NearestClassCentroidClassifier as NC_ex1
from Exercise2.exercise2 import NearestSubClassCentroidClassifier as NC_ex2
from Exercise3.exercise3 import NearestNeighborClassifier as NC_ex3

#Perception algorithms
from Exercise4.exercise4 import PerceptronBackpropogation as P_ex4
from Exercise5.exercise5 import LeastSquareSolution as P_ex5

#Import to measure execution time
import time
#import to check if model is a class
import inspect


class MachineLearning:
    def train(ml_class, data, labels, properties = None):
        #Start time for stat measurement
        start_time = time.time()
        #Train model
        model = ml_class.train(data, labels, properties)
        #Calculate stats
        train_stats = {'time': time.time() - start_time}
        #Return model and stats
        return model, train_stats

    def use(ml_class, model, data_point, properties = None):
        #Use the trained model to find a label
        point_class = ml_class.use(model, data_point, properties)
        return point_class

    def get_accuracy(ml_class, model, data_points, data_labels, properties = None):
        #Check if user have split model and stats
        if isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], type({})):
            model = model[0]
        #Start timer used to claculate stats
        start_time = time.time()
        #Call 'use' for each datapoint and match it with a label
        bool_list = [MachineLearning.use(ml_class, model, point, properties) == label for point, label in zip(data_points, data_labels)]
        #Calcualte accuracy
        total_count = len(bool_list)
        count_true = bool_list.count(True)
        accuracy = count_true / total_count
        #Calculate stats
        time_diff = time.time() - start_time
        test_stat = {'time': time_diff / len(data_points), 'score': accuracy, 'full_time': time_diff }
        #Return accuracy and stats
        return accuracy, test_stat



class NearestClassifiers:
    class ClassCentroid:
        def train(data_train, labels):
            return MachineLearning.train(NC_ex1, data_train, labels)

        def use(model, data_point):
            return MachineLearning.use(NC_ex1, model, data_point)

        def test(model, data_test, labels_test):
            return MachineLearning.get_accuracy(NC_ex1, model, data_test, labels_test)


    class SubClassCentroid:
        def train(data_train, labels, properties = 1):
            return MachineLearning.train(NC_ex2, data_train, labels, properties)

        def use(model, data_point):
            return MachineLearning.use(NC_ex2, model, data_point)

        def test(model, data_test, labels_test):
            return MachineLearning.get_accuracy(NC_ex2, model, data_test, labels_test)


    class Neighbor:
        def train(data_train, labels, properties = None):
            return MachineLearning.train(NC_ex3, data_train, labels, properties)

        def use(model, data_point):
            return MachineLearning.use(NC_ex3, model, data_point)

        def test(model, data_test, labels_test, properties = None):
            return MachineLearning.get_accuracy(NC_ex3, model, data_test, labels_test, properties)





class Perceptrons:
    class Backpropagation(MachineLearning):
        def train(data_train, labels, properties):
            return MachineLearning.train(P_ex4, data_train, labels, properties)

        def use(model, data_point):
            return MachineLearning.use(P_ex4, model, data_point)

        def test(model, data_test, labels_test):
            return MachineLearning.get_accuracy(P_ex4, model, data_test, labels_test)


    class LeastSquare(MachineLearning):
        def train(data_train, labels):
            return MachineLearning.train(P_ex5, data_train, labels)

        def use(model, data_point):
            return MachineLearning.use(P_ex5, model, data_point)

        def test(model, data_test, labels_test):
            return MachineLearning.get_accuracy(P_ex5, model, data_test, labels_test)