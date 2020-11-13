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
        prediction_list = [{'label': label, 'prediction': MachineLearning.use(ml_class, model, point, properties)} for point, label in zip(data_points, data_labels)]
        #Stop time
        time_diff = time.time() - start_time
        #Calcualte overall score
        total_count = len(prediction_list)
        count_true = [prediction['label'] == prediction['prediction'] for prediction in prediction_list].count(True)
        accuracy = count_true / total_count
        #Calculate score for each category
        category_scores = {label: MachineLearning._label_score(prediction_list, label) for label in set(data_labels)}
        #Calculate stats
        test_stat = {'time': time_diff / len(data_points), 'score': accuracy, 'detailed_score': category_scores, 'full_time': time_diff }
        #Return accuracy and stats
        return accuracy, test_stat

    def _label_score(predictions, label):
        #get label predictions
        label_prediction = [prediction['label'] == prediction['prediction'] for prediction in predictions if prediction['label'] == label]
        #Calculate score
        label_score = label_prediction.count(True) / len(label_prediction)
        #Return score
        return label_score




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