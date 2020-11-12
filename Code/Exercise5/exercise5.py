import numpy as np 
import copy


class LeastSquareSolution:
    def train(data, labels, properties = None):
        #Flatten data
        data_flattened = data.flatten()
        data_f = data_flattened.reshape((data.shape[0], int(data_flattened.shape[0]/data.shape[0])))
        #Create a model where each label have a least square solution
        model = { label: LeastSquareSolution._calculate_least_square_weights_for_label(data_f, labels, label) for label in set(labels)}
        #Return model 
        return model 

    def _calculate_least_square_weights_for_label(data, labels, label):
        #Append off-set
        ones = np.ones((len(data), 1))
        data_o = np.append(data, ones, axis = 1)
        #Transform labels to 1 if they match the label given as argument and -1 if not
        pooled_labels = labels.copy().astype('int')
        for count, original_label in enumerate(pooled_labels):
            pooled_labels[count] = 1 if original_label == label else -1
        #Calcualte A matrix
        A = LeastSquareSolution._calculate_A_matrix(data_o, pooled_labels)
        #Return matrix A
        return A


    def _calculate_A_matrix(data, labels):
        #Dot transposed data and data
        dot_data_matrix = data.transpose().dot(data)
        #Dot transposed data and labels
        dot_label_matrix = np.dot(data.transpose(), labels.transpose())
        #Inverse Of dot data matrix
        inverse = np.linalg.pinv(dot_data_matrix)
        #Dot the inverse matrix with dot label matrix
        A = np.dot(inverse, dot_label_matrix)
        #Return A
        return A


    def use(model, data_point, properties = None):
        #Flatten data point and append 1 as in the training
        data_point_f = np.append(data_point.flatten(), [1])
        #Go through each category and calculate score
        scores = { key: np.dot(data_point_f, model[key]) for key in model }
        #Find category with largest score
        category = max(scores, key=scores.get)
        #Return category
        return category