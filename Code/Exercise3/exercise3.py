import numpy as np
import math  
import random


class NearestNeighborClassifier:
    count = 0
    def train(data, labels, properties = None):
        #Calculate nr of random points to choose
        nr_of_new_points = NearestNeighborClassifier._calc_nr_of_new_points(data, labels)
        #Get points and labels
        jump_size = int(len(data)/nr_of_new_points)
        new_points = [data[index] for index in range(0, len(data), jump_size)]
        new_labels = [labels[index] for index in range(0, len(data), jump_size)]
        #Makes every random point a class
        model =  [{'class': category, 'coor':new_points[index].flatten()} for index, category in enumerate(new_labels)]
        return model


    def use(model, data_point):
        NearestNeighborClassifier.count += 1
        if NearestNeighborClassifier.count % 100 == 0:
            print(NearestNeighborClassifier.count)
        #Calculate distance to each category
        distances = [np.linalg.norm(category['coor'] - data_point.flatten()) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category

    
    def _calc_nr_of_new_points(data, labels):
        nr_of_categories = len(set(labels))
        nr_of_points = len(data)
        nr_of_points_per_category = nr_of_points / nr_of_categories
        nr_of_new_points_per_category = math.ceil(nr_of_points_per_category * 0.1)        #Can be made more mathematical
        nr_of_new_point = nr_of_new_points_per_category * nr_of_categories
        if nr_of_new_point < 100: 
            nr_of_new_point = nr_of_points if nr_of_points < 100 else 100
        print(f"Nr of points: {nr_of_new_point}")
        return nr_of_new_point