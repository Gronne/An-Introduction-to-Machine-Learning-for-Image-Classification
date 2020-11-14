import numpy as np
import math  
import random


class NearestNeighborClassifier:
    def train(data, labels, properties = None):
        if isinstance(properties, type(None)):
            properties = {'density': 1}
        #Calculate nr of random points to choose
        nr_of_new_points = NearestNeighborClassifier._calc_nr_of_new_points(data, labels, properties['density'])
        #Get points and labels
        jump_size = int(round(len(data)/nr_of_new_points))
        new_points = [data[index] for index in range(0, len(data), jump_size)]
        new_labels = [labels[index] for index in range(0, len(data), jump_size)]
        #Makes every random point a class
        model =  [{'class': category, 'coor':new_points[index].flatten()} for index, category in enumerate(new_labels)]
        return model


    def use(model, data_point, properties = None):
        #Set to one neighbor if nothing have been set
        if isinstance(properties, type(None)):
            properties = {'neighbors': 1}
        #Calculate distance to each category
        distances = [{'index': index, 'dist': np.linalg.norm(category['coor'] - data_point.flatten()), 'class': category['class']} for index, category in enumerate(model)]
        #Sort distance list
        distances_sorted = sorted(distances, key=lambda k: k['dist']) 
        #Get closest neighbors
        clostest_neighbors = distances_sorted[0:properties['neighbors']]
        #Find most dominant category
        category = NearestNeighborClassifier._dominant_category(clostest_neighbors)
        #Return closest category
        return category

    
    def _dominant_category(clostest_neighbors):
        #Split up into categories
        categories = {category: [] for category in set([neighbor['class'] for neighbor in clostest_neighbors])}
        for neighbor in clostest_neighbors: categories[neighbor['class']].append(neighbor)
        #Calculate category weight
        category_strengths = [NearestNeighborClassifier._category_strength(key, categories[key]) for key in categories]
        #Return strongest category
        return sorted(category_strengths, key=lambda k: k['score'])[0]['category'] 

    
    def _category_strength(category, neighbors):
        #Maximum distance and then invert it to get the strongest
        return {'category': category, 'score': len(neighbors)}

    
    def _calc_nr_of_new_points(data, labels, density):
        nr_of_categories = len(set(labels))
        nr_of_points = len(data)
        nr_of_points_per_category = nr_of_points / nr_of_categories
        nr_of_new_points_per_category = math.ceil(nr_of_points_per_category * density)        #Can be made more mathematical
        nr_of_new_point = nr_of_new_points_per_category * nr_of_categories
        if nr_of_new_point < 100: 
            nr_of_new_point = nr_of_points if nr_of_points < 100 else 100
        return nr_of_new_point