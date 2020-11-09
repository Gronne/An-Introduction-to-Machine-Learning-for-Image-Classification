import numpy as np


class NearestClassCentroidClassifier:
    def train(data, labels, properties = None):
        CR = NearestClassCentroidClassifier
        #Find center of each category and add it to the model
        model = [{'class': category, 'coor': CR._get_centroid_for_category(data, labels, category)} for category in set(labels)]
        return model

    def use(model, data_point, properties = None):
        #Calculate distance to each category
        distances = [np.linalg.norm(category['coor'] - data_point.flatten()) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category


    def _get_centroid_for_category(data, labels, category):
        #Find points in category
        category_points = NearestClassCentroidClassifier._get_points_from_category(data, labels, category)
        #Flatten points
        flattened_points = NearestClassCentroidClassifier._flatten_points(category_points) 
        #Find mean poing
        mean_point = np.mean(flattened_points, axis=0)
        return mean_point

    def _get_points_from_category(points, labels, category):
        #Find points in category
        return [points[index] for index in NearestClassCentroidClassifier._indexes_w_cat(labels, category)]
    
    def _indexes_w_cat(labels, category):
        #Find labels in category
        return [index for index, value in enumerate(labels) if value == category]

    def _flatten_points(points):
        #Flatten points
        return [point.flatten() for point in points]