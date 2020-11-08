import numpy as np


class NearestClassCentroidClassifier:
    def train(data, labels, properties = None):
        #Find center of each category
        model = [{'class': category, 'coor': np.mean([data[i].flatten() for i in [index for index, value in enumerate(labels) if value == category]], axis=0)} for category in set(labels)]
        return model

    def use(model, data_point):
        #Calculate distance to each category
        distances = [np.linalg.norm(data_point.flatten() - category['coor']) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category