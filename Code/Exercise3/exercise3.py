


class NearestNeighborClassifier:
    def train(data, labels, properties = None):
        return 1

    def use(model, data_point):
        #Calculate distance to each category
        distances = [np.linalg.norm(data_point, category['coor']) for category in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category