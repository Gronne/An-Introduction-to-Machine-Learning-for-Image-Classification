


class LeastSquareSolution:
    def train(data, labels, properties = None):
        pass

    def use(model, data_point, properties = None):
        #Calculate distance to each category
        distances = [np.linalg.norm(cluster['coor'] - data_point.flatten()) for cluster in model]
        #Find index of minimum distance
        model_index = distances.index(min(distances))
        #Get category corrosponding to minimum distance
        category = model[model_index]['class']
        #Return closest category
        return category