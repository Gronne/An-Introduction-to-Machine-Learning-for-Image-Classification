import numpy as np 



class DataTransformers:
    class PCA:
        def tranform(data, dimensions):
            if isinstance(dimensions, int):
                return PrincipalComponentAnalysis.tranform(data, dimensions)
            elif isinstance(dimensions, list) or isinstance(dimensions, np.ndarray):
                return PrincipalComponentAnalysis.transform_w_model(data, dimensions)
            else:
                raise Exception("Dimensions must either be a integer or a previous found model")





class PrincipalComponentAnalysis:
        def tranform(data, dimensions, normalize = False):
            #Flatten data
            data_flattened = data.flatten()
            data_f = data_flattened.reshape((data.shape[0], int(data_flattened.shape[0]/data.shape[0])))
            #Center data for optimal reduction
            data_mean = data_f.mean(axis=0)
            data_centered = data_f - data_mean
            #Calculate Covariance matrix
            co_mat = np.cov(data_centered.transpose())
            #Find eigenvalues for sort them - Smallest will be the strongest direction
            eigen_values, eigen_vectors = np.linalg.eig(co_mat)
            sorted_indexes = eigen_values.argsort()[::-1]            #Reverse
            eigen_vectors_sorted = eigen_vectors[:,sorted_indexes]
            eigen_values_sorted = eigen_values[sorted_indexes]
            #Get the same amount of eigen vectors as the number of dimensions
            if dimensions > 0: eigen_vectors = eigen_vectors_sorted[:,:dimensions] 
            else: raise Exception("More then 0 dimensions required")
            #Normalize
            if normalize: eigen_vectors = PrincipalComponentAnalysis._normalize_eigen_vector(eigen_values_sorted, eigen_vectors_sorted)
            #Transform data
            data_transformed = np.dot(eigen_vectors.transpose(), data_centered.transpose())
            #Return transformed dimensions
            return data_transformed.transpose(), eigen_vectors


        def _normalize_eigen_vector(eigen_values, eigen_vectors):
            nr_vectors = eigen_vectors.shape[1]
            for index in range(nr_vectors):
                vector = eigen_vectors[:,index]
                value = eigen_values[index]
                normalized_vector = vector / np.linalg.norm(vector) * value.sqrt()
                eigen_vectors[:, index] = normalized_vector
            return eigen_vectors

        
        def transform_w_model(data, eigen_vectors):
            #Flatten data
            data_flattened = data.flatten()
            data_f = data_flattened.reshape((data.shape[0], int(data_flattened.shape[0]/data.shape[0])))
            #Center data for optimal reduction
            data_mean = data_f.mean(axis=0)
            data_centered = data_f - data_mean
            #Transform data
            data_transformed = np.dot(eigen_vectors.transpose(), data_centered.transpose())
            #Return transformed dimensions
            return data_transformed.transpose(), eigen_vectors



