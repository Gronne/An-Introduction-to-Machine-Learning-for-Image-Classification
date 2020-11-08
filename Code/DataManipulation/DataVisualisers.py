import matplotlib.pyplot as plt
import math
import numpy as NP



class DataVisualiser:
    def plot2dData(data_x_axis, data_y_axis = None, labels = None):
        #Split data is not already split
        if not isinstance(data_y_axis, list) and not isinstance(data_y_axis, NP.ndarray):
            data_y_axis = data_x_axis[:,1]
            data_x_axis = data_x_axis[:,0]
        
        #Generate colors from categories given
        if isinstance(labels, list) or isinstance(labels, NP.ndarray):
            colors = plt.cm.get_cmap('hsv', int(len(set(labels))))
            cat_to_color = {category: colors(index) for index, category in enumerate(set(labels))}
            labels = [cat_to_color[label] for label in labels]

        #Plot data
        fig=plt.figure()
        ax=fig.add_axes([0,0,1,1])
        ax.scatter(data_x_axis, data_y_axis, color=labels)
        plt.show()



    def plotImages(images, labels = []):
        #Check if data is formatted correctly
        if not isinstance(images, NP.ndarray) and not isinstance(images, list): DataVisualiser.plotImages([images], labels); return
        if isinstance(images, NP.ndarray) and len(images.shape) != 3: DataVisualiser.plotImages([images], labels); return
        if not isinstance(labels, NP.ndarray) and not isinstance(labels, list): DataVisualiser.plotImages(images, [labels]); return
        if isinstance(labels, NP.ndarray) and len(labels.shape) != 1: DataVisualiser.plotImages(images, [labels]); return

        #Calculate number of columns and rows
        length = len(images)
        columns = math.ceil(math.sqrt(length))
        rows = math.ceil(length/columns)

        #Creating the image
        fig = plt.figure()
        for count, image in enumerate(images):
            ax = fig.add_subplot(rows, columns, count + 1)
            plt.imshow(image)
            if count < len(labels):
                ax.set_title(str(labels[count]))

        #Showing the image
        plt.show()
