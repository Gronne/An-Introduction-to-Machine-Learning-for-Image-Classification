import matplotlib.pyplot as plt
import math
import numpy as NP



class DataVisualiser:
    def plot2dData(x_axis, y_axis = None):
        pass 


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
