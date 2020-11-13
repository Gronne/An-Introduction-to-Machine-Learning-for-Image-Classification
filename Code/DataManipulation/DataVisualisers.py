import matplotlib.pyplot as plt
import math
import numpy as np

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.ticker as ticker



class DataVisualiser:
    def plot2dData(data_x_axis, data_y_axis = None, labels = None):
        #Split data is not already split
        if not isinstance(data_y_axis, list) and not isinstance(data_y_axis, np.ndarray):
            data_y_axis = data_x_axis[:,1]
            data_x_axis = data_x_axis[:,0]
        
        #Generate colors from categories given
        if isinstance(labels, list) or isinstance(labels, np.ndarray):
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
        if not isinstance(images, np.ndarray) and not isinstance(images, list): DataVisualiser.plotImages([images], labels); return
        if isinstance(images, np.ndarray) and len(images.shape) != 3: DataVisualiser.plotImages([images], labels); return
        if not isinstance(labels, np.ndarray) and not isinstance(labels, list): DataVisualiser.plotImages(images, [labels]); return
        if isinstance(labels, np.ndarray) and len(labels.shape) != 1: DataVisualiser.plotImages(images, [labels]); return

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


    def plotStats(train_stats, test_stats = None, normalize = False, title = "Stats"):
        #Define dimension names
        dimension_names = ['Accuracy', 'Training time', 'Prediction time']
        #Check if data is split and if so, combine it
        data = DataVisualiser._spider_setup_data(train_stats, test_stats)
        #Normalize data
        if normalize == True:
            data, max_values = DataVisualiser._normalize_data(data)
            dimension_names = ["\n" + name + "\n" + str(round(max_values[count], 6)) for count, name in enumerate(dimension_names)]
        #Number of dimensions
        nr_dimensions = len(dimension_names)
        #Create spider plot object
        theta = SpiderPlot.spider_factory(nr_dimensions)
        #Setup spider plot
        SpiderPlot.plot(theta, title, dimension_names, data)
        #Plot
        plt.show()

    def _spider_setup_data(train_stats, test_stats):
        #Check if None
        if isinstance(test_stats, type(None)):
            return train_stats
        #Check if list or dict
        if isinstance(train_stats, type({})):
            DataVisualiser._spider_setup_data(dimension_names, [train_stats], [test_stats])
        #Check if user have split model and stats
        train_stats = [stat if DataVisualiser._check_if_is_splitted(stat) else stat[1] for stat in train_stats]
        test_stats = [stat if DataVisualiser._check_if_is_splitted(stat) else stat[1] for stat in test_stats]
        #Check if the same amounts of training and test stats are given
        if len(train_stats) != len(test_stats):
            raise Exception("The same amount of training stats and test stats must be given")
        #Combine stat data
        data = [ [test_stats[count]['score'], train_stats[count]['time'], test_stats[count]['time']] for count in range(len(train_stats))]
        #Call Spiderplot with combines data
        return data
    
    def _check_if_is_splitted(model):
        return not(isinstance(model, tuple) and len(model) == 2 and isinstance(model[1], type({})))


    def _normalize_data(data):
        max_values = []
        #Normalize
        for column_count in range(len(data[0])):
            #Get dependent data
            column = [data[row_count][column_count] for row_count in range(len(data))]
            #Find max value to normalize with
            max_value = max(column)
            max_values.append(max_value)
            #Normalize dependent data and put into new list
            for row_count in range(len(data)): data[row_count][column_count] /= max_value
        return data, max_values


    def plotColumnDiagram(dimension_names, data, individual_scaling = False):
        #Create plot
        column_diagram = ColumnPlot(dimension_names, individual_scaling)
        #Insert data
        column_diagram.plot(data)
        #Show plot
        plt.show()









#https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
class SpiderPlot:
    def plot(factory, title, names, data):
        #Create plot template
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
        fig.subplots_adjust(top=0.85, bottom=0.05)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], fontsize=0) 
        ax.set_title(title, ha='center', pad=12, fontsize=16)
        #Plot data
        for d in data:
            line = ax.plot(factory, d)
            ax.fill(factory, d,  alpha=0.25)
        ax.set_varlabels(names)
        
    
    def spider_factory(num_vars, frame='polygon'):
        # calculate evenly-spaced axis angles
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarAxes(PolarAxes):

            name = 'radar'

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # rotate plot such that the first axis is at the top
                self.set_theta_zero_location('N')

            def fill(self, *args, closed=True, **kwargs):
                """Override fill so that line is closed by default"""
                return super().fill(closed=closed, *args, **kwargs)

            def plot(self, *args, **kwargs):
                """Override plot so that line is closed by default"""
                lines = super().plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                x, y = line.get_data()
                # FIXME: markers at x[0], y[0] get doubled-up
                if x[0] != x[-1]:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
                # in axes coordinates.
                if frame == 'circle':
                    return Circle((0.5, 0.5), 0.5)
                elif frame == 'polygon':
                    return RegularPolygon((0.5, 0.5), num_vars,
                                            radius=.5, edgecolor="k")
                else:
                    raise ValueError("unknown value for 'frame': %s" % frame)

            def draw(self, renderer):
                """ Draw. If frame is polygon, make gridlines polygon-shaped """
                if frame == 'polygon':
                    gridlines = self.yaxis.get_gridlines()
                    for gl in gridlines:
                        gl.get_path()._interpolation_steps = num_vars
                super().draw(renderer)


            def _gen_axes_spines(self):
                if frame == 'circle':
                    return super()._gen_axes_spines()
                elif frame == 'polygon':
                    # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                    spine = Spine(axes=self,
                                    spine_type='circle',
                                    path=Path.unit_regular_polygon(num_vars))
                    # unit_regular_polygon gives a polygon of radius 1 centered at
                    # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                    # 0.5) in axes coordinates.
                    spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                        + self.transAxes)


                    return {'polar': spine}
                else:
                    raise ValueError("unknown value for 'frame': %s" % frame)

        register_projection(RadarAxes)
        return theta





class ColumnPlot:
    def __init__(self, column_names, individual_scaling):
        fig, ax = plt.subplots()
        self._fig = fig  
        self._ax = ax
        self._x = np.arange(len(column_names))
        self._width = 1/len(column_names)
        #Setup plot
        self._ax.set_ylabel('Scores')
        self._ax.set_title('Column diagram')
        self._ax.set_xticks(self._x)
        self._ax.set_xticklabels(column_names)
        self._ax.legend()


    def plot(self, data):
        #Set each column rectangle
        rects = [self._ax.bar(self._x + (self._width*count), d, self._width) for count, d in enumerate(data)]
        #plot
        for rect in rects:
            for point in rect:
                height = point.get_height()
                self._ax.annotate('{}'.format(height),
                            xy=(point.get_x() + point.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')
        self._fig.tight_layout()


