import numpy as np


class PerceptronBackpropogation:                    #What about batch size?
    def train(data, labels, properties = None):
        #Defining default properties
        default_properties = PerceptronBackpropogation._get_default_properties(labels)
        #Check if properties have been given and otherwise set it to empty
        if isinstance(properties, type(None)):
            properties = {}
        #Set default in where properties is missing
        properties = { **default_properties, **properties }
        #Normalize data - between 0 and 1
        data_normalized = data * (1/data.max())
        #Create Neural network model
        NN_model = NeuralNetworkModel( properties['hidden_layers'], 
                                       properties['activation'], 
                                       len(data[0].flatten()), 
                                       properties['categories'],
                                       properties['verbose'])
        #Train model
        NN_model.train(data_normalized, labels, properties['epochs'], properties['learning_rate'])
        #Return model
        return NN_model
        

    def use(model, data_point, properties = None):
        #Find category
        category = model.predict(data_point)
        #Return category
        return category

    def _get_default_properties(labels):
        default_properties = { 'hidden_layers': [{'nodes': 8}, {'nodes': 8}, {'nodes': 8}], 
                               'activation': 'sigmoid', 
                               'learning_rate': 1, 
                               'epochs': 10, 
                               'categories': list(set(labels)), 
                               'verbose': False }
        return default_properties







class NeuralNetworkModel:
    def __init__(self, hidden_layers, activation, input_layers, output_layer, verbose):
        #Create combined list of all layers
        self._layers = [{'nodes': input_layers }] + hidden_layers + [{'nodes': len(output_layer)}]
        #Activation function used in nodes
        self._activation_function = self._get_activation(activation)
        #Output layer names
        self._output_layer = output_layer
        #Weights used for the connect between layers
        self._weights = self._make_weight_list(self._layers)
        #Set verbose - Define if the model should print out feedback as it trains
        self._verbose = verbose
    

    def _make_weight_list(self, layers):  
        #create weight list with random starting values - len(layers) is minimum 2 (input and output)
        weights = [np.random.randn(layers[index]['nodes'], layers[index+1]['nodes']) for index in range(len(layers)-1)]
        #Return weights
        return weights


    def train(self, data, labels, epochs = 10, training_rate = 0.5):
        #Flatten data
        data_flattened = data.flatten()
        data_flattened = data_flattened.reshape((data.shape[0], int(data_flattened.shape[0]/data.shape[0])))
        #Ready label list for training
        labels = self._transform_labels(labels)
        #Train
        for iteration_nr in range(epochs):
            #Forward prediction and backward propogation
            forward_layer_outputs = self._feed_forward(data_flattened)
            self._back_propogation(data_flattened, labels, forward_layer_outputs)
            #Verbose
            if self._verbose == True:
                loss = np.mean(np.square(labels - forward_layer_outputs[-1]))
                print(f"Epoch: {iteration_nr}, Loss: {loss}")

    
    def _transform_labels(self, labels):
        #New label list
        label_list = np.zeros((len(labels), len(self._output_layer)))
        #Make each label into an array
        for index, label in enumerate(labels):
            #Find label index in output
            label_index = self._output_layer.index(label)
            #Change label index to high
            label_list[index][label_index] = 1.0
        return label_list


    def predict(self, data_point):
        #Flatten point
        point = data_point.flatten()
        #Get output
        forward_layer_outputs = self._feed_forward(point)
        output = forward_layer_outputs[-1]
        #Get largest category
        largest_value_index = np.unravel_index(np.argmax(output, axis=None), output.shape)
        largest_category = self._output_layer[largest_value_index[0]]
        #Return category
        return largest_category
        

    def print(self):
        print(f"Layers: {[layer['nodes']] for layer in self._layers}")


    def _feed_forward(self, data):
        #Forward is the normal way through the system and is what "prediction" also will use
        forward_outputs = [data]
        #Feed forward through each layer until values is aviable for each of the output nodes
        for weight in self._weights:
            #Dot the data with the weights between the current and next layers
            result_weighted = np.dot(forward_outputs[-1], weight)
            #Run activation for each node in second layer
            result_filtered = self._activation_function(result_weighted)
            forward_outputs.append(result_filtered)
        #Return forward outputs
        return forward_outputs


    def _get_activation(self, activation_function):
        #Switch case
        if activation_function == 'sigmoid':
            return self._sigmoid


    def _sigmoid(self, values, derived_value = False):
        return values * (1 - values) if derived_value else 1 /(1 + np.exp(-values))


    def _back_propogation(self, data, labels, forward_layer_outputs):
        #Calcualte error deltas backward through the system
        backward_deltas = self._calc_backward_error_deltas(labels, forward_layer_outputs)
        #Adjust weights forwards through the system
        self._adjust_weights_forward(data, forward_layer_outputs, backward_deltas)


    def _calc_backward_error_deltas(self, labels, forward_layer_outputs):
        #Calculate first delta based on labels
        output_error = labels - forward_layer_outputs[-1]
        output_delta = output_error * self._activation_function(forward_layer_outputs[-1], derived_value=True)
        backward_error_deltas = [output_delta]
        #Calculate error backwards through the system
        for index in range(len(forward_layer_outputs)-2):
            #Find the relevant indexes
            weight_index = -(index+1)
            layer_index = -(index+2)
            #Calculate error for layer
            layer_error = backward_error_deltas[-1].dot(self._weights[weight_index].transpose())
            #Calculate the difference from feed forward
            layer_delta = layer_error * self._activation_function(forward_layer_outputs[layer_index], derived_value=True)
            #Append to the delta list
            backward_error_deltas.append(layer_delta)
        #Return backward deltas
        return backward_error_deltas


    def _adjust_weights_forward(self, data, forward_layer_outputs, backward_deltas):
        #Adjust the rest of the layers based on the backward deltas - First forward_layer is the data
        for index in range(len(self._weights)):
            #Find relevant indexes
            delta_index = -(index + 1)
            #Adjust weight
            self._weights[index] = np.add(self._weights[index], forward_layer_outputs[index].transpose().dot(backward_deltas[delta_index]))

