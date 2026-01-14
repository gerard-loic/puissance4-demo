import numpy as np
import os
import pickle

#Activation functions
def identity(x):
    return x

def identity_deriv(x):
    return 1.0

def relu(x):
    return (x > 0) * x

def relu_deriv(x):
    return (x > 0) * 1.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - (tanh(x)**2)


class SimpleNeuralNetwork:
    activation_fct_map = {
        'identity' : identity,
        'relu' : relu,
        'sigmoid' : sigmoid,
        'tanh' : tanh
    }
    activation_fct_deriv_map = {
        'identity' : identity_deriv,
        'relu' : relu_deriv,
        'sigmoid' : sigmoid_deriv,
        'tanh' : tanh_deriv
    }

    def __init__(self, layers_sizes:list, activation_functions:list, alpha:float=0.005, iterations:int=350, file_name:str=None, silent:bool=False):
        self.alpha = alpha  #Learning rate
        self.iterations = iterations
        self.layers_sizes = layers_sizes
        self.activation_functions = activation_functions
        self.file_name = file_name

        #Initialisation des poids
        self.weights = []
        if file_name is None or not os.path.exists(file_name):
            for j in range(len(self.layers_sizes) - 1):
                self.weights.append(0.2 * np.random.random((self.layers_sizes[j], layers_sizes[j + 1])) - 0.1)
        else:
            with open(file_name, 'rb') as f:
                for j in range(len(self.layers_sizes) - 1):
                    weights_table = pickle.load(f)
                    assert weights_table.shape == (self.layers_sizes[j], self.layers_sizes[j + 1])
                    self.weights.append(weights_table)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            for j in range(len(self.weights)):
                pickle.dump(self.weights[j], f)


    def forward(self, input_matrix:list):
        output_layer = np.array(input_matrix)

        #On passe à travers toutes les couches
        for i in range(len(self.weights)):
            output_layer = np.dot(output_layer, self.weights[i])
            act_fn = self.activation_fct_map[self.activation_functions[i]]
            output_layer = act_fn(output_layer)

        return output_layer
    
    def evaluate(self, test_inputs:list, test_labels:list):
        error = 0.0

        for i in range(len(test_inputs)):
            input_layer = test_inputs[i:i +1 ]
            output_layer = self.forward(input_layer)
            error += np.sum((test_labels[i:i+1] - output_layer)**2)
            #correct_cnt += int(np.argmax(output_layer)) == np.argmax(test_labels[i:i+1])

        return f"Evaluation | avg error: {error / float(len(test_inputs))}  "

    def train(self, inputs:list, labels:list, test_inputs:list=None, test_labels:list=None, with_dropout:bool=True, print_every_other_step:int=10, silent:bool=False):
        for iteration in range(self.iterations):
            error = 0.0

            for i in range(len(inputs)):
                input_layer = inputs[i:i+1]

                layers = []
                dropout_masks = []
                layers.append(input_layer)


                for k in range(len(self.weights)):
                    #Application des poids
                    layer = np.dot(layers[-1], self.weights[k])


                    #Activation fonction
                    act_fct = self.activation_fct_map[self.activation_functions[k]]
                    layer = act_fct(layer)

                    #Dropout
                    if with_dropout and not k == len(self.weights) -1: #On n'applique pas si c'est la dernière couche
                        dropout_mask = np.random.randint(2, size=layer.shape)
                        layer *= 2 * dropout_mask 
                        dropout_masks.append(dropout_mask)
                    layers.append(layer)

                r = np.sum((labels[i:i + 1] - layers[-1]) **2)
                error += np.sum((labels[i:i + 1] - layers[-1]) **2)
                #correct_cnt += int(np.argmax(layers[-1]) == np.argmax(labels[i:i + 1]))

                #calculate delta of the layers (en commençant par la dernière couche)
                layer_deltas = [(labels[i:i + 1] - layers[-1])]
                act_fn_deriv = self.activation_fct_deriv_map[self.activation_functions[-1]]
                layer_deltas[-1] *= act_fn_deriv(layers[-1])

                for j in range(len(self.weights) - 1, 0, -1): #On remonte les couches
                    layer_delta = np.dot(layer_deltas[-1], self.weights[j].T)
                    act_fn_deriv = self.activation_fct_deriv_map[self.activation_functions[j-1]]
                    layer_delta *= act_fn_deriv(layers[j])

                    if with_dropout:
                        layer_delta *= dropout_masks[j-1]
                    layer_deltas.append(layer_delta)

                layer_deltas = layer_deltas[::-1]   #inverse la liste (pour l'avoir dans le mm sens, comme on l'a construite à l'envers)
                
                #Mise à jour des poids
                for j in range(len(self.weights) - 1, -1, -1):  #Dans l'ordre inverse
                    self.weights[j] += self.alpha * np.dot(layers[j].T, layer_deltas[j])

            #Affichage infos d'évolution
            if (iteration % print_every_other_step == 0 or iteration == self.iterations-1) and not silent:
                msg_a = f"\r I: {iteration+1}    Error: {error/float(len(inputs))}    "
                msg_b = self.evaluate(test_inputs=test_inputs, test_labels=test_labels)
                if test_inputs is None or test_labels is None:
                    print(msg_a)
                else:
                    print(f"{msg_a} {msg_b}")

if __name__ == "__main__":
    from keras.datasets import mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images, labels = (x_train[0:1000].reshape(1000, 28*28) / 255, y_train[0:1000])
    test_images = x_test.reshape(len(x_test), 28*28) / 255
    test_labels = np.zeros((len(y_test), 10))

    one_hot_labels = np.zeros((len(labels), 10))
    for i, l in enumerate(labels):
        one_hot_labels[i][l] = 1
    labels = one_hot_labels

    for i, l in enumerate(y_test):
        test_labels[i][l] = 1

    nn = SimpleNeuralNetwork(
        layers_sizes=[28*28, 40, 10],
        activation_functions=["relu","identity"]
    )

    nn.train(
        inputs=images,
        labels=labels,
        test_inputs=test_images,
        test_labels=test_labels
    )
                

                


