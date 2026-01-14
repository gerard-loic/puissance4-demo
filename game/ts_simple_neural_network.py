from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

class NeuralNetwork:
    def __init__(self, position:int, input_shape:tuple, layer_sizes:list, activation_functions:list, with_dropout:bool, dropout_ratio:float=0.3, learning_rate:float=0.005):
        self.position = position

        #Reproductibilité
        np.random.seed(42)
        tf.random.set_seed(42)

        #Initialisation du réseau de neurones
        self.network = Sequential()
        #Ajout des couches
        for i in range(len(layer_sizes)):
            layer_size = layer_sizes[i]
            activation_fct = activation_functions[i]

            if i == 0:
                #Première couche
                self.network.add(Dense(layer_size, activation=activation_fct, input_shape=input_shape))
            else:
                self.network.add(Dense(layer_size, activation=activation_fct))
            
            #Dropout (pas sur la dernière couche)
            if with_dropout and i < len(layer_sizes) - 1:
                self.network.add(Dropout(dropout_ratio))

        self.network.compile(
            optimizer=SGD(learning_rate=learning_rate),
            loss='mse', 
            metrics=['mae']
        )

    def save(self, file_path:str):
        self.network.save(file_path)

    def load(self, file_path: str):
        self.network = load_model(file_path)

    def forward(self, input_data:np.ndarray):
        output = self.network(input_data, training=False)
        return output
    
    def train(self, inputs:list, labels:list, test_inputs:list, test_labels:list, iterations:int=350):
        history = self.network.fit(
            inputs, 
            labels, 
            validation_data=(test_inputs, test_labels),
            epochs=iterations,
            batch_size=8, 
            verbose=0
        )
        return history

    def evaluate(self, test_inputs:list, test_labels:list):
        mae, acc = self.network.evaluate(test_inputs, test_labels, verbose=0)
        return f"Evaluation | MAE: {mae}  "

    def copy_weights_from(self, other_network):
        """Copie les poids d'un autre réseau"""
        self.network.set_weights(other_network.network.get_weights())

    def soft_update_from(self, other_network, tau=0.1):
        """Soft update depuis un autre réseau"""
        self_weights = self.network.get_weights()
        other_weights = other_network.network.get_weights()
        
        updated_weights = []
        for self_w, other_w in zip(self_weights, other_weights):
            updated_weights.append(tau * other_w + (1 - tau) * self_w)
        
        self.network.set_weights(updated_weights)