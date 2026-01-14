from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

class TsSimpleNeuralNetwork:
    def __init__(self, input_shape:tuple, layer_sizes:list, activation_functions:list, with_dropout:bool, loss_fct:str='mse', metrics:list=['mae'], dropout_ratio:float=0.3, learning_rate:float=0.005):
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
            loss=loss_fct, 
            metrics=metrics
        )

        # ✅ Compile la fonction forward en graph pour la vitesse
        self._forward_graph = tf.function(self._forward_impl, 
                                          reduce_retracing=True,
                                          experimental_relax_shapes=True)
    def _forward_impl(self, input_data):
        """Implémentation interne compilée en graph"""
        return self.network(input_data, training=False)

    def forward(self, input_data:np.ndarray):

        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)
        
        # ✅ Utilise la version compilée en graph
        output = self._forward_graph(input_data)
        return output.numpy()
    
    def forward_batch(self, input_data:np.ndarray):
        if not isinstance(input_data, tf.Tensor):
            input_data = tf.constant(input_data, dtype=tf.float32)
        
        # ✅ Utilise la version compilée en graph (même que forward)
        output = self._forward_graph(input_data)
        return output.numpy()
    

    def save(self, file_path:str):
        self.network.save(file_path)
        print(f"Model saved in {file_path}")

    def load(self, file_path: str):
        self.network = load_model(file_path)
        print(f"Model loaded from {file_path}")

    
    def train(self, inputs:list, labels:list, test_inputs:list=None, test_labels:list=None, iterations:int=350, verbose:bool=False, callbacks:list=None):
        if test_inputs != None and test_labels != None:
            history = self.network.fit(
                inputs, 
                labels, 
                validation_data=(test_inputs, test_labels),
                epochs=iterations,
                batch_size=8, 
                verbose=verbose,
                callbacks=callbacks
            )
        else:
            history = self.network.fit(
                inputs, 
                labels, 
                epochs=iterations,
                batch_size=8, 
                verbose=verbose,
                callbacks=callbacks
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

    nn = TsSimpleNeuralNetwork(
        input_shape=(28*28,),
        layer_sizes=[40, 10],
        activation_functions=['relu', 'softmax'],
        with_dropout=True,
        loss_fct='categorical_crossentropy',
        metrics=['accuracy']
    )

    nn.train(
        inputs=images,
        labels=labels,
        test_inputs=test_images,
        test_labels=test_labels,
        verbose=True
    )
                

                
