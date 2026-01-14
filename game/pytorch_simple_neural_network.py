import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from board import Board, PLAYER_ONE, PLAYER_TWO, EMPTY_CELL

class TsSimpleNeuralNetwork:
    def __init__(self, input_shape: tuple, layer_sizes: list, activation_functions: list, 
                 with_dropout: bool, loss_fct: str = 'mse', metrics: list = ['mae'], 
                 dropout_ratio: float = 0.3, learning_rate: float = 0.005):
        
        # Reproductibilité
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mapping des fonctions d'activation
        activation_map = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=-1),
            'linear': nn.Identity()
        }
        
        # Construction du réseau
        layers = []
        input_size = input_shape[0]
        
        for i in range(len(layer_sizes)):
            layer_size = layer_sizes[i]
            activation_fct = activation_functions[i].lower()
            
            # Couche linéaire
            layers.append(nn.Linear(input_size, layer_size))
            
            # Activation
            if activation_fct in activation_map:
                layers.append(activation_map[activation_fct])
            
            # Dropout (pas sur la dernière couche)
            if with_dropout and i < len(layer_sizes) - 1:
                layers.append(nn.Dropout(dropout_ratio))
            
            input_size = layer_size
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = self._get_loss_function(loss_fct)
        self.metrics = metrics
        
    def _get_loss_function(self, loss_fct: str):
        """Retourne la fonction de perte appropriée"""
        loss_map = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'categorical_crossentropy': nn.CrossEntropyLoss(),
            'binary_crossentropy': nn.BCELoss()
        }
        return loss_map.get(loss_fct.lower(), nn.MSELoss())
    
    def forward(self, input_data: np.ndarray):
        """Forward pass pour une seule entrée ou batch"""
        self.network.eval()
        with torch.no_grad():
            if not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, dtype=torch.float32)
            input_data = input_data.to(self.device)
            output = self.network(input_data)
            return output.cpu().numpy()
    
    def forward_batch(self, input_data: np.ndarray):
        """Forward pass pour un batch (identique à forward en PyTorch)"""
        return self.forward(input_data)
    
    def save(self, file_path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, file_path)
        print(f"Model saved in {file_path}")
    
    def load(self, file_path: str):
        """Charge le modèle"""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {file_path}")
    
    def train(self, inputs: np.ndarray, labels: np.ndarray, 
              test_inputs: np.ndarray = None, test_labels: np.ndarray = None,
              iterations: int = 350, batch_size: int = 8, 
              verbose: bool = False, callbacks: list = None):
        """Entraîne le réseau"""
        
        # Conversion en tenseurs
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).to(self.device)
        
        # Gestion des labels pour CrossEntropyLoss (besoin d'indices de classe)
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            if len(labels.shape) > 1 and labels.shape[1] > 1:
                labels_tensor = torch.argmax(labels_tensor, dim=1)
        
        # Dataset et DataLoader
        dataset = torch.utils.data.TensorDataset(inputs_tensor, labels_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Données de test
        has_validation = test_inputs is not None and test_labels is not None
        if has_validation:
            test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32).to(self.device)
            test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(self.device)
            if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
                    test_labels_tensor = torch.argmax(test_labels_tensor, dim=1)
        
        # Historique
        history = {
            'loss': [],
            'val_loss': [] if has_validation else None,
            'accuracy': [] if 'accuracy' in self.metrics else None,
            'val_accuracy': [] if has_validation and 'accuracy' in self.metrics else None
        }
        
        # Boucle d'entraînement
        for epoch in range(iterations):
            self.network.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_inputs, batch_labels in dataloader:
                # Forward
                outputs = self.network(batch_inputs)
                loss = self.loss_fn(outputs, batch_labels)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calcul accuracy si nécessaire
                if 'accuracy' in self.metrics:
                    if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                        predicted = torch.argmax(outputs, dim=1)
                        correct += (predicted == batch_labels).sum().item()
                    else:
                        predicted = (outputs > 0.5).float()
                        correct += (predicted == batch_labels).sum().item()
                    total += batch_labels.size(0)
            
            # Moyenne de la perte
            avg_loss = epoch_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            # Accuracy
            if 'accuracy' in self.metrics and total > 0:
                accuracy = correct / total
                history['accuracy'].append(accuracy)
            
            # Validation
            if has_validation:
                val_loss, val_acc = self._validate(test_inputs_tensor, test_labels_tensor)
                history['val_loss'].append(val_loss)
                if val_acc is not None:
                    history['val_accuracy'].append(val_acc)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, history)
            
            # Verbose
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{iterations} - loss: {avg_loss:.4f}"
                if 'accuracy' in self.metrics and total > 0:
                    msg += f" - accuracy: {accuracy:.4f}"
                if has_validation:
                    msg += f" - val_loss: {val_loss:.4f}"
                    if val_acc is not None:
                        msg += f" - val_accuracy: {val_acc:.4f}"
                print(msg)
        
        return history
    
    def _validate(self, test_inputs, test_labels):
        """Validation sur les données de test"""
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(test_inputs)
            loss = self.loss_fn(outputs, test_labels).item()
            
            accuracy = None
            if 'accuracy' in self.metrics:
                if isinstance(self.loss_fn, nn.CrossEntropyLoss):
                    predicted = torch.argmax(outputs, dim=1)
                    accuracy = (predicted == test_labels).float().mean().item()
                else:
                    predicted = (outputs > 0.5).float()
                    accuracy = (predicted == test_labels).float().mean().item()
            
            return loss, accuracy
    
    def evaluate(self, test_inputs: np.ndarray, test_labels: np.ndarray):
        """Évalue le modèle"""
        test_inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32).to(self.device)
        test_labels_tensor = torch.tensor(test_labels, dtype=torch.float32).to(self.device)
        
        if isinstance(self.loss_fn, nn.CrossEntropyLoss):
            if len(test_labels.shape) > 1 and test_labels.shape[1] > 1:
                test_labels_tensor = torch.argmax(test_labels_tensor, dim=1)
        
        self.network.eval()
        with torch.no_grad():
            outputs = self.network(test_inputs_tensor)
            loss = self.loss_fn(outputs, test_labels_tensor).item()
            
            mae = None
            if 'mae' in self.metrics:
                mae = torch.abs(outputs - test_labels_tensor).mean().item()
            
            return f"Evaluation | MAE: {mae}" if mae else f"Evaluation | Loss: {loss}"
    
    def copy_weights_from(self, other_network):
        """Copie les poids d'un autre réseau"""
        self.network.load_state_dict(other_network.network.state_dict())
    
    def soft_update_from(self, other_network, tau=0.1):
        """Soft update depuis un autre réseau"""
        for target_param, source_param in zip(self.network.parameters(), 
                                               other_network.network.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )


if __name__ == "__main__":
    from torchvision import datasets, transforms
    
    # Chargement MNIST avec torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Sous-ensemble pour test
    images = torch.stack([train_dataset[i][0] for i in range(1000)])
    labels = torch.tensor([train_dataset[i][1] for i in range(1000)])
    
    test_images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
    test_labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])
    
    # One-hot encoding
    one_hot_labels = torch.zeros((len(labels), 10))
    for i, l in enumerate(labels):
        one_hot_labels[i][l] = 1
    
    test_one_hot = torch.zeros((len(test_labels), 10))
    for i, l in enumerate(test_labels):
        test_one_hot[i][l] = 1
    
    # Création du réseau
    nn_model = TsSimpleNeuralNetwork(
        input_shape=(28*28,),
        layer_sizes=[40, 10],
        activation_functions=['relu', 'softmax'],
        with_dropout=True,
        loss_fct='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement
    nn_model.train(
        inputs=images.numpy(),
        labels=one_hot_labels.numpy(),
        test_inputs=test_images.numpy(),
        test_labels=test_one_hot.numpy(),
        verbose=True
    )