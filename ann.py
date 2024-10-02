import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier import Classifier

class ANN(nn.Module, Classifier):
    '''Artificial Neural Network Classifier'''
    
    def __init__(self, input_size=784, hidden_size=[128, 64], output_size=10, activation='ReLu', filename=None):
        nn.Module.__init__(self)
        Classifier.__init__(self, "ANN")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # Remember which device we're using
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._build_model()

        if filename is not None:
            self.load(filename)

    def _build_model(self):
        layers = []
        prev_size = self.input_size
        for size in self.hidden_size:
            layers.append(nn.Linear(prev_size, size))
            if self.activation == 'ReLu':
                layers.append(nn.ReLU())
            elif self.activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif self.activation == 'PreLU':
                layers.append(nn.PReLU())
            elif self.activation == 'ELU':
                layers.append(nn.ELU())
            elif self.activation == 'SELU':
                layers.append(nn.SELU())
            else:
                raise ValueError(f"Capa de activacion no definida: {self.activation}'")
            
            prev_size = size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=1)


    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.output_size = checkpoint['output_size']
        self.activation = checkpoint['activation']

        # Reconstruir las capas
        # Reconstruir las capas
        self.model = self._build_model()

        # Cargar los parámetros del modelo
        self.model.load_state_dict(checkpoint['model'])
        self.to(self.device)
        self.eval()

    def save(self, filename):
        self.to('cpu')# It's customary to save in 'cpu' mode
        #torch.save({'model': self..state_dict()}, filename)
        checkpoint = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'activation': self.activation,
            'model': self.model.state_dict()
        }
        torch.save(checkpoint, filename)
        self.to(self.device) # but restore the mode

    def predict(self, image):
        image_tensor = torch.from_numpy(image).float().to(self.device)
        data = image_tensor.view(-1, 28 * 28).to(self.device)
        output = self(data)
        _, pred = torch.max(output, 1)
        return pred.item()
