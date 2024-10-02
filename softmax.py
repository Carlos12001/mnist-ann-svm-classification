# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

import torch.nn.functional as F
import torch

from classifier import Classifier

# Define the Softmax Classifier
class Softmax(torch.nn.Module,Classifier):
    """SoftmaxClassifier"""
    
    def __init__(self,filename=None):
        torch.nn.Module.__init__(self)
        Classifier.__init__(self,"SOFTMAX")
        
        self.linear = torch.nn.Linear(784, 10)

        # Remember which device we're using
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            

        if filename is not None:
            self.load(filename)

        
    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=1)
    
    def load(self,filename):
        # Load the classifier model

        self.load_state_dict(torch.load(filename,map_location=self.device))
        self.to(self.device)
        self.eval()

    def save(self,filename):
        # Save the classifier model
        self.to("cpu") # It's customary to save in "cpu" mode
        torch.save(self.state_dict(), filename)
        self.to(self.device) # but restore the mode

    def predict(self, image):
        # Convert the numpy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image).float().to(self.device)
        
        # Predict the class label for the given image
        data = image_tensor.view(-1, 28 * 28).to(self.device)
        output = self(data)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item()
