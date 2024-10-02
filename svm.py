# Importar las bibliotecas necesarias
import numpy as np
from sklearn.svm import SVC
import joblib
from classifier import Classifier

class SVM(Classifier):
    """Support Vector Machine Classifier"""
    
    def __init__(self,filename=None, C=10.0,kernel="linear",gamma="scale",
                 degree=3):
        # Set the classifier name
        super().__init__("SVM")
        # Create the classifier
        self.model = SVC(C=C,kernel=kernel,gamma=gamma,degree=degree)

        if filename is not None:
            self.load(filename)


    def fit(self, Xtrain, Ytrain):
        self.model.fit(Xtrain, Ytrain)

    def load(self, filename):
        self.model = joblib.load(filename)
    
    def save(self, filename):
        joblib.dump(self.model, filename)
    
    def predict(self, image):
        # Predecir la clase label para la imagen dada
        return self.model.predict((image.reshape(1, -1))/255)[0]
    
    def get_params(self):
        return self.model.get_params()
    
    def get_model(self):
        return self.model
