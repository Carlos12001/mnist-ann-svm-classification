
import torch
from svm import SVM
from torchvision import datasets, transforms
import numpy as np
import time
import wandb

# Configurar Weights & Biases
wandb.init(project="svm", entity="proyecto1")

# Configurar SVM
svm_config = {
    "C": 42.98620007644219,
    "kernel": "rbf",
    "gamma": 0.021332766431507475
}

# Configurar wandb para rastrear configuraciones
wandb.config.update(svm_config)

# Crear el modelo SVM
model = SVM(**svm_config)

# Cargar los datos de entrenamiento de MNIST
train_data = datasets.MNIST(root="./data", train=True, download=True, 
                            transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train_data, 
                                           batch_size=len(train_data),
                                           shuffle=True)


# Preprocesamiento de los datos de entrenamiento
images, labels = next(iter(train_loader))
images = images.numpy().reshape(len(train_data), -1)
labels = labels.numpy()

# Entrenamiento del modelo
start_time = time.time()
print("Start learning at {}".format(str(start_time)))
model.fit(images, labels)
end_time = time.time() 
print("Stop learning {}".format(str(end_time)))
elapsed_time= end_time - start_time
print("Elapsed learning {}".format(str(elapsed_time)))

# Guardar el modelo entrenado
model.save("mnist_svm.pkl")

# Registrar el tiempo de entrenamiento en wandb
wandb.log({"training_time": elapsed_time})

# Cerrar Weights & Biases   
wandb.finish()
