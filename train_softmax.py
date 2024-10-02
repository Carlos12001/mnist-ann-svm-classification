# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

print("Importing packages...")
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

from softmax import Softmax

# Load the MNIST Dataset
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Softmax().to(device)
optimizer = optim.Adam(model.parameters())

# Use CrossEntropyLoss as the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Training the Model
def train(epoch):
    model.train() # start training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28).to(device) # Flatten the images
        target = target.to(device)
        optimizer.zero_grad() # reset gradient tensors
        output = model(data)  # propagate the data
        loss = loss_fn(output, target) # and compute loss
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, 20):
    train(epoch)

# Save the Model
model.save("mnist_softmax.pt")
    
