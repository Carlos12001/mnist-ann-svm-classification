# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

print("Importing packages...")
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

from softmax import Softmax

# Load the MNIST Dataset
test_data = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=True)

model = Softmax()

# Load the saved Model
model.load("mnist_softmax.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test the Modeldef test():
def test():
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)

    # Draw a colorful confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)

    # Compute precision and recall
    precision = precision_score(all_targets, all_preds, average="macro")
    recall = recall_score(all_targets, all_preds, average="macro")

    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))

    plt.show()

try:    
    test()
except KeyboardInterrupt:
    print("\nGracefully shutting down from Ctrl+C")
finally:
    # Any cleanup code here
    print("\nCleanup complete")
