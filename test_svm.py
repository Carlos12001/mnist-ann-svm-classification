# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

print("Importing packages...")

from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from svm import SVM

# Load the MNIST Dataset
test_data = datasets.MNIST(root="./data", train=False,
                           transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                          shuffle=True)

model = SVM()
# Load the saved Model
model.load("mnist_svm.pkl")

# Test the Model
def test():
    print("Testing the Model...")
    all_preds = []
    all_targets = []
    total_batches = len(test_loader)
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader, start=1):
            progress_bar = tqdm(enumerate(data), total=len(data),
                                desc=f"Batch {i}/{total_batches}",
                                leave=True,
                                unit="img",
                                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")
            data = data.squeeze(1).numpy()  # Remove channel color and convert to numpy
            for index, img in progress_bar:
                pred_digit = model.predict(255*img)  # Make the prediction
                all_preds.append(pred_digit)     # Save the prediction

            all_targets.extend(target.numpy())  # Save the targets
            progress_bar.close()

    # Compute confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)

    # Draw a colorful confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)

    # Compute precision and recall
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="macro")
    recall = recall_score(all_targets, all_preds, average="macro")

    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))

    # Print classification report
    print("Classification Report:")
    print(classification_report(all_targets, all_preds, digits=3))

    plt.show()

try:
    test()
except KeyboardInterrupt:
    print("\nGracefully shutting down from Ctrl+C")
finally:
    # Any cleanup code here
    print("\nCleanup complete")
