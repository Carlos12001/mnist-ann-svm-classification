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

# Load the saved Model
model = SVM()
model.load("mnist_svm.pkl")

# Load the new dataset
data = np.load("data/Data.npy")
labels = np.load("data/Labels.npy")

def test():
  print("Testing the Model...")
  all_preds = []
  all_targets = []
  
  for img, target in zip(data, labels):
    pred_digit = model.predict(img*255)  # Make the prediction
    all_preds.append(pred_digit)  # Save the prediction
    all_targets.append(target)  # Save the target

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