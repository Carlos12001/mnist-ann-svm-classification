# Copyright (C) 2024 Pablo Alvarado
# EL5857 Aprendizaje Automático
# Escuela de Ingeniería Electrónica
# I Semestre 2024
# Proyecto 1

print("Importing packages...")
import torch
from torchvision import datasets, transforms
import torch.nn.functional as Fs
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, ConfusionMatrixDisplay, 
                             classification_report)

import matplotlib.pyplot as plt
from ann import ANN

test_data = datasets.MNIST(root='./data', train=False, 
                           transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_data, 
                                          batch_size=1000, shuffle=True)



print("Total images for testing: ",len(test_loader))

# Test the Modeldef test():
def test(model, device, loss_function = torch.nn.CrossEntropyLoss()):
    model.eval()
    all_preds = []   ## Predicciones
    all_targets = [] ## Las verdaderas etiquetas 
    total_loss = 0   ## Perdida
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            output = model(data)

            current_loss = loss_function(output, target)
            total_loss += current_loss.item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute confusion matrix
    conf_mat = confusion_matrix(all_targets, all_preds)

    # Draw a colorful confusion matrix using ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues)

    # Compute precision and recall
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    classification = classification_report(all_targets, all_preds, digits=3)

    # Compute loss
    loss = total_loss / len(test_loader)

    return accuracy, precision, recall, classification, loss

def define_data():
    # Define the model
    model = ANN()

    # Load the saved Model
    model.load("mnist_ann.pt")

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define loss function
    loss_function = torch.nn.CrossEntropyLoss()

    return test(model, device, loss_function)



if __name__ == "__main__":
    print("Running tests...")
    
    try:
        accuracy, precision, recall, classification, loss = define_data()
        # Print Metrics
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("Loss: {:.4f}".format(loss))

        # Print classification report
        print("Classification Report:")
        print(classification)

        # Show Confusion Matrix
        plt.show()
    except KeyboardInterrupt:
        print("\nGracefully shutting down from Ctrl+C")
    finally:
        # Any cleanup code here
        print("\nCleanup complete")
