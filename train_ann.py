print("Importing packages...")
from ann import ANN
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import precision_score, recall_score


# Start a new run
run = wandb.init(project="ann", 
                 entity = "proyecto1", 
                 name = "",
                 config = {"hidden_sizes": [512, 256, 128, 64, 32], 
                           "activation": "PreLU", 
                           "learning_rate": 0.002871827239750484,
                           "optim":"Adam",
                           'momentum': 0.475
                           }
                 )

config = run.config
# 2. Neural Network configuration
ann_config = {
    'input_size': 28 * 28,
    'hidden_size': config.hidden_sizes,
    'output_size': 10,
    'activation': config.activation
}

# ANN model creation
ann_model = ANN(**ann_config)

# Mover el modelo a la GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ann_model.to(device)

data_augmentation = True

if data_augmentation:
    # Transformaciones de aumento de datos
    data_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.3, 0.3), scale=(0.5, 1.1)),
        transforms.ToTensor()
    ])
else:
    # Sin aumento de datos
    data_transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, pin_memory=True)

if config.optim == "Adam":
    optimizer = optim.Adam(ann_model.parameters(), lr=config.learning_rate)
elif config.optim == "Nadam":
    optimizer = optim.NAdam(ann_model.parameters(), lr=config.learning_rate)
elif config.optim == "SGD":
    optimizer = optim.SGD(ann_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
elif config.optim == "RMSprop":
    optimizer = optim.RMSprop(ann_model.parameters(), lr=config.learning_rate, momentum=config.momentum)
else:
    raise ValueError(f"El optimizador especificado en 'config.optim: {config.optim}' no es válido.")

# Log gradients and model parameters
run.watch(ann_model)

log_interval=1

run.name=f'Config: {ann_config}, lr:{run.config.learning_rate}'
# Entrenamiento del modelo
for epoch in range(1, 15+1):
    
    ann_model.train() # Modo de entrenamiento
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28).to(device)  # Mover los datos de entrada a la GPU si está disponible
        target = target.to(device)  # Mover las etiquetas a la GPU si está disponible
        optimizer.zero_grad()  # Restablecer los gradientes
        output = ann_model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        with torch.no_grad():
            _, predicted = torch.max(output, 1)
            batch_precision = precision_score(target.cpu(), predicted.cpu(), average='macro', zero_division=0)
            batch_recall = recall_score(target.cpu(), predicted.cpu(), average='macro', zero_division=0)

        optimizer.step()

        if batch_idx % log_interval == 0:
        # Log metrics to visualize performance
            run.log({"loss": loss.item(), "precision": batch_precision, "recall": batch_recall, "epoch": epoch})

    print('Epoch {}: Loss: {:.6f}'.format(epoch, loss.item()))

# Guardar el modelo entrenado
ann_model.save("mnist_ann.pt")