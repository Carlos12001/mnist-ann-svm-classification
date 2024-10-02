print("Importing packages...")
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import datetime as dt
from ann import ANN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score

# Definir device como variable global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(batch_size=1000 ,data_augmentation=True):
    if data_augmentation:
        # Transformaciones de aumento de datos
        data_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.7, 1.1)),
            transforms.ToTensor()
        ])
    else:
        # Sin aumento de datos
        data_transform = transforms.ToTensor()

    train_data = datasets.MNIST(root="./data", train=True, 
                                download=True, transform=data_transform)
    
    #test_data = datasets.MNIST(root="./data", train=False, 
    #                           transform=transforms.ToTensor())

    # Mover los datos de entrenamiento y prueba al dispositivo GPU si está disponible
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    #test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)




    data = torch.load('data/Data.pt')
    labels = torch.load('data/Labels.pt')

    # Convertir los tensores en un dataset de PyTorch
    dataset = torch.utils.data.TensorDataset(data, labels)

    # Crear un DataLoader para manejar los lotes de datos
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)




    return train_loader, test_loader

def evaluate_model(model, test_loader):
    model.eval()  # Establece el modelo en modo de evaluación

    correct = 0
    total = 0
    with torch.no_grad():  # Desactiva el cálculo de gradientes durante la evaluación
        for data, target in test_loader:
            data = data.view(-1, 28*28).to(device)  # Mover los datos de entrada a la GPU si está disponible
            target = target.to(device)  # Mover las etiquetas a la GPU si está disponible
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
            epoch_precision = precision_score(target.cpu(), predicted.cpu(), average='macro', zero_division=0)
            epoch_recall = recall_score(target.cpu(), predicted.cpu(), average='macro', zero_division=0)


    accuracy = correct / total
    return accuracy , epoch_precision,  epoch_recall


def objective():
    
    run = wandb.init(project="ann_optimize_ultimate", entity="proyecto1", name="ann", reinit=True)
    config = run.config
    print(config)
    ann_config = {
        "input_size": 28 * 28,
        'hidden_size': config.hidden_size,
        'output_size': 10,
        'activation': config.activation_layer
    }

    ann_model = ANN(**ann_config)
    ann_model.to(device)

    print("Optimizador usado: ", config.optim)
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

    train_loader, test_loader = load_data()

    # Training
    start_time = dt.datetime.now()
    print("Start learning at {}".format(str(start_time)))

    log_interval = 1
    run.name = f'Config: {config}'
    for epoch in range(1, config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(-1, 28*28).to(device)  
            target = target.to(device)  
            optimizer.zero_grad()  # Restablece los gradientes
            output = ann_model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                wandb.log({"loss": loss.item()})
        
        print('Epoch {}: Loss: {:.6f}'.format(epoch, loss.item()))

    end_time = dt.datetime.now()
    print("Stop learning at {}".format(str(end_time)))
    training_time = end_time - start_time

    # Testing
    start_time = dt.datetime.now()
    print("Start testing at {}".format(str(start_time)))

    # Prueba de evaluación
    accuracy , epoch_precision,  epoch_recall = evaluate_model(ann_model, test_loader)
    assert 0 <= accuracy <= 1

    end_time = dt.datetime.now()
    print("Stop testing at {}".format(str(end_time)))
    testing_time = end_time - start_time

    print(f"Training time: {training_time}, Testing time: {testing_time}")

    # Log metrics
    wandb.log({
        "training_time": training_time.total_seconds(),
        "testing_time": testing_time.total_seconds(),
        "hidden_size": config.hidden_size,
        "learning_rate": config.learning_rate,
        "epochs": config.epochs,
        "accuracy":accuracy,
        "activation_layer": config.activation_layer,
        "optim": config.optim,
        "momentum": config.momentum,
        "precision": epoch_precision,
        "recall": epoch_recall,
        "loss": loss.item()
    })
    wandb.finish()

    return accuracy

if __name__ == "__main__":
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "hidden_size": {
                "values": [
                    [256, 128, 64],        # Tres capas ocultas
                    [512, 256, 128, 64],   # Cuatro capas ocultas
                    [512, 256, 128, 64, 32],  # Cinco capas ocultas
                    [512, 256, 128, 64, 32, 16]  # Seis capas ocultas
                ]
            },
            "learning_rate": {"min": 1e-6, "max": 0.01, "distribution": "uniform"}, 
            "momentum": {"values": [0.92/2, 0.95/2, 0.99/2]},  # Agrega el parámetro de momentum
            "epochs": {"values": [15]},
            "activation_layer": {"values": ["ReLu", "ELU", "PreLU", "SELU", "sigmoid"]},
            "optim": {"values": ["Adam", "Nadam"]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project='ann_optimize_ultimate', entity="proyecto1")
    wandb.agent(sweep_id, function=objective, count=25)
