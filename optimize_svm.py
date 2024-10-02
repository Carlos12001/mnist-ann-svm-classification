import wandb
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score
import time

def load_data():
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    train_data = datasets.MNIST(root="./data", train=True,
                                download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root="./data", train=False,
                               transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=int(len(train_data)), shuffle=True)
    test_loader = DataLoader(test_data, batch_size=int(len(test_data)*0.25), shuffle=True)

    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))

    train_images = train_images.view(-1, 28*28).numpy()
    train_labels = train_labels.numpy()
    test_images = test_images.view(-1, 28*28).numpy()
    test_labels = test_labels.numpy()

    return train_images, train_labels, test_images, test_labels


def objective():
    config = {
        "C": 10,
        "kernel": "rbf",
        "gamma": 0.1,
        "degree": 1
    }

    run = wandb.init(project="svm-optimization", entity="proyecto1", config=config, reinit=True)


    config = run.config
    model = SVC(C=config.C, kernel=config.kernel, gamma=config.gamma, degree=config.degree)
    # Training
    start_time = time.time()
    print("Start learning at {}".format(str(start_time)))
    model.fit(train_images, train_labels)
    end_time = time.time()
    print("Stop learning {}".format(str(end_time)))
    training_time = end_time - start_time

    # Testing
    start_time = time.time()
    print("Start testing at {}".format(str(start_time)))
    preds = model.predict(test_images)
    end_time = time.time()
    print("Stop testing {}".format(str(end_time)))
    testing_time = end_time - start_time

    # Metrics Accuracy and Recall
    accuracy = accuracy_score(test_labels, preds)
    recall = recall_score(test_labels, preds, average="macro")
    precision = precision_score(test_labels, preds, average="macro")

    print(f"Training time: {training_time}, Testing time: {testing_time}")
    print(f"Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}")
    wandb.log({
        "training_time": training_time,
        "testing_time": testing_time,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "degree": config.degree,
        "C": config.C,
        "gamma": config.gamma,
        "kernel": config.kernel
    })

    wandb.finish()
    return accuracy


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_data()
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "C": {"min": 1, "max": 50, "distribution": "uniform"},
            "kernel": {"values": ["rbf", "poly"]},
            "gamma": {"min": 0.01, "max": 1, "distribution": "uniform"},
            "degree": {"min": 1, "max": 5, "distribution": "uniform"}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project="svm-optimization", entity="proyecto1")


    start_time = time.time()
    wandb.agent(sweep_id, function=objective, count=25)
    end_time = time.time()


    # Finaliza el sweep
    wandb.finish()

    # Calcula la duración de la ?corrida
    duration = end_time - start_time

    print(f"Duration time Optimizing {duration}s")
