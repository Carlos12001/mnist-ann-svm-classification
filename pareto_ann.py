import os
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt

def is_pareto_optimal(fitness):
    """ction(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/carlos/y/envs/p1/lib/
    Find the pareto-optimal points
    :param fitness: An (n_points, n_fitness) array (each data in a row)
    :return: A (n_points,) boolean array, indicating whether each point is Pareto optimal
    """
    is_efficient = np.ones(fitness.shape[0], dtype=bool)
    for i, c in enumerate(fitness):
        if is_efficient[i]:
            # Marca como no eficientes aquellos peores en alguna dimensión
            is_efficient[is_efficient] = np.any(fitness[is_efficient] > c, axis=1)
            is_efficient[i] = True  # Asegura mantener el punto actual como eficiente
    return is_efficient


def load_data_from_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["data"], data["hyperparameters"]
    else:
        return None, None

def save_data_to_file(file_path, data, hyperparameters):
    with open(file_path, "w") as file:
        json.dump({"data": data, "hyperparameters": hyperparameters}, file)

# Ruta del archivo de configuraciones guardadas
config_file_path = "data/ann_configs.json"

# Verifica si el archivo de configuraciones existe
data, hyperparameters = load_data_from_file(config_file_path)

if data is None or hyperparameters is None:
    api = wandb.Api()
    runs = api.runs("proyecto1/ann_optimize_ultimate")
    data = []
    hyperparameters = []
    for run in runs:
        precision = run.summary.get('precision', 0)
        recall = run.summary.get('recall', 0)
        data.append([precision, recall])
        hyperparameters.append({
            "hidden_size": run.config.get("hidden_size", [256, 128, 64]),
            "learning_rate": run.config.get("learning_rate", 0.001),
            "momentum": run.config.get("momentum", 0.92),
            "epochs": run.config.get("epochs", 15),
            "activation_layer": run.config.get("activation_layer", "ReLU"),
            "optim": run.config.get("optim", "Adam")
        })
    
    save_data_to_file(config_file_path, data, hyperparameters)

data = np.array(data)
fitness = data

is_pareto = is_pareto_optimal(fitness)

pareto_hyperparameters = [hyperparameters[i] 
                          for i in range(len(hyperparameters)) if is_pareto[i]]


# Imprime los hiperparámetros de los puntos en el frente de Pareto
print("Hiperparámetros de los puntos en el frente de Pareto:")
for i, params in enumerate(pareto_hyperparameters):
    print(f"Punto {i+1}:")
    for key, value in params.items():
        print(f"  {key}: {value}")


# Visualización de los resultados
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Puntos del sweep')
plt.scatter(data[is_pareto, 0], data[is_pareto, 1], color='red', label='Frente de Pareto')
plt.xlabel('Precisión')
plt.ylabel('Exhaustividad')
plt.title('Espacio Precisión/Exhaustividad')
plt.legend()
plt.grid(True)
plt.xlim(0.9, 1)  
plt.ylim(0.9, 1)  
plt.show()