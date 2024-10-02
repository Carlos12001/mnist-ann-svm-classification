import os
import json
import numpy as np
import wandb
import matplotlib.pyplot as plt

def is_pareto_optimal(fitness):
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
config_file_path = "data/svm_configs.json"

# Verifica si el archivo de configuraciones existe
data, hyperparameters = load_data_from_file(config_file_path)

if data is None or hyperparameters is None:
    # Si el archivo no existe, descarga los datos de wandb
    api = wandb.Api()
    runs = api.runs("proyecto1/svm-optimization")
    data = []
    hyperparameters = []
    for run in runs:
        metrics = run.history(keys=["precision", "recall"])
        for _, row in metrics.iterrows():
            data.append([row["precision"], row["recall"]])
            hyperparameters.append({
                "C": run.config.get("C", 1.0),  # Valor predeterminado: 1.0
                "kernel": run.config.get("kernel", "rbf"),  # Valor predeterminado: "rbf"
                "gamma": run.config.get("gamma", "scale"),  # Valor predeterminado: "scale"
                "degree": run.config.get("degree", 3)  # Valor predeterminado: 3
            })
    
    # Guarda los datos descargados en el archivo
    save_data_to_file(config_file_path, data, hyperparameters)

data = np.array(data)


pareto_points = is_pareto_optimal(data)
pareto_front = data[pareto_points]
pareto_hyperparameters = [hyperparameters[i] 
                          for i in range(len(hyperparameters)) if pareto_points[i]]

print("Hiperparámetros de los puntos en el frente de Pareto:")
for i, point in enumerate(pareto_front):
    print(f"Punto {i+1}: Precision = {point[0]}, Recall = {point[1]}")
    print(f"  C = {pareto_hyperparameters[i]['C']}")
    print(f"  kernel = {pareto_hyperparameters[i]['kernel']}")
    print(f"  gamma = {pareto_hyperparameters[i]['gamma']}")
    print(f"  degree = {pareto_hyperparameters[i]['degree']}")



plt.scatter(data[:, 0], data[:, 1], color="blue", label="Datos")
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color="red", label="Frente de Pareto")
plt.title("Frente de Pareto - Precisión vs. Recall")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend()
plt.xlim(0.9, 1)  
plt.ylim(0.9, 1)  
plt.show()