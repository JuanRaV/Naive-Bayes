import numpy as np

# Definir los datasets de entrenamiento y prueba
X_train = {
    'Outlook': ['overcast', 'overcast', 'rainy', 'rainy', 'rainy', 'rainy', 'sunny', 'sunny', 'sunny', 'sunny'],
    'Temp': [64, 72, 70, 65, 75, 71, 85, 80, 72, 75],
    'Humidity': [65, 90, 96, 70, 80, 91, 85, 90, 95, 70],
    'Windy': [True, True, False, True, False, True, False, True, False, True]
}
y_train = ['no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'no']

X_test = {
    'Outlook': ['overcast', 'overcast', 'rainy', 'sunny'],
    'Temp': [83, 81, 68, 69],
    'Humidity': [86, 75, 80, 70],
    'Windy': [False, False, False, False]
}
y_test = ['no', 'yes', 'no', 'yes']

# Función para calcular la media y la desviación estándar
def calculate_mean_std(values, labels, target_label):
    # Filtrar los valores según la etiqueta objetivo
    filtered_values = [values[i] for i in range(len(values)) if labels[i] == target_label]
    # Eliminar valores duplicados
    filtered_values = list(set(filtered_values))
    # Calcular la media
    mean = np.mean(filtered_values)
    # Calcular la desviación estándar muestral (ddof=1)
    std = np.std(filtered_values, ddof=1)
    return mean, std

# Calcular medias y desviaciones estándar para Temp y Humidity en el conjunto de entrenamiento
mean_temp_yes, std_temp_yes = calculate_mean_std(X_train['Temp'], y_train, 'yes')
mean_temp_no, std_temp_no = calculate_mean_std(X_train['Temp'], y_train, 'no')
mean_humidity_yes, std_humidity_yes = calculate_mean_std(X_train['Humidity'], y_train, 'yes')
mean_humidity_no, std_humidity_no = calculate_mean_std(X_train['Humidity'], y_train, 'no')

# Calcular las densidades de probabilidad para cada atributo y posible clase
def calculate_density(value, mean, std):
    return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean) ** 2) / (2 * std ** 2))

# Calcular las probabilidades de cada clase
def calculate_probabilities(temp, humidity, class_probs, means, stds):
    temp_density_yes = calculate_density(temp, means['yes']['temp'], stds['yes']['temp'])
    humidity_density_yes = calculate_density(humidity, means['yes']['humidity'], stds['yes']['humidity'])
    temp_density_no = calculate_density(temp, means['no']['temp'], stds['no']['temp'])
    humidity_density_no = calculate_density(humidity, means['no']['humidity'], stds['no']['humidity'])
    # print("class", class_probs)
    prob_yes = class_probs['yes'] * temp_density_yes * humidity_density_yes * class_probs['yes'] * 0.7
    prob_no = class_probs['no'] * temp_density_no * humidity_density_no * class_probs['no'] * 0.3
    
    total_prob = prob_yes + prob_no
    return prob_yes / total_prob, prob_no / total_prob


class_probs = {'yes': y_train.count('yes') / len(y_train), 'no': y_train.count('no') / len(y_train)}
means = {
    'yes': {'temp': mean_temp_yes, 'humidity': mean_humidity_yes},
    'no': {'temp': mean_temp_no, 'humidity': mean_humidity_no}
}
stds = {
    'yes': {'temp': std_temp_yes, 'humidity': std_humidity_yes},
    'no': {'temp': std_temp_no, 'humidity': std_humidity_no}
}

# Calcular las probabilidades para la instancia nueva en el conjunto de entrenamiento
prob_yes_train, prob_no_train = calculate_probabilities(69, 85, class_probs, means, stds)

# Calcular el accuracy basado en las probabilidades calculadas
accuracy_train = prob_yes_train if prob_yes_train > prob_no_train else prob_no_train

print(f"Naive Bayes Model - Calculated Accuracy: {accuracy_train}")
