import numpy as np
import pandas as pd

# Crear el DataFrame vacío
df = pd.DataFrame()
impuestos = np.array([11, 20, 30, 180, 308])

# Definir los datos de entrenamiento
num_features = 5  # 400 etiquetas de cualidades
num_services = 100
num_periods = 56

# Generar datos de ejemplo
num_users = 1000
estados_impositivos = np.random.randint(0, 2, size=(num_users, num_features))
servicios_accedidos = np.random.randint(0, 2, size=(num_users, num_services))
semanas = np.random.randint(0, num_periods, size=num_users)

# Listas para acumular los datos
impuestos_list = []
impuestos_n_list = []
semanas_list = []
servicios_list = []

for usuario, servicio, semana in zip(estados_impositivos, servicios_accedidos, semanas):
    # Convertir los elementos a cadenas antes de unirlos
    impuestos_str = " ".join(map(str, impuestos * usuario))
    impuestos_n_str = " ".join(map(str, usuario))
    servicio_str = " ".join(map(str, servicio))
    
    impuestos_list.append(impuestos_str)
    impuestos_n_list.append(impuestos_n_str)
    semanas_list.append(semana)
    servicios_list.append(servicio_str)

# Crear el DataFrame con las listas acumuladas
df['impuesto'] = impuestos_list
df['impuesto_normalizado'] = impuestos_n_list
df['semana'] = semanas_list
df['servicio'] = servicios_list

df.to_excel('data.xlsx')

print(df)
