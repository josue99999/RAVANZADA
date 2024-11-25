#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Función para leer los datos de un archivo
def read_data(filename):
    data = np.loadtxt(filename)
    return data

# Leer los datos de los tres archivos
pfr_des_data = read_data("/tmp/pfr_des.txt")  # Datos de la posición deseada
q_data = read_data("/tmp/q.txt")              # Datos de las configuraciones articulares
pfr_data = read_data("/tmp/pfr.txt")          # Datos de la posición actual

# Crear la figura y los ejes para los gráficos
fig, axs = plt.subplots(2, 1, figsize=(10, 12))

# Gráfico 1: Posición deseada (pfr_des) y posición real (pfr) en una sola gráfica
axs[0].plot(pfr_des_data[:, 0], pfr_des_data[:, 1], label='pfr_des_x', linestyle='--')
axs[0].plot(pfr_des_data[:, 0], pfr_des_data[:, 2], label='pfr_des_y', linestyle='--')
axs[0].plot(pfr_des_data[:, 0], pfr_des_data[:, 3], label='pfr_des_z', linestyle='--')

axs[0].plot(pfr_data[:, 0], pfr_data[:, 1], label='pfr_x')
axs[0].plot(pfr_data[:, 0], pfr_data[:, 2], label='pfr_y')
axs[0].plot(pfr_data[:, 0], pfr_data[:, 3], label='pfr_z')

axs[0].set_title('Posición Deseada vs Posición Real')
axs[0].set_xlabel('Tiempo (s)')
axs[0].set_ylabel('Posición (m)')
axs[0].legend()
axs[0].grid(True)

# Gráfico 2: Configuración articular (q)
for i in range(1, q_data.shape[1]):
    axs[1].plot(q_data[:, 0], q_data[:, i], label=f'q{i}')
axs[1].set_title('Configuración Articular (q)')
axs[1].set_xlabel('Tiempo (s)')
axs[1].set_ylabel('Ángulo/Posición')
axs[1].legend()
axs[1].grid(True)

# Mostrar los gráficos
plt.tight_layout()
plt.show()
