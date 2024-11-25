import matplotlib.pyplot as plt
import numpy as np

# Leer los datos de los archivos
xcurrent_data = np.loadtxt('/tmp/xcurrent.txt')  # Posición actual del efector final
xdesired_data = np.loadtxt('/tmp/xdesired.txt')  # Posición deseada del efector final
q_data = np.loadtxt('/tmp/q.txt')              # Ángulos articulares

# Crear una figura para ambas gráficas
plt.figure(figsize=(12, 10))

# ---- Subplot 1: Posiciones del efector final ----
plt.subplot(2, 1, 1)
# Posición X
plt.plot(xcurrent_data[:, 0], label='X Actual', color='red')
plt.plot(xdesired_data[:, 0], label='X Deseado', color='green', linestyle='dashed')
# Posición Y
plt.plot(xcurrent_data[:, 1], label='Y Actual', color='blue')
plt.plot(xdesired_data[:, 1], label='Y Deseado', color='cyan', linestyle='dashed')
# Posición Z
plt.plot(xcurrent_data[:, 2], label='Z Actual', color='orange')
plt.plot(xdesired_data[:, 2], label='Z Deseado', color='purple', linestyle='dashed')

plt.title('Posiciones del Efector Final')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

# ---- Subplot 2: Ángulos articulares ----
plt.subplot(2, 1, 2)
joint_labels = ['HipX', 'HipY', 'Knee']
for i in range(q_data.shape[1]):
    plt.plot(q_data[:, i], label=f'Joint {joint_labels[i]}')

plt.title('Ángulos Articulares')
plt.xlabel('Iteration')
plt.ylabel('Joint Angles (rad)')
plt.legend()
plt.grid()

# Ajustar el diseño para evitar solapamientos
plt.tight_layout()

# Mostrar las gráficas
plt.show()
