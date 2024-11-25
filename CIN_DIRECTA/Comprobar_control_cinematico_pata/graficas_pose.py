import matplotlib.pyplot as plt
import numpy as np

# Leer el archivo de datos
data = np.loadtxt('/tmp/robot_pose_data.txt', delimiter='\t', skiprows=1)

# Extraer columnas de interés
time = data[:, 0]  # Tiempo
joint_positions = data[:, 1:4]  # Posiciones articulares (3 columnas)
position = data[:, 4:7]  # Posición (x, y, z)
orientation = data[:, 7:11]  # Orientación (cuaternión)
error_position = data[:, 11:14]  # Error en posición
error_orientation = data[:, 14:18]  # Error en orientación

# Graficar las posiciones articulares
plt.figure()
plt.plot(time, joint_positions)
plt.title('Joint Positions vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Joint Positions (rad)')
plt.legend(['Joint 1', 'Joint 2', 'Joint 3'])

# Graficar la posición (x, y, z)
plt.figure()
plt.plot(time, position)
plt.title('Position (x, y, z) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend(['x', 'y', 'z'])

# Graficar el error en la posición
plt.figure()
plt.plot(time, error_position)
plt.title('Position Error (x, y, z) vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Error (m)')
plt.legend(['Error in x', 'Error in y', 'Error in z'])

plt.show()
