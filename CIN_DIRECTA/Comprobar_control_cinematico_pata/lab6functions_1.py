import numpy as np
from copy import copy
from pyquaternion import Quaternion

pi = np.pi


def Trotx(angle):
    """
    Homogeneous transformation matrix from a rotation about x

    """
    T = np.eye(4)
    ca = np.cos(angle); sa = np.sin(angle)
    R = np.array([[1., 0., 0.],
                  [0., ca, -sa],
                  [0., sa, ca]])
    T[0:3,0:3] = R
    return T


def Troty(angle):
    """
    Homogeneous transformation matrix from a rotation about y

    """
    T = np.eye(4)
    ca = np.cos(angle); sa = np.sin(angle)
    R = np.array([[ ca, 0., sa],
                  [ 0., 1., 0.],
                  [-sa, 0., ca]])
    T[0:3,0:3] = R
    return T


def Trotz(angle):
    """
    Homogeneous transformation matrix from a rotation about z

    """
    T = np.eye(4)
    ca = np.cos(angle); sa = np.sin(angle)
    R = np.array([[ ca, -sa, 0.],
                  [ sa,  ca, 0.],
                  [ 0.,  0., 1.]])
    T[0:3,0:3] = R
    return T

def Ttransl(d):
    """
    Homogeneous transformation matrix for a translation d

    """
    T = np.eye(4)
    T[0,3] = d[0]
    T[1,3] = d[1]
    T[2,3] = d[2]
    return T



def fkine(q):

    L1 = 0.08
    L2 = 0.18
    L3 = 0.2
    L = 0.350
    W = 0.110
    # Matrices DH (completar)
    T01 = np.dot(Ttransl([-L / 2, W / 2, 0.0]), Trotx(q[0]))
    T12 = np.dot(Ttransl([0.0, L1, 0.0]), Troty(-q[1]))
    T23 = np.dot(Ttransl([0.0, 0.0, -L2]), Troty(-q[2]))
    T34 = Ttransl([0.0, 0.0, -L3])

    T = T01 @ T12 @ T23 @ T34

    return T



def jacobian(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion de un brazo robotico de n grados de libertad. 
    Retorna una matriz de 3xn y toma como entrada el vector de configuracion articular 
    q=[q1, q2, q3, ..., qn]
    """
    # Crear una matriz 3x8
    J = np.zeros((3,3))

    # Transformacion homogenea inicial (usando q)
    T = fkine(q)
    
    # Iteracion para la derivada de cada columna
    for i in range(3):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta

        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine(dq)

        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0,i] = (dT[0,3] - T[0,3])/delta #derivadas de x
        J[1,i] = (dT[1,3] - T[1,3])/delta #derivadas de y
        J[2,i] = (dT[2,3] - T[2,3])/delta #derivadas de z
    return J



def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analítico para posición y orientación utilizando cuaterniones.
    Retorna una matriz de 7xn y toma como entrada el vector de configuración articular q.
    
    Parámetros:
    q -- Vector de configuración articular (tamaño n)
    delta -- Incremento pequeño para diferencias finitas (default 0.0001)
    
    Retorno:
    J -- Jacobiano de posición y orientación (matriz 7xn)
    """
    n = q.size
    J = np.zeros((7, n))  # Inicializar matriz Jacobiana 7xn

    # Transformación inicial
    T = fkine(q)
    x = TF2xyzquat(T)  # Convertir transformación a vector de posición y orientación

    for i in range(n):
        # Crear una copia de la configuración articular
        dq = copy(q)
        
        # Incrementar la articulación i-ésima con un pequeño delta
        dq[i] += delta
        
        # Transformación con la configuración modificada
        Td = fkine(dq)
        xd = TF2xyzquat(Td)
        
        # Aproximación de la derivada para la posición
        J[0:3, i] = (xd[0:3] - x[0:3]) / delta  # Derivadas de x, y, z
        
        # Aproximación de la derivada para la orientación (cuaterniones)
        quat_diff = Quaternion(xd[3:7]) * Quaternion(x[3:7]).inverse
        J[3:7, i] = np.array([quat_diff.w, quat_diff.x, quat_diff.y, quat_diff.z]) / delta

    return J



def TF2xyzquat(T):
    """
    Convert a homogeneous transformation matrix into the a vector containing the
    pose of the robot.

    Input:
    T -- A homogeneous transformation
    Output:
    X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
            is Cartesian coordinates and the last part is a quaternion
    """
    quat = Quaternion(matrix=T[0:3,0:3])
    return np.array([T[0,3], T[1,3], T[2,3], quat.w, quat.x, quat.y, quat.z])



def PoseError(x,xd):
    """
    Determine the pose error of the end effector.

    Input:
    x -- Actual position of the end effector, in the format [x y z ew ex ey ez]
    xd -- Desire position of the end effector, in the format [x y z ew ex ey ez]
    Output:
    err_pose -- Error position of the end effector, in the format [x y z ew ex ey ez]
    """
    pos_err = x[0:3]-xd[0:3]
    qact = Quaternion(x[3:7])
    qdes = Quaternion(xd[3:7])
    qdif =  qdes*qact.inverse
    qua_err = np.array([qdif.w,qdif.x,qdif.y,qdif.z])
    err_pose = np.hstack((pos_err,qua_err))
    return err_pose