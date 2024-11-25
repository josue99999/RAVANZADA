import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi


    
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


def fkine_ur5(q):
    """
    Calcular la cinematica directa del brazo robotico dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, ..., qn]
    """
    L1 = 0.08
    L2 = 0.18
    L3 = 0.2
    L = 0.350
    W = 0.110
    # Matrices DH (completar)
    T01 = np.dot(Ttransl([-L / 2, -W / 2, 0.0]), Trotx(-q[0]))
    T12 = np.dot(Ttransl([0.0, -L1, 0.0]), Troty(-q[1]))
    T23 = np.dot(Ttransl([0.0, 0.0, -L2]), Troty(-q[2]))
    T34 = Ttransl([0.0, 0.0, -L3])

    T = T01 @ T12 @ T23 @ T34
    return T




def jacobian_ur5(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
    """
    # Crear una matriz 3x8
    J = np.zeros((3,3))

    # Transformacion homogenea inicial (usando q)
    T = fkine_ur5(q)
    
    # Iteracion para la derivada de cada columna
    for i in range(3):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta

        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_ur5(dq)

        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0,i] = (dT[0,3] - T[0,3])/delta #derivadas de x
        J[1,i] = (dT[1,3] - T[1,3])/delta #derivadas de y
        J[2,i] = (dT[2,3] - T[2,3])/delta #derivadas de z
    return J


def ikine_ur5(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    Emplear el metodo de newton
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001
    

    q  = copy(q0)
    for i in range(max_iter):
        #main loop
        T = fkine_ur5(q)
        x = T[0:3,3]

        # error
        e = xdes - x
#        E[i,:] = e

        #Calculo del nuevo q:
        J = jacobian_ur5(q, delta)
        Jinv = np.linalg.pinv(J)
        
        q = q + Jinv@e
        
        #Condicion de cierre
        if (np.linalg.norm(e) < epsilon):
            print("\nValores articulares obtenidos: ", np.round(q,4))
            print("\nNumero de iteraciones: ", i)
            break

        if (i == max_iter-1):
            print("\nNo se encontro solucion en ", i, "iteraciones.")
            q = q-q
    return q
