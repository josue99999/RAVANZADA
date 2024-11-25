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
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5, q6]
    """
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


def jacobian_pose(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion y orientacion (usando un
    cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
    configuracion articular q=[q1, q2, q3, q4, q5, q6]

    """
    J = np.zeros((7,8))
    # Implementar este Jacobiano aqui
    # Transformacion homogenea inicial (usando q)
    T = fkine_ur5(q)
   
    
    # Iteracion para la derivada de cada columna
    for i in range(8):
        
        #Obtener la matriz de rotacion
        R=T[0:2,0:2]
        #Obtener el cuaternion equivalente
        Q=rot2quat(R)
        
        
        # Copiar la configuracion articular inicial
        dq = copy(q)
        
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        

        # Transformacion homogenea luego del incremento (q+delta)
        dT = fkine_ur5(dq)
        dR = dT[0:2,0:2]
        dQ = rot2quat(dR)
        

        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0,i] = (dT[0,3] - T[0,3])/delta #derivadas de x
        J[1,i] = (dT[1,3] - T[1,3])/delta #derivadas de y
        J[2,i] = (dT[2,3] - T[2,3])/delta #derivadas de z
        
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[3,i] = (dQ[0,0] - Q[0,0])/delta #derivadas de w
        J[4,i] = (dQ[0,1] - Q[0,1])/delta #derivadas de ex
        J[5,i] = (dQ[0,2] - Q[0,2])/delta #derivadas de ey
        J[6,i] = (dQ[0,3] - Q[0,3])/delta #derivadas de ez
        
    
    return J



def rot2quat(R):
    """
    Convertir una matriz de rotacion en un cuaternion

    Entrada:
      R -- Matriz de rotacion
    Salida:
      Q -- Cuaternion [ew, ex, ey, ez]

    """
    dEpsilon = 1e-6
    quat = 4*[0.,]

    quat[0] = 0.5*np.sqrt(R[0,0]+R[1,1]+R[2,2]+1.0)
    if ( np.fabs(R[0,0]-R[1,1]-R[2,2]+1.0) < dEpsilon ):
        quat[1] = 0.0
    else:
        quat[1] = 0.5*np.sign(R[2,1]-R[1,2])*np.sqrt(R[0,0]-R[1,1]-R[2,2]+1.0)
    if ( np.fabs(R[1,1]-R[2,2]-R[0,0]+1.0) < dEpsilon ):
        quat[2] = 0.0
    else:
        quat[2] = 0.5*np.sign(R[0,2]-R[2,0])*np.sqrt(R[1,1]-R[2,2]-R[0,0]+1.0)
    if ( np.fabs(R[2,2]-R[0,0]-R[1,1]+1.0) < dEpsilon ):
        quat[3] = 0.0
    else:
        quat[3] = 0.5*np.sign(R[1,0]-R[0,1])*np.sqrt(R[2,2]-R[0,0]-R[1,1]+1.0)

    return np.array(quat)


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
    quat = rot2quat(T[0:3,0:3])
    res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
    return np.array(res)


def skew(w):
    R = np.zeros([3,3])
    R[0,1] = -w[2]; R[0,2] = w[1]
    R[1,0] = w[2];  R[1,2] = -w[0]
    R[2,0] = -w[1]; R[2,1] = w[0]
    return R
