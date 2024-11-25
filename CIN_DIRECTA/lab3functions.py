#!/usr/bin/env python3

import numpy as np
from utils import *



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




class Cuadrupedo1(object):
    def __init__(self):
        self.L1 = 0.08
        self.L2 = 0.18
        self.L3 = 0.2
        self.L = 0.350
        self.W = 0.110


    def update_config(self, q):
        self.q = q

    def cin_Pata_base(self, idx):

        if idx < 0 or idx > 3:
            raise ValueError("El índice de la pata debe estar entre 0 y 3.")

        # Para obtener los ángulos correspondientes de q
        base_idx = (idx * 3) + 7
        q1, q2, q3 = self.q[base_idx:base_idx + 3]

        if idx == 0:
            T01 = np.dot(Ttransl([-self.L / 2, -self.W / 2, 0.0]), Trotx(-q1))
            T12 = np.dot(Ttransl([0.0, -self.L1, 0.0]), Troty(-q2))
        elif idx == 1:
            T01 = np.dot(Ttransl([self.L / 2, -self.W / 2, 0.0]), Trotx(-q1))
            T12 = np.dot(Ttransl([0.0, -self.L1, 0.0]), Troty(-q2))
        elif idx == 2:
            T01 = np.dot(Ttransl([self.L /2, self.W /2, 0.0]), Trotx(q1))
            T12 = np.dot(Ttransl([0.0, self.L1, 0.0]), Troty(-q2))
        elif idx == 3:
            T01 = np.dot(Ttransl([-self.L / 2, self.W/2 , 0.0]), Trotx(q1))
            T12 = np.dot(Ttransl([0.0, self.L1, 0.0]), Troty(-q2))

        
        T23 = np.dot(Ttransl([0.0, 0.0, -self.L2]), Troty(-q3))
        T34 = Ttransl([0.0, 0.0, -self.L3])

        # Matriz de transformación total
        #T = np.dot(np.dot(np.dot(T01, T12), T23), T34)
        T = T01 @ T12 @ T23 @ T34
        return T
    
    def jacobian_pata_base(self, idx):

        if idx < 0 or idx > 3:
            raise ValueError("El índice de la pata debe estar entre 0 y 3.")

        # Para obtener los ángulos correspondientes de q
        base_idx = (idx * 3) + 7
        q1, q2, q3 = self.q[base_idx:base_idx + 3]

        if idx == 0:
            T01 = np.dot(Ttransl([-self.L / 2, -self.W / 2, 0.0]), Trotx(-q1))
            T12 = np.dot(Ttransl([0.0, -self.L1, 0.0]), Troty(-q2))
        elif idx == 1:
            T01 = np.dot(Ttransl([self.L / 2, -self.W / 2, 0.0]), Trotx(-q1))
            T12 = np.dot(Ttransl([0.0, -self.L1, 0.0]), Troty(-q2))
        elif idx == 2:
            T01 = np.dot(Ttransl([self.L /2, self.W /2, 0.0]), Trotx(q1))
            T12 = np.dot(Ttransl([0.0, self.L1, 0.0]), Troty(-q2))
        elif idx == 3:
            T01 = np.dot(Ttransl([-self.L / 2, self.W/2 , 0.0]), Trotx(q1))
            T12 = np.dot(Ttransl([0.0, self.L1, 0.0]), Troty(-q2))

        
        T23 = np.dot(Ttransl([0.0, 0.0, -self.L2]), Troty(-q3))
        T34 = Ttransl([0.0, 0.0, -self.L3])

        
        T02 = np.dot(T01, T12)
        T03 = np.dot(T02, T23)
        T04 = np.dot(T03, T34)

        w1 = T01[0:3,0]
        w2 = T02[0:3,1]
        w3 = T03[0:3,1]

        p14 = T04[0:3,3]-T01[0:3,3]
        p24 = T04[0:3,3]-T02[0:3,3]
        p34 = T04[0:3,3]-T03[0:3,3]

        v1 = np.cross(w1, p14)
        v2 = np.cross(w2, p24)
        v3 = np.cross(w3, p34)

        # Construcción del Jacobiano
        J = np.zeros((6, 3))

        J[0:3, 0] = v1
        J[0:3, 1] = v2
        J[0:3, 2] = v3

        J[3:6, 0] = w1
        J[3:6, 1] = w2
        J[3:6, 2] = w3

        return J

    def cin_Pata_inercia(self, idx):
        
        Pata_T = self.cin_Pata_base(idx)
        Base_R = rotationFromQuat(self.q[3:7])

        Pata_T[0:3,3] = np.dot(Base_R, Pata_T[0:3,3]) + self.q[0:3]
        Pata_T[0:3,0:3] = np.dot(Base_R, Pata_T[0:3,0:3])
        
        return Pata_T
    
    def jacobian_Pata_inercia(self, idx):

        JP = self.jacobian_pata_base(idx)
        TP = self.cin_Pata_inercia(idx)

        J = np.zeros((6,19))

        R = rotationFromQuat(self.q[3:7])
        Tq = Tmat(self.q)

        base_idx = (idx * 3) + 7

        J[0:3,base_idx:base_idx + 3] = np.dot(R, JP[0:3,:])
        J[3:6,base_idx:base_idx + 3] = np.dot(R, JP[3:6,:])

        J[0:3,0:3] = np.eye(3)
        J[0:3,3:7] = np.dot(skew(self.q[0:3]-TP[0:3,3]), Tq)
        J[3:6,3:7] = Tq

        return J
    
    def error_Pata_inercia(self, idx,pdes):
        error = pdes -self.cin_Pata_inercia(idx)[0:3,3]
        return error 
    


"""

def fkine(q):
     
    #Calcular la cinematica directa del brazo robotico dados sus valores articulares. 
    #q es un vector numpy de la forma [q1, q2, q3, ..., qn]
    
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



def jacobian(q, delta=0.0001):
    
    
    #Jacobiano analitico para la posicion de un brazo robotico de n grados de libertad. 
    #Retorna una matriz de 3xn y toma como entrada el vector de configuracion articular 
    #q=[q1, q2, q3, ..., qn]
    
    # Crear una matriz 3xn
    n = q.size
    J = np.zeros((3,n))
    # Calcular la transformacion homogenea inicial (usando q)
    T = fkine(q)
    # Iteracion para la derivada de cada articulacion (columna)
    for i in range(n):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Calcular nuevamenta la transformacion homogenea e
        # Incrementar la articulacion i-esima usando un delta
        dq[i] += delta
        # Transformacion homogenea luego del incremento (q+delta)
        T_inc = fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        J[0:3,i]=(T_inc[0:3,3]-T[0:3,3])/delta
    return J

def jacobian_pose(q, delta=0.0001):
    
    
    #Jacobiano analitico para la posicion y orientacion (usando un
    #cuaternion). Retorna una matriz de 7xn y toma como entrada el vector de
    #configuracion articular q=[q1, q2, q3, ..., qn]
    
    n = q.size
    J = np.zeros((7,n))
    # Implementar este Jacobiano aqui
    
        
    return J

def TF2xyzquat(T):
    
    #Convert a homogeneous transformation matrix into the a vector containing the
    #pose of the robot.

    #Input:
    #T -- A homogeneous transformation
    #Output:
    #X -- A pose vector in the format [x y z ew ex ey ez], donde la first part
    #        is Cartesian coordinates and the last part is a quaternion
    
    quat = Quaternion(matrix=T[0:3,0:3])
    return np.array([T[0,3], T[1,3], T[2,3], quat.w, quat.x, quat.y, quat.z])

def PoseError(x,xd):
    
    #Determine the pose error of the end effector.

    #Input:
    #x -- Actual position of the end effector, in the format [x y z ew ex ey ez]
    #xd -- Desire position of the end effector, in the format [x y z ew ex ey ez]
    #Output:
    #err_pose -- Error position of the end effector, in the format [x y z ew ex ey ez]
    
    pos_err = x[0:3]-xd[0:3]
    qact = Quaternion(x[3:7])
    qdes = Quaternion(xd[3:7])
    qdif =  qdes*qact.inverse
    qua_err = np.array([qdif.w,qdif.x,qdif.y,qdif.z])
    err_pose = np.hstack((pos_err,qua_err))
    return err_pose

"""