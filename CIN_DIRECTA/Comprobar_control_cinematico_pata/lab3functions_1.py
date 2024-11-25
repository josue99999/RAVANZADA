#!/usr/bin/env python3

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

def fkine(q):
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
    T01 = np.dot(Ttransl([-L / 2, W / 2, 0.0]), Trotx(q[0]))
    T12 = np.dot(Ttransl([0.0, L1, 0.0]), Troty(-q[1]))
    T23 = np.dot(Ttransl([0.0, 0.0, -L2]), Troty(-q[2]))
    T34 = Ttransl([0.0, 0.0, -L3])

    T = T01 @ T12 @ T23 @ T34
    return T

