#!/usr/bin/env python3
import numpy as np
import osqp
from scipy import sparse

class OSQPController:
   def __init__(self, weights, lambdas, dt):
        self.w1 = weights[0]
        self.w2 = weights[1]
        self.w3 = weights[2]
        self.w4 = weights[3]
        self.lambda1 = lambdas[0]
        self.lambda2 = lambdas[1]
        self.lambda3 = lambdas[2]
        self.lambda4 = lambdas[3]
        self.dt = dt
        self.n = 19
        # Joint limits
        self.qmin = np.array([-1.4, -1.4, -2.5, -1.4, -1.4, -2.5, -1.4, -1.4,
                              -2.5, -1.4, -1.4, -2.5])
        self.qmax = np.array([1.4, 1.4, 2.5, 1.4, 1.4, 2.5, 1.4, 1.4, 2.5, 1.4,
                              1.4, 2.5])
        self.dqmax = 10.0*np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0])
        self.dqmin = -self.dqmax
        # Bounds for the floating base
        low = -1e6
        high = 1e6
        self.lfb = np.array([low, low, low, low, low, low, low])
        self.ufb = np.array([high, high, high, high, high, high, high])
        self.A = sparse.eye(self.n, format='csc')
        
def get_dq(self, q, e1, J1, e2, J2, e3, J3, e4, J4):
        de1 = self.lambda1 * e1
        de2 = self.lambda2 * e2
        de3 = self.lambda3 * e3
        de4 = self.lambda4 * e4

        W = self.w1 * np.dot(J1.T, J1) + self.w2 * np.dot(J2.T, J2) + \
            self.w3 * np.dot(J3.T, J3) + self.w4 * np.dot(J4.T, J4)
        p = -2 * (self.w1 * np.dot(J1.T, de1) + self.w2 * np.dot(J2.T, de2) + \
                   self.w3 * np.dot(J3.T, de3) + self.w4 * np.dot(J4.T, de4))
        
        W = sparse.csc_matrix(W)

        lower_limits = np.maximum((self.qmin - q[7:]) / self.dt, self.dqmin)
        upper_limits = np.minimum((self.qmax - q[7:]) / self.dt, self.dqmax)

        lower_limits = np.hstack((self.lfb, lower_limits))
        upper_limits = np.hstack((self.ufb, upper_limits))

        print("lower_limits: ",lower_limits,"Upperlimits",upper_limits)
        # Create OSQP solver
        solver = osqp.OSQP()
        solver.setup(P=W, q=p, A=self.A, l=lower_limits, u=upper_limits, verbose=False)

        # Solve the problem
        res = solver.solve()

        # Get the solution
        dq = res.x
        return dq
