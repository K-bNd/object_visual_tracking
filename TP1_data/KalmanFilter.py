import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.1, u_x=1, u_y=1, std_acc=1, x_dt_meas=0.1, y_dt_meas=0.1) -> None:
        self.dt = dt
        self.u = np.matrix([[u_x], [u_y]]) 
        self.std_acc = std_acc
        self.x_dt_meas = x_dt_meas
        self.y_dt_meas = y_dt_meas
        self.state_matrix = np.zeros((4, 1))
        self.A = np.matrix([[1, 0, dt, 0],
                            [0, 1, 0, dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.matrix([[(dt**2)/2, 0],
                            [0, (dt**2)/2],
                            [dt, 0],
                            [0, dt]])
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]]) * self.std_acc**2
        self.R = np.matrix([[x_dt_meas**2, 0],
                            [0, y_dt_meas**2]])
        
        self.P = np.eye(self.A.shape[1])

    def predict(self, current_x, current_y):
        """
        current_x: current x position
        current_y: current y position
        """
        self.state_matrix = np.matrix([[current_x], [current_y], [0], [0]])
        self.state_matrix = np.dot(self.A, self.state_matrix) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.state_matrix
    
    def update(self, Z):
        """
        Z: measurement vector (x, y)
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        Z = np.matrix([[self.x_dt_meas], [self.y_dt_meas]])
        y = Z - np.dot(self.H, self.state_matrix)
        self.state_matrix = self.state_matrix + np.dot(K, y)
        self.P = (np.eye(self.H.shape[1]) - (K * self.H)) * self.P
        return self.state_matrix