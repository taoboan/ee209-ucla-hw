#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:56:02 2018

@author: andytao
"""

import numpy as np
from math import sin, cos
import sympy as sp
import matplotlib.pyplot as plt


class CAR:
    def __init__(self,length=750, width=500, wheel_r=20, wheel_d=85,dt=0.1, state0=None, rpm=130,
                 std_sensor_percent=0.03, std_IMU=np.radians(0.1), speed_error_percent=0.05):
        #Initialize environment and sensor standard error
        #The max motor speed is 130rpm at 6V
        #The standard deviation of range sensors is 3% of the max range
        #The standard deviation of IMU is 0.1◦
        self.l = length
        self.w = width
        self.r = wheel_r
        self.d = wheel_d
        self.dt = dt
        self.rpm = rpm
        self.max_motor_angularV = rpm * 2 * np.pi / 60
        self.std_sensor_percent = std_sensor_percent
        self.std_IMU = std_IMU
        self.std_speed = self.max_motor_angularV * speed_error_percent
        self.speed_error = np.random.normal(0,self.std_speed**2)

        if state0 is None:
            self.state = np.array([0, 0, 0, 0, 0])
        else:
            self.state = state0
        #Using jacobian method to get the expression of matrix equation in kalman filter
        x, y, theta, angularVl, angularVr, errorVl, errorVr, v,bias = sp.symbols('x, y, angularVl, theta, angularVr, errorVl, errorVr, v,bias')

        self.f = sp.Matrix([[x + (1 / 2 * (angularVl + errorVl+ angularVr + errorVr) * self.r) * sp.cos(theta) * self.dt],
                            [y + (1 / 2 * (angularVl + errorVl+ angularVr + errorVr) * self.r) * sp.sin(theta) * self.dt],
                            [theta + ((angularVl - angularVr) * self.r) / self.d * self.dt],
                            [v],
                            [bias]])

        self.sym_state = sp.Matrix([x, y, theta, v, bias])
        self.sym_process_noise = sp.Matrix([ errorVl, errorVr])
        self.F = self.f.jacobian(self.sym_state)
        self.W = self.f.jacobian(self.sym_process_noise)
        self.P = np.eye(5)
        self.Q = np.eye(2)*self.speed_error ** 2
        self.R = np.diag([(1200 * std_sensor_percent) ** 2, (1200 * std_sensor_percent) ** 2, std_IMU ** 2])


    #Get the state of car basing on velocity and time
    def move(self, state, motor_angularV, time):
        real_angular_velocity = motor_angularV + self.speed_error
        vl = real_angular_velocity [0] * self.r
        vr = real_angular_velocity [1] * self.r
        linear_velocity = (vl + vr) / 2
        angular_velocity = (vl - vr) / self.d
        trajactory_length = linear_velocity * time
        angle_change = angular_velocity * time

        current_angle = state[2]
        dx = trajactory_length * cos(current_angle + angle_change/2)
        dy = trajactory_length * sin(current_angle + angle_change/2)

        next_state = state + np.array((dx, dy, angle_change, 0, 0))
        next_state[3] = linear_velocity

        return next_state

    #Time update for ekf
    def time_update(self, motor_angularV, step):

            self.state = self.move(self.state, motor_angularV, self.dt * step)


            x, y, theta, angularVl, angularVr, errorVl, errorVr,v, bias = sp.symbols('x, y, angularVl, theta, angularVr, errorVl, errorVr,v, bias')

            subs_value = {x: self.state[0], y: self.state[1], theta: self.state[2], v:self.state[3],bias: self.state[4],
                          angularVl: motor_angularV[0], angularVr: motor_angularV[1],errorVl: self.speed_error, errorVr: self.speed_error}


            F_update = np.array(self.F.evalf(subs=subs_value))
            W_update = np.array(self.W.evalf(subs=subs_value))
            self.P = np.dot( np.dot(F_update, self.P), F_update.T ) + np.dot( np.dot(W_update, self.Q), W_update.T )

    #Change angle to the domain (-pi,pi)
    def angle_range_process(self, angle):

            Angle = angle % (2 * np.pi)
            if Angle >= np.pi:
                Angle -= 2 * np.pi
            return Angle
    #Calculate the distance basing on angle 
    def get_distance(self, x, y, angle, outline):

            angle = self.angle_range_process(angle)

            if 0 <= y + (outline[0] - x) * np.tan(angle) <= self.w:
                return (outline[0] - x) / np.cos(angle)
            else:
                return (outline[1] - y) / np.sin(angle)
    #Calculate the corresponding distance basing on sensor measurement       
    def range_distance_calcualte(self, state):

            angle = self.angle_range_process(state[2])
            x = state[0]
            y = state[1]

            if -np.pi <= angle < -np.pi / 2:
                if angle != -np.pi:
                    outline = [0, 0]
                    front_distance = self.get_distance(x, y, angle, outline)
                    right_distance = self.get_distance(x, y, angle - np.pi / 2, outline)
                else:
                    front_distance = x
                    right_distance = self.w - y

            elif 0 > angle >= -np.pi / 2:
                if angle != -np.pi / 2:
                    outline = [self.l, 0]
                    front_distance = self.get_distance(x, y, angle, outline)
                    right_distance = self.get_distance(x, y, angle - np.pi / 2, outline)
                else:
                    front_distance = y
                    right_distance = x
            elif 0 <= angle < np.pi / 2:
                if angle != 0:
                    outline = [self.l, self.w]
                    front_distance = self.get_distance(x, y, angle, outline)
                    right_distance = self.get_distance(x, y, angle - np.pi / 2, outline)
                else:
                    front_distance = self.l - x
                    right_distance = y
            else:
                if angle != np.pi / 2:
                    outline = [0, self.w]
                    front_distance = self.get_distance(x, y, angle, outline)
                    right_distance = self.get_distance(x, y, angle - np.pi / 2, outline)
                else:
                    front_distance = self.w - y
                    right_distance = self.l - x
            return np.array([front_distance, right_distance, angle])



    # Jacobian calculation process
    def Jacobian_form(self, x, y, angle, outline):

            if 0 <= y + (outline[0] - x) * np.tan(angle) <= self.w:
                return np.array([-1 / np.cos(angle), 0, (outline[0] - x) * np.sin(angle) / np.cos(angle) ** 2])
            else:
                return np.array([0, -1 / np.sin(angle), -(outline[1] - y) * np.cos(angle) / np.sin(angle) ** 2])
    # Get the jacobian form of sensor measurement
    def get_Jacobian_form_sensor_measure(self, state, HEAD):

            if HEAD == 'f':
                angle = self.angle_range_process(state[2])
            else:
                angle = self.angle_range_process(state[2] - np.pi / 2)

            x = state[0]
            y = state[1]

            if -np.pi <= angle < -np.pi / 2:
                if angle != -np.pi:
                    outline = [0, 0]
                    return self.Jacobian_form(x, y, angle, outline)
                else:
                    return np.array([1, 0, 0])
            elif 0 > angle >= -np.pi / 2:
                if angle != -np.pi / 2:
                    outline = [self.l, 0]
                    return self.Jacobian_form(x, y, angle, outline)
                else:
                    return np.array([0, 1, 0])
            elif 0 <= angle < np.pi / 2:
                if angle != 0:
                    outline = [self.l, self.w]
                    return self.Jacobian_form(x, y, angle, outline)
                else:
                    return np.array([-1, 0, 0])
            else:
                if angle != np.pi / 2:
                    outline = [0, self.w]
                    return self.Jacobian_form(x, y, angle, outline)
                else:
                    return np.array([0, -1, 0])
    # H matrix in pbservation update
    def H_matrix(self, state):
           
            H_front_sensor = self.get_Jacobian_form_sensor_measure(state, 'f')
            H_right_sensor = self.get_Jacobian_form_sensor_measure(state, 'r')
            angle = np.array([0, 0, 1])

            return np.array([H_front_sensor, H_right_sensor, angle])
   # Do the observation process
    def observation_update(self, measurement):

            H_array = np.array(self.H_matrix(self.state[:3]))
            delta_r = measurement - self.range_distance_calcualte(self.state)
            X1 = np.dot(self.P[:3, :3], H_array.T)
            X2 = np.dot( np.dot(H_array, self.P[:3, :3]),H_array.T) + self.R
            X3 = np.linalg.inv(np.matrix( X2, dtype=float))
            K = np.dot(X1, X3)
            self.state[:3] = self.state[:3] + K.dot(delta_r)
            self.P[:3, :3] = self.P[:3, :3] - np.dot(K, H_array).dot(self.P[:3, :3])

# Simulation process
def simulation(time, state00, input, steps, initial_know):

    if initial_know is True:
        car = CAR(state0 = state00)
    else:
        car = CAR()
    state = state00
    dt = car.dt
    x_traj = []
    y_traj = []

    x_ekf = []
    y_ekf = []

    for i in range(int(time / dt)):
        velocity = input[i]

        state = car.move(state, velocity, dt)
        x_traj.append(state[0])
        y_traj.append(state[1])


        if  (state[0] < 0 or state[0] > 750 or state[1] < 0 or  state[1] > 500):
            print("Out of bound!")
            break

        if i % steps == 0:
           
            car.time_update(velocity, steps)
            observe_cal = car.range_distance_calcualte(state)
            car.observation_update(observe_cal)
            x_ekf.append(car.state[0])
            y_ekf.append(car.state[1])

    return x_traj, y_traj, x_ekf, y_ekf

# plot trajactory
def plot_trajectory(x_path, y_path, x_estimator, y_estimator):

        plt.figure()
        plt.grid(True)
        plt.xlim((0, 750))
        plt.ylim((0, 500))

        plt.plot(x_path, y_path)
        plt.plot(x_estimator, y_estimator, marker='d', markersize=2)
        plt.legend(['True', 'Estimated'])
        plt.title('Extended Kalman Filter')
        plt.show()

if __name__ == '__main__':

        time = 80
        state000 = np.array([50, 50, np.pi/4, 0, 0])

        input = np.zeros((800, 2))
        for i in range(800):
            if i <= 100:
                input[i] = np.array([1, 1.2])
            elif 100<i<=200:
                input[i] = np.array([1.2, 1])
            elif i>200: 
                input[i] = np.array([1, 1.3])



        x1, y1, x2, y2 = simulation(time, state000, input, steps=4, initial_know=False)
        plot_trajectory(x1, y1, x2, y2)
