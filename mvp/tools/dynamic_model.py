import math
import numpy as np


class KinematicBicycleModel:

    def __init__(self, wheelbase: float=2.5, dt: float=0.1, c_r: float=0.0, c_a: float=0.0, velocity_bound=[0, 50], throttle_bound=[-30, 30], steering_angle_bound=[-2.0, 2.0]):
        """
        2D Kinematic Bicycle Model
        At initialisation
        :param wheelbase:           (float) vehicle's wheelbase [m]
        :param dt:                  (float) discrete time period [s]
        :param c_r:                 (float) vehicle's coefficient of resistance 
        :param c_a:                 (float) vehicle's aerodynamic coefficient
    
        At every time step  
        :param x:                   (float) vehicle's x-coordinate [m]
        :param y:                   (float) vehicle's y-coordinate [m]
        :param yaw:                 (float) vehicle's heading [rad]
        :param velocity:            (float) vehicle's velocity in the x-axis [m/s]
        :param throttle:            (float) vehicle's accleration [m/s^2]
        :param delta:               (float) vehicle's steering angle [rad]
    
        :return new_x:              (float) vehicle's x-coordinate [m]
        :return new_y:              (float) vehicle's y-coordinate [m]
        :return new_yaw:            (float) vehicle's heading [rad]
        :return new_velocity:       (float) vehicle's velocity in the x-axis [m/s]
        :return steering_angle:     (float) vehicle's steering angle [rad]
        :return angular_velocity:   (float) vehicle's angular velocity [rad/s]
        """

        self.dt = dt
        self.wheelbase = wheelbase
        self.velocity_bound = velocity_bound
        self.throttle_bound = throttle_bound
        self.steering_angle_bound = steering_angle_bound
        self.c_r = c_r
        self.c_a = c_a

    def step(self, x: float, y: float, yaw: float, velocity: float, throttle: float, steering_angle: float):
        # Limit steering angle to physical vehicle limits
        fixed_steering_angle = self.steering_angle_bound[0] if steering_angle < self.steering_angle_bound[0] else self.steering_angle_bound[1] if steering_angle > self.steering_angle_bound[1] else steering_angle
        fixed_throttle = self.throttle_bound[0] if throttle < self.throttle_bound[0] else self.throttle_bound[1] if throttle > self.throttle_bound[1] else throttle

        # Compute the local velocity in the x-axis
        friction     = velocity * (self.c_r + self.c_a*velocity)
        new_velocity = np.clip(velocity + self.dt*(fixed_throttle - friction), self.velocity_bound[0], self.velocity_bound[1])

        # Compute the angular velocity
        angular_velocity = velocity * math.tan(fixed_steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x   = x + velocity*math.cos(yaw)*self.dt
        new_y   = y + velocity*math.sin(yaw)*self.dt
        normalise_angle = lambda angle: math.atan2(math.sin(angle), math.cos(angle))
        new_yaw = normalise_angle(yaw + angular_velocity * self.dt)

        return new_x, new_y, new_yaw, new_velocity, fixed_throttle, fixed_steering_angle, angular_velocity

    def step_torch(self, x, y, yaw, velocity, throttle, steering_angle):
        import torch
        # Limit steering angle to physical vehicle limits
        fixed_steering_angle = torch.clip(steering_angle, min=self.steering_angle_bound[0], max=self.steering_angle_bound[1])
        fixed_throttle = torch.clip(throttle, min=self.throttle_bound[0], max=self.throttle_bound[1])

        # Compute the local velocity in the x-axis
        friction     = velocity * (self.c_r + self.c_a*velocity)
        new_velocity = torch.clip(velocity + self.dt*(fixed_throttle - friction), min=self.velocity_bound[0], max=self.velocity_bound[1])

        # Compute the angular velocity
        angular_velocity = velocity * torch.tan(fixed_steering_angle) / self.wheelbase

        # Compute the final state using the discrete time model
        new_x   = x + velocity*torch.cos(yaw)*self.dt
        new_y   = y + velocity*torch.sin(yaw)*self.dt
        normalise_angle = lambda angle: torch.atan2(torch.sin(angle), torch.cos(angle))
        new_yaw = normalise_angle(yaw + angular_velocity * self.dt)
        
        return new_x, new_y, new_yaw, new_velocity, fixed_throttle, fixed_steering_angle, angular_velocity

    def fit(self, bboxes):
        length = len(bboxes)
        angular_velocity = (bboxes[1:, 6] - bboxes[:-1, 6]) / self.dt
        velocity = np.zeros(length - 1)
        for i in range(length - 1):
            if np.absolute(bboxes[i + 1, 0] - bboxes[i, 0]) > np.absolute(bboxes[i + 1, 1] - bboxes[i, 1]):
                velocity[i] = ((bboxes[i + 1, 0] - bboxes[i, 0]) / np.cos(bboxes[i, 6]) / self.dt)
            else:
                velocity[i] = ((bboxes[i + 1, 1] - bboxes[i, 1]) / np.sin(bboxes[i, 6]) / self.dt)
        steering_angle = np.arctan(angular_velocity * self.wheelbase / velocity)
        for i in range(len(steering_angle)):
            if np.isnan(steering_angle[i]):
                if i == 0:
                    steering_angle[i] = 0
                else:
                    steering_angle[i] = steering_angle[i - 1]
        throttle = (velocity[1:] - velocity[:-1]) / self.dt + velocity[:-1] * (self.c_r + self.c_a * velocity[:-1])

        return np.concatenate((velocity, np.ones(1) * velocity[-1]), axis=None), np.concatenate((throttle, np.ones(2) * throttle[-1]), axis=None), np.concatenate((steering_angle, np.ones(1) * steering_angle[-1]), axis=None)
    
    def check(self, bboxes):
        velocity, throttle, steering_angle = self.fit(bboxes)
        return np.sum(velocity < self.velocity_bound[0]) == 0 and np.sum(velocity > self.velocity_bound[1]) == 0 and \
               np.sum(throttle < self.throttle_bound[0]) == 0 and np.sum(throttle > self.throttle_bound[1]) == 0 and \
               np.sum(steering_angle < self.steering_angle_bound[0]) == 0 and np.sum(steering_angle > self.steering_angle_bound[1]) == 0

    def check_score(self, bboxes):
        velocity, throttle, steering_angle = self.fit(bboxes)
        return (
            (np.sum(np.clip(self.velocity_bound[0] - velocity, 0, 100)) + np.sum(np.clip(velocity - self.velocity_bound[1], 0, 100))) / (self.velocity_bound[1] - self.velocity_bound[0]) + \
            (np.sum(np.clip(self.throttle_bound[0] - throttle, 0, 100)) + np.sum(np.clip(throttle - self.throttle_bound[1], 0, 100))) / (self.throttle_bound[1] - self.throttle_bound[0]) + \
            (np.sum(np.clip(self.steering_angle_bound[0] - steering_angle, 0, 100)) + np.sum(np.clip(steering_angle - self.steering_angle_bound[1], 0, 100))) / (self.steering_angle_bound[1] - self.steering_angle_bound[0])
        ) / len(bboxes)