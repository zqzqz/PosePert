import numpy as np
from mvp.tools.dynamic_model import KinematicBicycleModel
from mvp.tools.iou import iou3d


class UncertaintyModel:
    def __init__(self, params):
        self.params = params

    def sample(self, x, k=1):
        raise NotImplementedError()


class TrajectoryUncertaintyModel(UncertaintyModel):
    def __init__(self, params):
        default_params = {
            "type": "ade",
            "bound": 0.8,
            "throttle_perturb": 5,
            "steering_angle_perturb": 0.1,
        }
        default_params.update(params)
        super().__init__(default_params)
        self.dynamic_model = KinematicBicycleModel()

    def score(self, x1, x2):
        if self.params["type"] == "ade":
            return np.sqrt(np.mean(np.sum((x1[:, :2] - x2[:, :2]) ** 2, axis=1)))
        else:
            raise NotImplementedError()

    def bound(self):
        return self.params["bound"]

    def random_perturb(self, x, k=1, start_frame_id=0):
        traj_len, traj_dim = x.shape[0], x.shape[1]

        velocity, throttle, steering_angle = self.dynamic_model.fit(x)
        throttle_perturb = (np.random.random((k, *(throttle.shape))) * 2 - 1) * self.params["throttle_perturb"]
        steering_angle_perturb = (np.random.random((k, *(steering_angle.shape))) * 2 - 1) * self.params["steering_angle_perturb"]

        result = np.tile(x, (k, 1, 1))
        for i in range(k):
            velocity_tmp = velocity[0]
            for j in range(x.shape[0] - 1):
                new_x, new_y, new_yaw, velocity_tmp, _, _, _ = self.dynamic_model.step(
                    result[i, j, 0], result[i, j, 1], result[i, j, 6], velocity_tmp,
                    throttle[j] + throttle_perturb[i, j], steering_angle[j] + steering_angle_perturb[i, j],
                )
                if j + 1 >= start_frame_id:
                    result[i, j + 1, 0], result[i, j + 1, 1], result[i, j + 1, 6] = new_x, new_y, new_yaw

        return result

    def sample(self, x, k=1, start_frame_id=0):
        # Trajectory N * 7 as input
        traj_len, traj_dim = x.shape[0], x.shape[1]
        assert(traj_len > 3)
        if traj_dim != 7:
            new_x = np.zeros((traj_len, 7))
            new_x[:, :2] = x[:, :2]
            new_x[:, 3:6] = np.array([4, 1.7, 1.6])
            new_x[:traj_len - 1, 6] = np.arctan2(x[1:, 1] - x[:-1, 1], x[1:, 0] - x[:-1, 0])
            new_x[traj_len - 1, 6] = new_x[traj_len - 2, 6]
            x = new_x
        
        result = [x]
        while len(result) < k:
            perturbed_x = self.random_perturb(x, k=1, start_frame_id=start_frame_id)[0]
            score = self.score(x, perturbed_x)
            if score < self.bound():
                result.append(perturbed_x)
        
        return np.asarray(result)


class ShiftAttackUncertaintyModel(UncertaintyModel):
    def __init__(self, params):
        default_params = {
            "type": "iou",
            "bound": 0.3,
            "xy_perturb": 0.4,
            "yaw_perturb": 0.2,
        }
        default_params.update(params)
        super().__init__(default_params)

    def score(self, x1, x2):
        if self.params["type"] == "iou":
            return iou3d(x1, x2)
        else:
            raise NotImplementedError()

    def bound(self):
        return self.params["bound"]

    def random_perturb(self, x, k=1):
        xy_perturb = (np.random.random((k, 2)) * 2 - 1) * self.params["xy_perturb"]
        yaw_perturb = (np.random.random(k) * 2 - 1) * self.params["yaw_perturb"]

        result = np.tile(x, (k, 1))
        result[:, :2] += xy_perturb
        result[:, 6] += yaw_perturb

        return result

    def sample(self, x, k=1):
        # Bounding box (7,) as input
        assert(len(x) == 7)
        
        result = [x]
        while len(result) < k:
            perturbed_x = self.random_perturb(x, k=1)[0]
            score = self.score(x, perturbed_x)
            print(score)
            if score < self.bound():
                result.append(perturbed_x)
        
        return np.asarray(result)
