import copy
import numpy as np
import torch
import logging

import os, sys
root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../third_party/AdvTrajectoryPrediction")
sys.path.append(root)

from .scenario_attacker import ScenarioAttacker
from .scenario_attacker_util import *
from .uncertainty_model import TrajectoryUncertaintyModel, ShiftAttackUncertaintyModel
from .differentiable_kalman import DifferentiableKalmanSurrogate
from mvp.tools.dynamic_model import KinematicBicycleModel


class ScenarioShiftMoveinAttacker(ScenarioAttacker):
    def __init__(self, use_uncertainty=True, learned_uncertainty_model=None, no_online_update=False, use_tracking_surrogate=False, observation_noise_std=0.0, observation_noise_k=5, **kwargs):
        super().__init__(**kwargs)
        self.name = "scenario_perturb_movein"
        self.dynamic_model = KinematicBicycleModel()
        self.use_uncertainty = use_uncertainty
        self.learned_uncertainty_model = learned_uncertainty_model
        self.no_online_update = no_online_update
        self.use_tracking_surrogate = use_tracking_surrogate
        self.observation_noise_std = observation_noise_std
        self.observation_noise_k = observation_noise_k
        if self.use_tracking_surrogate:
            self.kalman_surrogate = DifferentiableKalmanSurrogate()
        if self.use_uncertainty:
            self.trajectory_uncertainty_model = TrajectoryUncertaintyModel(params={})
            self.shift_attack_uncertainty_model = ShiftAttackUncertaintyModel(params={})

    def attack_fitness(self, case, attack_opts, start_frame_id=None, end_frame_id=None, requires_grad=False):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["attacker_victim_track_id"]
        if start_frame_id is None:
            start_frame_id = self.attack_start_frame_id
        if end_frame_id is None:
            end_frame_id = self.attack_end_frame_id
        attack_num_frames = end_frame_id - start_frame_id + 1

        if "target_trajectory" not in attack_opts:
            self.apply_perturbation(case, attack_opts, requires_grad=requires_grad)
        target_id, target_trajectory = attack_opts["attacker_target_track_id"], attack_opts["target_trajectory"]
        decay = 0.5

        if requires_grad:
            coeficient = torch.logspace(0, attack_num_frames - 1, attack_num_frames, base=decay)
        else:
            coeficient = np.logspace(0, attack_num_frames - 1, base=decay, num=attack_num_frames)

        if requires_grad:
            fitness_list = torch.zeros(attack_num_frames)
        else:
            fitness_list = np.zeros(attack_num_frames)

        for frame_index, frame_id in enumerate(range(start_frame_id, end_frame_id + 1)):
            observed_trajectories = copy.deepcopy(case[frame_id][attacker_vehicle_id]["observed_trajectories"])
            if requires_grad:
                for vid in observed_trajectories:
                    observed_trajectories[vid] = torch.from_numpy(observed_trajectories[vid]).to(self.device)
            
            copy_length = min(self.history_num_frames, frame_id - self.attack_start_frame_id + 1)

            if copy_length > 0 and target_trajectory is not None:
                if requires_grad and self.use_tracking_surrogate:
                    filtered = self._apply_tracking_surrogate(
                        case, attack_opts, target_trajectory[:copy_length])
                    tracked = target_trajectory[:copy_length].clone()
                    tracked[:, :2] = filtered
                    observed_trajectories[target_id][-copy_length:] = tracked
                else:
                    observed_trajectories[target_id][-copy_length:] = target_trajectory[:copy_length]

            model_args = {"model_api": self.prediction_model_api,
                         "obs_length": self.history_num_frames,
                         "pred_length": self.predict_num_frames}
            if self.prediction_func == prediction_grip and requires_grad:
                original_trajectory = torch.from_numpy(case[frame_id][attacker_vehicle_id]["observed_trajectories"][target_id]).to(self.device)[-self.history_num_frames:, :2]
                perturbed_trajectory = observed_trajectories[target_id][-self.history_num_frames:, :2]
                perturbation = perturbed_trajectory - original_trajectory
                model_args.update({"target_id": target_id, "perturbation": perturbation})

            predicted_trajectories = self.prediction_func(observed_trajectories, model_args=model_args, num_frames=self.predict_num_frames, object_ids=[target_id], requires_grad=requires_grad)
            key_trajectory = predicted_trajectories[target_id][:self.predict_num_frames]

            use_obs_noise = (not requires_grad and self.observation_noise_std > 0
                             and copy_length > 0 and target_trajectory is not None)
            if use_obs_noise:
                obs_list = []
                for _k in range(self.observation_noise_k):
                    obs_noisy = copy.deepcopy(observed_trajectories)
                    noise = np.random.normal(0, self.observation_noise_std, (copy_length, 2))
                    obs_noisy[target_id][-copy_length:, :2] += noise
                    obs_list.append(obs_noisy)
                pred_list = self.prediction_func(obs_list, model_args=model_args,
                                                  num_frames=self.predict_num_frames,
                                                  object_ids=[target_id], batch=True, requires_grad=False)
                obs_noise_key_samples = [p[target_id][:self.predict_num_frames] for p in pred_list]

            # Injects the uncertainty model of attacker's prediction on the target vehicle v.s. victim's prediction.
            if not requires_grad and self.use_uncertainty and not use_obs_noise:
                if self.learned_uncertainty_model is not None:
                    # Use learned model: sample error-perturbed trajectories
                    target_traj_for_sample = attack_opts.get("target_trajectory", None)
                    if target_traj_for_sample is not None:
                        key_trajectory_sampled, detect_prob = \
                            self.learned_uncertainty_model.sample_target_trajectories(
                                key_trajectory, attack_opts, case, frame_id, k=10)
                        # Convert to list of trajectories for compatibility
                        key_trajectory_sampled = [key_trajectory_sampled[i] for i in range(len(key_trajectory_sampled))]
                    else:
                        key_trajectory_sampled = self.trajectory_uncertainty_model.sample(key_trajectory, k=10)
                else:
                    key_trajectory_sampled = self.trajectory_uncertainty_model.sample(key_trajectory, k=10)

            victim_trajectory = case[frame_id][attacker_vehicle_id]["predicted_trajectories"][victim_vehicle_id][:self.predict_num_frames].copy()
            if requires_grad:
                victim_trajectory = torch.from_numpy(victim_trajectory).to(self.device)
            # Injects the uncertainty model of attacker's prediction on the victim vehicle v.s. ground truth.
            if not requires_grad and self.use_uncertainty and not use_obs_noise:
                victim_trajectory_sampled = self.trajectory_uncertainty_model.sample(victim_trajectory, k=10)

            if requires_grad:
                attack_opts["ideal_predicted_trajectories"][frame_id] = key_trajectory.cpu().detach().numpy()
            else:
                attack_opts["ideal_predicted_trajectories"][frame_id] = key_trajectory

            if requires_grad:
                fitness_list[frame_index] = -coeficient[frame_index] * torch.log(torch.clip(min_trajectory_distance(key_trajectory, victim_trajectory, requires_grad=requires_grad), 0.1, 100))
            elif use_obs_noise:
                fitness_list[frame_index] = -coeficient[frame_index] * np.mean([np.log(np.clip(min_trajectory_distance(_key, victim_trajectory, requires_grad=False), 0.1, 100)) for _key in obs_noise_key_samples])
            elif self.use_uncertainty:
                fitness_list[frame_index] = -coeficient[frame_index] * np.mean([np.log(np.clip(min_trajectory_distance(_key, _victim, requires_grad=requires_grad), 0.1, 100)) for _key in key_trajectory_sampled for _victim in victim_trajectory_sampled])
            else:
                fitness_list[frame_index] = -coeficient[frame_index] * np.log(np.clip(min_trajectory_distance(key_trajectory, victim_trajectory, requires_grad=requires_grad), 0.1, 100))
        
        fitness = fitness_list.sum() / coeficient.sum()
        return fitness
    
    def attack_fitness_blackbox_batch(self, case, attack_opts, perturbation_list, start_frame_id=None, end_frame_id=None):
        attacker_vehicle_id, victim_vehicle_id, target_id = attack_opts["attacker_vehicle_id"], attack_opts["attacker_victim_track_id"], attack_opts["attacker_target_track_id"]
        if start_frame_id is None:
            start_frame_id = self.attack_start_frame_id
        if end_frame_id is None:
            end_frame_id = self.attack_end_frame_id
        attack_num_frames = end_frame_id - start_frame_id + 1

        batch_size = len(perturbation_list)
        decay = 0.5
        coeficient = np.logspace(0, attack_num_frames - 1, base=decay, num=attack_num_frames)
        fitness_list = np.zeros((batch_size, attack_num_frames))

        target_trajectory_list = []
        for perturbation in perturbation_list:
            attack_opts_tmp = copy.deepcopy(attack_opts)
            attack_opts_tmp["perturbation"] = perturbation
            self.apply_perturbation(case, attack_opts_tmp, requires_grad=False)
            target_trajectory = attack_opts_tmp["target_trajectory"]

            tracks = copy.deepcopy(case[self.attack_start_frame_id - 1][attacker_vehicle_id]["tracks"])
            # tracks = {target_id: tracks[target_id]}
            for i in range(target_trajectory.shape[0]):
                tracks, indexed_detections = self.tracking_func(tracks, (self.attack_start_frame_id + i) * 0.1, target_trajectory[np.newaxis, i, :])
                if target_id in indexed_detections:
                    target_trajectory[i] = indexed_detections[target_id]

            target_trajectory_list.append(target_trajectory)

        for frame_index, frame_id in enumerate(range(start_frame_id, end_frame_id + 1)):
            observed_trajectories_copy = copy.deepcopy(case[frame_id][attacker_vehicle_id]["observed_trajectories"])
            copy_length = min(self.history_num_frames, frame_id - self.attack_start_frame_id + 1)

            use_obs_noise = self.observation_noise_std > 0 and copy_length > 0

            if use_obs_noise:
                K = self.observation_noise_k
                observed_trajectories_list = []
                for i in range(batch_size):
                    for k in range(K):
                        obs = copy.deepcopy(observed_trajectories_copy)
                        obs[target_id][-copy_length:] = target_trajectory_list[i][:copy_length].copy()
                        noise = np.random.normal(0, self.observation_noise_std, (copy_length, 2))
                        obs[target_id][-copy_length:, :2] += noise
                        observed_trajectories_list.append(obs)
            else:
                observed_trajectories_list = [copy.deepcopy(observed_trajectories_copy) for i in range(batch_size)]
                if copy_length > 0:
                    for i in range(batch_size):
                        observed_trajectories_list[i][target_id][-copy_length:] = target_trajectory_list[i][:copy_length]

            model_args = {"model_api": self.prediction_model_api,
                         "obs_length": self.history_num_frames,
                         "pred_length": self.predict_num_frames}
            predicted_trajectories_list = self.prediction_func(observed_trajectories_list, model_args=model_args, num_frames=self.predict_num_frames, object_ids=[target_id], batch=True, requires_grad=False)

            victim_trajectory = case[frame_id][attacker_vehicle_id]["predicted_trajectories"][victim_vehicle_id][:self.predict_num_frames].copy()
            if not use_obs_noise and self.use_uncertainty:
                victim_trajectory_sampled = self.trajectory_uncertainty_model.sample(victim_trajectory, k=10)

            for i in range(batch_size):
                if use_obs_noise:
                    sample_fs = []
                    for k in range(K):
                        kt = predicted_trajectories_list[i * K + k][target_id][:self.predict_num_frames]
                        f = np.log(np.clip(min_trajectory_distance(kt, victim_trajectory, requires_grad=False), 0.1, 100))
                        sample_fs.append(f)
                    fitness_list[i, frame_index] = -coeficient[frame_index] * np.mean(sample_fs)
                elif self.use_uncertainty:
                    if self.learned_uncertainty_model is not None:
                        key_sampled, _ = self.learned_uncertainty_model.sample_target_trajectories(
                            predicted_trajectories_list[i][target_id][:self.predict_num_frames], attack_opts, case, frame_id, k=10)
                        fitness_list[i, frame_index] = -coeficient[frame_index] * np.mean([
                            np.log(np.clip(min_trajectory_distance(key_sampled[j], _victim, requires_grad=False), 0.1, 100))
                            for j in range(len(key_sampled)) for _victim in victim_trajectory_sampled])
                    else:
                        fitness_list[i, frame_index] = -coeficient[frame_index] * np.mean([np.log(np.clip(min_trajectory_distance(predicted_trajectories_list[i][target_id][:self.predict_num_frames], _victim, requires_grad=False), 0.1, 100)) for _victim in victim_trajectory_sampled])
                else:
                    fitness_list[i, frame_index] = -coeficient[frame_index] * np.log(np.clip(min_trajectory_distance(predicted_trajectories_list[i][target_id][:self.predict_num_frames], victim_trajectory, requires_grad=False), 0.1, 100))
        
        fitness = fitness_list.sum(axis=1) / coeficient.sum()
        return fitness
    
    def _apply_tracking_surrogate(self, case, attack_opts, target_trajectory_slice):
        """Run target positions through differentiable Kalman surrogate."""
        attacker_vehicle_id = attack_opts["attacker_vehicle_id"]
        target_track_id = attack_opts["attacker_target_track_id"]
        pre_frame = self.attack_start_frame_id - 1
        tracks = case[pre_frame][attacker_vehicle_id].get("tracks")
        if tracks is None:
            return target_trajectory_slice[:, :2]

        prior_state, prior_P = self.kalman_surrogate.extract_prior_state(
            tracks, target_track_id)
        if prior_state is None:
            return target_trajectory_slice[:, :2]

        n = target_trajectory_slice.shape[0]
        frame_times = [(self.attack_start_frame_id + i) * 0.1
                       for i in range(n)]

        return self.kalman_surrogate.forward(
            prior_state, target_trajectory_slice[:, :2],
            frame_times, prior_P, device=self.device)

    def attack_fitness_simplified(self, case, attack_opts, frame_id=None, requires_grad=False):
        return self.attack_fitness(case, attack_opts, start_frame_id=frame_id, requires_grad=requires_grad)

    def attack_constraint(self, case, attack_opts, frame_id=None):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        frame_ids = self.attack_frame_ids
        self.apply_perturbation(case, attack_opts, requires_grad=False)
        perturbation, target_trajectory = attack_opts["perturbation"], attack_opts["target_trajectory"]

        if self.perturbation_type == "location":
            bound = getattr(self, 'location_bound', 0.5)
            hard_bound = np.sum(np.sqrt(np.sum(perturbation ** 2, axis=1)) > bound) == 0
        elif self.perturbation_type == "control":
            hard_bound = (np.sum(np.absolute(perturbation[:, 0]) > 50) + np.sum(np.absolute(perturbation[:, 1]) > 4)) == 0
        else:
            raise NotImplementedError()

        # conflict_areas_list = [case[frame_id][victim_vehicle_id]["free_areas"] for frame_id in frame_ids]
        # conflict_area = check_trajectory_overlap(target_trajectory, conflict_areas_list, threshold=3)
        # undetectable = conflict_area.sum() == 0
        undetectable = True

        return hard_bound and undetectable
    
    def init_seeds(self, case, attack_opts):
        attacker_vehicle_id, victim_vehicle_id = attack_opts["attacker_vehicle_id"], attack_opts["victim_vehicle_id"]
        frame_ids = self.attack_frame_ids
        start_frame_id, end_frame_id = frame_ids[0], frame_ids[-1]
        trajectories = case[end_frame_id][attacker_vehicle_id]["trajectories"]
        seeds = []
        
        for target_id, trajectory in trajectories.items():
            if target_id in [victim_vehicle_id, attacker_vehicle_id]:
                continue
            complete_trajectory = get_complete_trajectory(trajectory)
            if len(complete_trajectory) < self.attack_num_frames + 3:
                continue
            
            if self.perturbation_type == "location":
                init_perturbation = np.zeros((self.attack_num_frames, 2))
            elif self.perturbation_type == "control":
                init_perturbation = np.hstack((np.random.random((self.attack_num_frames, 1)) * 1 - 0.5,
                                            np.random.random((self.attack_num_frames, 1)) * 0.04 - 0.02))
            else:
                raise NotImplementedError()

            seed = copy.deepcopy(attack_opts)
            seed.update({
                "target_id": target_id,
                "perturbation": init_perturbation,
                "ideal_predicted_trajectories": {},
                "real_predicted_trajectories": {},
                "target_trajectory": None,
                "real_target_trajectory": np.zeros((self.attack_num_frames, 7)),
            })
           
            if "gt" in seed and seed["gt"]:
                victim_target_track_id = seed["target_id"]
                attacker_target_track_id = seed["target_id"]
                attacker_victim_track_id = seed["victim_vehicle_id"]
            else:
                victim_target_track_id = self.get_track_id(case, seed["victim_vehicle_id"], seed["target_id"])
                attacker_target_track_id = self.get_track_id(case, seed["attacker_vehicle_id"], seed["target_id"])
                attacker_victim_track_id = self.get_track_id(case, seed["attacker_vehicle_id"], seed["victim_vehicle_id"])
            
            seed.update({
                "victim_target_track_id": victim_target_track_id,
                "attacker_target_track_id": attacker_target_track_id,
                "attacker_victim_track_id": attacker_victim_track_id,
            })
            seeds.append(seed)

        return seeds

    def _get_step_sizes(self):
        """Get optimizer step sizes based on perturbation type."""
        if self.perturbation_type == "location":
            bound = getattr(self, 'location_bound', 0.5)
            step = bound / 4  # reach bound in ~4 steps
            return step, step  # same for x and y
        else:  # control
            return 0.01, 0.0005  # throttle, steering

    def optimize_whitebox(self, case, attack_opts, frame_id, num_interations=1):
        frame_index = frame_id - self.attack_start_frame_id
        logging.info(f"frame_id {frame_id}")
        step0, step1 = self._get_step_sizes()

        for iter in range(num_interations):
            logging.debug(f"Iteration {iter}")
            perturbation_np = attack_opts["perturbation"]
            perturbation = torch.tensor(perturbation_np, requires_grad=True).to(self.device)
            perturbation.retain_grad()
            attack_opts["perturbation"] = perturbation
            self.apply_perturbation(case, attack_opts, requires_grad=True)

            fitness = self.attack_fitness(case, attack_opts, start_frame_id=self.attack_end_frame_id,
                                          end_frame_id=self.attack_end_frame_id, requires_grad=True)
            logging.debug(f"Iteration {iter} Fitness {fitness}")

            fitness.backward()
            grad = attack_opts["perturbation"].grad.detach().cpu().numpy()
            logging.debug("Iteration {} Grad {}".format(iter, grad.T))

            if self.optimization_type == "grad":
                perturbation_np[frame_index:] += grad[frame_index:] * step0
            elif self.optimization_type == "sign":
                perturbation_np[frame_index:, 0] += (grad[frame_index:, 0] > 0) * step0
                perturbation_np[frame_index:, 0] -= (grad[frame_index:, 0] < 0) * step0
                perturbation_np[frame_index:, 1] += (grad[frame_index:, 1] > 0) * step1
                perturbation_np[frame_index:, 1] -= (grad[frame_index:, 1] < 0) * step1
            else:
                raise NotImplementedError()

            attack_opts["perturbation"] = perturbation_np
            logging.debug("Iteration {} Perturbation {}".format(iter, attack_opts["perturbation"].T))
            self.attack_projection(case, attack_opts, frame_id)
            logging.debug("Iteration {} Rescaled perturbation {}".format(iter, attack_opts["perturbation"].T))

        logging.info("Frame {} Perturbation {}".format(frame_id, attack_opts["perturbation"].T))
        self.apply_perturbation(case, attack_opts, requires_grad=False)
        return attack_opts
    
    def optimize_blackbox(self, case, attack_opts, frame_id, num_interations=1):
        frame_index = frame_id - self.attack_start_frame_id
        logging.info(f"frame_id {frame_id}")
        step0, step1 = self._get_step_sizes()

        def forward(perturbation_list):
            batch_size = len(perturbation_list)
            outputs = np.zeros(batch_size)
            tmp_attack_opts = copy.deepcopy(attack_opts)
            outputs = self.attack_fitness_blackbox_batch(case, tmp_attack_opts, perturbation_list,
                                                         start_frame_id=self.attack_end_frame_id,
                                                         end_frame_id=self.attack_end_frame_id)
            return outputs

        for iter in range(num_interations):
            logging.debug(f"Iteration {iter}")
            perturbation_np = attack_opts["perturbation"]
            self.apply_perturbation(case, attack_opts, requires_grad=False)
            fitness = self.attack_fitness(case, attack_opts, start_frame_id=self.attack_end_frame_id,
                                          end_frame_id=self.attack_end_frame_id, requires_grad=False)
            logging.debug(f"Iteration {iter} Fitness {fitness}")

            grad = gradient_estimation(attack_opts["perturbation"], forward, delta=0.05)
            logging.debug("Iteration {} Grad {}".format(iter, grad.T))

            if self.optimization_type == "grad":
                perturbation_np[frame_index:] += grad[frame_index:] * step0
            elif self.optimization_type == "sign":
                perturbation_np[frame_index:, 0] += (grad[frame_index:, 0] > 0) * step0
                perturbation_np[frame_index:, 0] -= (grad[frame_index:, 0] < 0) * step0
                perturbation_np[frame_index:, 1] += (grad[frame_index:, 1] > 0) * step1
                perturbation_np[frame_index:, 1] -= (grad[frame_index:, 1] < 0) * step1
            else:
                raise NotImplementedError()

            attack_opts["perturbation"] = perturbation_np
            self.attack_projection(case, attack_opts, frame_id)
            logging.debug("Iteration {} Perturbation {}".format(iter, attack_opts["perturbation"].T))

        logging.info("Frame {} Perturbation {}".format(frame_id, attack_opts["perturbation"].T))
        self.apply_perturbation(case, attack_opts, requires_grad=False)
        return attack_opts
    
    def optimize(self, case, attack_opts, frame_id, num_interations=None):
        if num_interations is None:
            num_interations = getattr(self, 'opt_iterations', 5)
        # Reset learned uncertainty model state at each new frame
        if self.learned_uncertainty_model is not None:
            if frame_id == self.attack_start_frame_id:
                self.learned_uncertainty_model.reset()

        if self.no_online_update and frame_id != self.attack_start_frame_id:
            self.apply_perturbation(case, attack_opts, requires_grad=False)
            logging.info(f"frame_id {frame_id}")
            logging.info(f"Frame {frame_id} Perturbation {attack_opts['perturbation'].T}")
            return attack_opts

        if self.attack_type == "whitebox":
            return self.optimize_whitebox(case, attack_opts, frame_id, num_interations=num_interations)
        elif self.attack_type == "blackbox":
            return self.optimize_blackbox(case, attack_opts, frame_id, num_interations=num_interations)
        else:
            raise NotImplementedError("No attack type appliable.")