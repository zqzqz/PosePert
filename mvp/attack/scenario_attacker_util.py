import numpy as np
import copy
import torch

from mvp.data.util import bbox_sensor_to_map
from mvp.tools.dynamic_model import KinematicBicycleModel
from mvp.tools.polygon_space import bbox_to_polygon


def is_complete_trajectory(trajectory):
    return np.sum(np.sum(trajectory, axis=1) == 0) == 0


def get_complete_trajectory(trajectory, requires_grad=False):
    result_indices = []
    for i in range(len(trajectory)):
        bbox = trajectory[len(trajectory) - 1 - i]
        if not is_none_bbox(bbox):
            result_indices.append(len(trajectory) - 1 - i)
        else:
            break
    return trajectory[result_indices[::-1]]


def is_none_bbox(bbox):
    return bbox.sum() == 0


def trajectory_distance(t1, t2, requires_grad=False):
    t1 = get_complete_trajectory(t1, requires_grad=requires_grad)
    t2 = get_complete_trajectory(t2, requires_grad=requires_grad)
    t_length = min(len(t1), len(t2))
    if t_length == 0:
        return np.array([])
    t1 = t1[len(t1) - t_length:]
    t2 = t2[len(t2) - t_length:]
    if requires_grad:
        d = torch.sqrt(torch.sum((t1[:, :2] - t2[:, :2]) ** 2, dim=1))
    else:
        d = np.sqrt(np.sum((t1[:, :2] - t2[:, :2]) ** 2, axis=1))
    return d


def min_trajectory_distance(t1, t2, requires_grad=False):
    d = trajectory_distance(t1, t2, requires_grad=requires_grad)
    return d.min()


def weighted_trajectory_distance(t1, t2, decay=0.9, cut=0.0, requires_grad=False, device="cuda:0"):
    d = trajectory_distance(t1, t2, requires_grad=requires_grad)
    t_length = len(d)
    if requires_grad:
        cut_d = torch.clip(d, cut, 100)
        m = torch.logspace(0, t_length - 1, t_length, base=decay).to(device)
        result = torch.sum(torch.log(cut_d) * m) / m.sum()
    else:
        cut_d = np.clip(d, cut, 100)
        m = np.logspace(0, t_length - 1, base=decay, num=t_length)
        result = np.sum(np.log(cut_d) * m)
        result = result / np.sum(m)
    return result


def check_trajectory_kinematic(trajectory, model_args={}):
    model = KinematicBicycleModel(**model_args)
    trajectory = get_complete_trajectory(trajectory)
    return model.check_score(get_complete_trajectory(trajectory))


def check_trajectory_overlap(trajectory, areas_list, threshold=0.6):
    overlap = []
    for index, bbox in enumerate(trajectory):
        if is_none_bbox(bbox):
            continue
        bbox_area = bbox_to_polygon(bbox)
        overlap.append(sum([max(bbox_area.intersection(area).area - threshold, 0) for area in areas_list[index]]))
    return np.asarray(overlap)


def perception_opencood(multi_vehicle_case, ego_id, model_args={}):
    model_api = model_args["model_api"]
    pred_bboxes, pred_scores = model_api.run(multi_vehicle_case, ego_id)
    return pred_bboxes


def tracking_ab3dmot(tracks, t, detections):
    info = np.ones((detections.shape[0], 1))
    bboxes, bbox_ids, _ = tracks.update(t, detections, info)
    indexed_detections = {}
    for i in range(len(bboxes)):
        indexed_detections[bbox_ids[i]] = bboxes[i]
    return tracks, indexed_detections


def prediction_kinematic(observed_trajectories, model_args={}, num_frames=20):
    model = KinematicBicycleModel(**model_args)
    predicted_trajectories = {}

    for object_id, observed_trajectory in observed_trajectories.items():
        trajectory = get_complete_trajectory(observed_trajectory)
        if len(trajectory) < 3:
            continue
        predicted_trajectory = []
        velocity, throttle, steering_angle = model.fit(trajectory)
        velocity, throttle, steering_angle = velocity[-1], throttle[-1], steering_angle[-1]
        reference_bbox = trajectory[-1]
        x, y, yaw = reference_bbox[0], reference_bbox[1], reference_bbox[6]
        for _ in range(num_frames):
            x, y, yaw, velocity, throttle, steering_angle, angular_velocity = model.step(x, y, yaw, velocity, throttle, steering_angle)
            new_bbox = [x, y, reference_bbox[2], reference_bbox[3], reference_bbox[4], reference_bbox[5], yaw]
            predicted_trajectory.append(new_bbox)
        predicted_trajectories[object_id] = np.asarray(predicted_trajectory)

    return predicted_trajectories


def prediction_linear(observed_trajectories, model_args={}, num_frames=30, object_ids=None, requires_grad=False, device="cuda:0"):
    predicted_trajectories = {}
    
    if object_ids is None:
        object_ids = list(observed_trajectories.keys())
    
    for object_id in object_ids:
        observed_trajectory = observed_trajectories[object_id]
        trajectory = get_complete_trajectory(observed_trajectory)
        if len(trajectory) < 2:
            continue
        velocity = trajectory[-1] - trajectory[-2]

        if requires_grad:
            predicted_trajectory = trajectory[-1].expand(num_frames, 7) + \
                torch.arange(1, num_frames + 1)[:, None].expand(num_frames, 7).to(device) * velocity.expand(num_frames, 7)
            predicted_trajectories[object_id] = predicted_trajectory
        else:
            velocity = trajectory[-1] - trajectory[-2]
            predicted_trajectory = []
            latest_bbox = trajectory[-1]
            for _ in range(num_frames):
                latest_bbox = latest_bbox + velocity
                predicted_trajectory.append(latest_bbox)
            predicted_trajectories[object_id] = np.asarray(predicted_trajectory)
    return predicted_trajectories


def prediction_grip(observed_trajectories, model_args={}, num_frames=20, object_ids=None, batch=False, requires_grad=False, device="cuda:0"):
    model_api = model_args["model_api"]
    if "perturbation" in model_args:
        perturbation = model_args["perturbation"]
        target_id = model_args["target_id"]

    def preprocess(observed_trajectories):
        input_data = {
            "observe_length": model_args["obs_length"] if "obs_length" in model_args else 20,
            "predict_length": model_args["pred_length"] if "pred_length" in model_args else 20,
            "time_step": model_args["time_step"] if "time_step" in model_args else 0.1,
            "feature_dimension": 5,
            "objects": {}
        }
        for vehicle_id, trajectory in observed_trajectories.items():
            traj_length = trajectory.shape[0]
            if traj_length < 2:
                continue
            if traj_length > input_data["observe_length"]:
                trajectory = trajectory[-input_data["observe_length"]:]
                traj_length = input_data["observe_length"]
            vehicle_data = {
                "type": 1,
                "complete": True,
                "visible": True,
                "observe_trace": np.zeros((input_data["observe_length"],2)),
                "observe_feature": np.zeros((input_data["observe_length"],input_data["feature_dimension"])),
                "observe_mask": np.zeros(input_data["observe_length"]),
                "future_trace": np.zeros((input_data["predict_length"],2)),
                "future_feature": np.zeros((input_data["predict_length"],input_data["feature_dimension"])),
                "predict_trace": np.zeros((input_data["predict_length"],2)),
                "future_mask": np.zeros(input_data["predict_length"])
            }
            trajectory_np = trajectory.cpu().detach().numpy() if requires_grad else trajectory
            vehicle_data["observe_trace"][-traj_length:] = trajectory_np[:, :2]
            vehicle_data["observe_feature"][-traj_length:] = trajectory_np[:, 2:]
            vehicle_data["observe_mask"] = np.sum(vehicle_data["observe_trace"] ** 2, axis=1) > 0
            input_data["objects"][str(vehicle_id)] = vehicle_data
        return input_data
    
    def postprocess(output_data):
        predicted_trajectories = {}
        for vehicle_id, vehicle_data in output_data["objects"].items():
            vehicle_id = int(vehicle_id)
            if object_ids is not None and vehicle_id not in object_ids:
                continue

            if requires_grad:
                predicted_trajectories[vehicle_id] = vehicle_data["predict_trace_tensor"][:, :2]
            else:
                predicted_trajectories[vehicle_id] = vehicle_data["predict_trace"][:, :2]
        return predicted_trajectories

    if batch:
        input_data = [preprocess(t) for t in observed_trajectories]
        if hasattr(model_api, 'run_batch'):
            output_data = model_api.run_batch(input_data)
        else:
            output_data = [model_api.run(d) for d in input_data]
        predicted_trajectories = [postprocess(o) for o in output_data]
    else:
        input_data = preprocess(observed_trajectories)
        # FIX: empty trajectory data
        if len(input_data["objects"]) == 0:
            return {}
        if "perturbation" in model_args:
            output_data = model_api.run(input_data, perturbation={"ready_value": {str(target_id): perturbation}}, backward=requires_grad)
        else:
            output_data = model_api.run(input_data, perturbation=None, backward=requires_grad)
        predicted_trajectories = postprocess(output_data)

    return predicted_trajectories


def prediction_trajectron(observed_trajectories, model_args={}, num_frames=20, object_ids=None, batch=False, requires_grad=False, device="cuda:0"):
    """Trajectron++ prediction wrapper, same interface as prediction_grip."""
    model_api = model_args["model_api"]

    def preprocess(observed_trajectories):
        obs_len = model_args.get("obs_length", 20)
        input_data = {
            "observe_length": obs_len,
            "predict_length": num_frames,
            "time_step": model_args.get("time_step", 0.1),
            "feature_dimension": 5,
            "objects": {}
        }
        for vehicle_id, trajectory in observed_trajectories.items():
            traj_length = trajectory.shape[0]
            if traj_length < 2:
                continue
            # Count actual non-zero observation frames
            traj_check = trajectory[-obs_len:] if traj_length > obs_len else trajectory
            nonzero_frames = int(np.sum(np.sum(traj_check ** 2, axis=1) > 0))
            if nonzero_frames < 2:
                continue
            if traj_length > input_data["observe_length"]:
                trajectory = trajectory[-input_data["observe_length"]:]
                traj_length = input_data["observe_length"]
            vehicle_data = {
                "type": 1,
                "complete": True,
                "visible": True,
                "observe_trace": np.zeros((input_data["observe_length"], 2)),
                "observe_feature": np.zeros((input_data["observe_length"], input_data["feature_dimension"])),
                "observe_mask": np.zeros(input_data["observe_length"]),
                "future_trace": np.zeros((input_data["predict_length"], 2)),
                "future_feature": np.zeros((input_data["predict_length"], input_data["feature_dimension"])),
                "predict_trace": np.zeros((input_data["predict_length"], 2)),
                "future_mask": np.zeros(input_data["predict_length"])
            }
            trajectory_np = trajectory.cpu().detach().numpy() if requires_grad else trajectory
            vehicle_data["observe_trace"][-traj_length:] = trajectory_np[:, :2]
            vehicle_data["observe_feature"][-traj_length:] = trajectory_np[:, 2:]
            vehicle_data["observe_mask"] = np.sum(vehicle_data["observe_trace"] ** 2, axis=1) > 0
            # Provide dummy future data so Trajectron++ finds nodes at the
            # query timestep.  The future is only used as ground-truth labels
            # during training; during z_mode inference the prediction is
            # independent of these values.
            if traj_length >= 2:
                last_pos = trajectory_np[-1, :2]
                velocity = trajectory_np[-1, :2] - trajectory_np[-2, :2]
                for fi in range(num_frames):
                    vehicle_data["future_trace"][fi] = last_pos + velocity * (fi + 1)
                vehicle_data["future_mask"][:] = 1.0
                vehicle_data["future_feature"][:] = trajectory_np[-1, 2:]
            input_data["objects"][str(vehicle_id)] = vehicle_data
        return input_data

    def postprocess(output_data):
        predicted_trajectories = {}
        for vehicle_id, vehicle_data in output_data["objects"].items():
            vehicle_id = int(vehicle_id)
            if object_ids is not None and vehicle_id not in object_ids:
                continue
            pred = vehicle_data["predict_trace"][:, :2]
            # Trajectron may predict fewer frames than requested;
            # pad with last position if needed
            if len(pred) < num_frames:
                pad = np.tile(pred[-1:], (num_frames - len(pred), 1))
                pred = np.vstack([pred, pad])
            predicted_trajectories[vehicle_id] = pred[:num_frames]
        return predicted_trajectories

    if batch:
        input_data_list = [preprocess(t) for t in observed_trajectories]
        # Run all in one pass — Trajectron doesn't have run_batch,
        # but sequential run is unavoidable; at least avoid redundant setup
        output_list = [model_api.run(d) for d in input_data_list if len(d["objects"]) > 0]
        predicted_trajectories = [postprocess(o) for o in output_list]
        return predicted_trajectories

    input_data = preprocess(observed_trajectories)
    if len(input_data["objects"]) == 0:
        return {}
    output_data = model_api.run(input_data, perturbation=None, backward=requires_grad)
    predicted_trajectories = postprocess(output_data)
    return predicted_trajectories


def gradient_estimation(input, forward_func, delta=0.05):
    input_shape = list(input.shape)
    input_num_elements = np.prod(input_shape)

    queries = np.zeros([input_num_elements * 2] + input_shape)
    for i in range(input_num_elements):
        onehot = np.zeros(input_shape, dtype=int)
        index = np.unravel_index(i, input_shape)
        onehot[index] = 1
        queries[i * 2] = input + onehot * delta
        queries[i * 2 + 1] = input - onehot * delta

    # We assume the outputs is in size of queries
    outputs = forward_func(queries)
    grad = np.zeros_like(input)
    for i in range(input_num_elements):
        index = np.unravel_index(i, input_shape)
        grad[index] = (outputs[i * 2] - outputs[i * 2 + 1]) / (2 * delta)

    return grad
