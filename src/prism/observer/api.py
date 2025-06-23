# src/prism/observer/api.py

import pickle
import os
from typing import Dict, Any

from prism.observer.algorithm import RemainingTimeEstimator, InterventionPolicy, BaselinePolicy
from prism.tracker.algorithm import ViterbiTracker
from prism.tracker.algorithm.utils import get_graph, get_raw_cm
from prism import config

class ObserverAPI:
    def __init__(self, task_name: str, policy_config: Dict[Any, Dict[str, Any]] = None):
        """
        Initializes the ObserverAPI with the specified task and policy configuration.

        Args:
            task_name (str): Name of the task (e.g., 'latte_making').
            policy_config (Dict[Any, Dict[str, Any]]): Configuration for intervention policies.
                Example:
                {
                    target_step: {'h_threshold': 0.3, 'offset': 15},
                    ...
                }
        """
        self.task_name = task_name
        self.policy_config = policy_config if policy_config else {}
        self.task_dir = os.path.join(config.datadrive, 'tasks', self.task_name)
        self.graph = get_graph(self.task_name)
        self.cm = get_raw_cm(self.task_name)
        self.tracker = ViterbiTracker(self.graph, confusion_matrix=self.cm)
        self.estimator = RemainingTimeEstimator(self.graph)
        self.policies = self._initialize_policies()

    def _initialize_policies(self) -> Dict[Any, InterventionPolicy]:
        """
        Initializes intervention policies based on the provided configuration.

        Returns:
            Dict[Any, InterventionPolicy]: A dictionary mapping target steps to their policies.
        """
        policies = {}
        for target_step, config_dict in self.policy_config.items():
            policies[target_step] = InterventionPolicy(
                target_step=target_step,
                h_threshold=config_dict.get('h_threshold', 0.3),
                offset=config_dict.get('offset', 15)
            )
        return policies

    def load_models(self, test_pids: list = None):
        """
        Loads the necessary models and data for the specified test participants.

        Args:
            test_pids (list, optional): List of participant IDs to load. If None, loads all available.
        """
        if test_pids is None:
            test_pids = [
                fp.stem for fp in os.listdir(os.path.join(self.task_dir, 'models', 'lopo'))
                if os.path.isfile(os.path.join(self.task_dir, 'models', 'lopo', fp))
            ]
        self.test_pids = test_pids
        self.pid_data = {}
        for pid in self.test_pids:
            pid_dir = os.path.join(self.task_dir, 'models', 'lopo', pid)
            with open(os.path.join(pid_dir, 'pred_raw.pkl'), 'rb') as f:
                raw_pred_probas = pickle.load(f)
            with open(os.path.join(pid_dir, 'true.pkl'), 'rb') as f:
                y_test = pickle.load(f)
            self.pid_data[pid] = {
                'raw_pred_probas': raw_pred_probas,
                'y_test': y_test
            }

    def calculate_remaining_time(self, test_pid: str):
        """
        Runs the remaining time estimation algorithm for a specific participant.

        Args:
            test_pid (str): Participant ID.
        """
        if test_pid not in self.pid_data:
            raise ValueError(f"Test PID {test_pid} not loaded. Please load the models first.")

        data = self.pid_data[test_pid]
        raw_pred_probas = data['raw_pred_probas']
        y_test = data['y_test']

        self.tracker.reset()
        self.estimator.reset()

        ground_truth = {}
        time = 0
        for raw_pred_prob in raw_pred_probas:
            self.tracker.forward(raw_pred_prob)
            self.estimator.forward(self.tracker.curr_entries)
            time += 1
            if time == len(y_test):
                break
            if y_test[time - 1] != y_test[time] and self.graph.steps[y_test[time] + 1] not in ground_truth:
                ground_truth[self.graph.steps[y_test[time] + 1]] = time

        save_dir = os.path.join(self.task_dir, 'observer', 'lopo', test_pid)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'dt_distribution.pkl'), 'wb') as f:
            pickle.dump({
                'expectations': self.estimator.expectations,
                'entropys': self.estimator.entropys,
                'ground_truth': ground_truth
            }, f)

    def evaluate_policy(self, test_pid: str):
        """
        Evaluates the intervention policy for a specific participant.

        Args:
            test_pid (str): Participant ID.
        """
        if test_pid not in self.pid_data:
            raise ValueError(f"Test PID {test_pid} not loaded. Please load the models first.")

        # Load dt_distribution.pkl
        dt_dist_path = os.path.join(self.task_dir, 'observer', 'lopo', test_pid, 'dt_distribution.pkl')
        if not os.path.exists(dt_dist_path):
            raise FileNotFoundError(f"dt_distribution.pkl not found for PID {test_pid}.")

        with open(dt_dist_path, 'rb') as f:
            dt_distribution = pickle.load(f)

        # Placeholder for policy evaluation logic
        # This should include loading training data, finding best thresholds, and saving results
        # Refer to evaluate_policy.py for detailed implementation

        print(f"Policy evaluation for PID {test_pid} is not yet implemented in ObserverAPI.")

    def run_all(self):
        """
        Runs the complete pipeline: calculating remaining time and evaluating policies for all loaded participants.
        """
        for pid in self.test_pids:
            print(f"Processing PID: {pid}")
            self.calculate_remaining_time(pid)
            self.evaluate_policy(pid)
            print(f"Completed PID: {pid}")

    def get_intervention(self, pid: str, step: Any) -> Dict[str, Any]:
        """
        Retrieves the intervention details for a specific participant and step.

        Args:
            pid (str): Participant ID.
            step (Any): Target step.

        Returns:
            Dict[str, Any]: Intervention details including triggered time and threshold.
        """
        step_time_dict_path = os.path.join(self.task_dir, 'observer', 'lopo', pid, 'step_time_dict.pkl')
        step_threshold_dict_path = os.path.join(self.task_dir, 'observer', 'lopo', pid, 'step_threshold_dict.pkl')

        if not os.path.exists(step_time_dict_path) or not os.path.exists(step_threshold_dict_path):
            raise FileNotFoundError(f"Policy evaluation results not found for PID {pid}.")

        with open(step_time_dict_path, 'rb') as f:
            step_time_dict = pickle.load(f)
        with open(step_threshold_dict_path, 'rb') as f:
            step_threshold_dict = pickle.load(f)

        intervention_info = {
            'ground_truth': step_time_dict.get(step, {}).get('ground_truth'),
            'proposed': step_time_dict.get(step, {}).get('proposed'),
            'baseline': step_time_dict.get(step, {}).get('baseline'),
            'h_threshold': step_threshold_dict.get(step, None)
        }

        return intervention_info
