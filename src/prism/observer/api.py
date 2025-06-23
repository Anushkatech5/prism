import pickle
import numpy as np

from prism.tracker.api import TrackerAPI
from prism.har.api import HumanActivityRecognitionAPI
from .algorithm import RemainingTimeEstimator
from .algorithm.policy import InterventionPolicy

class ObserverAPI:
    """
    ObserverAPI provides a separate interface for monitoring tasks,
    utilizing TrackerAPI and HumanActivityRecognitionAPI independently.
    """

    def __init__(self, task_name, policy_config, allow_exceptional_transition=False):
        """
        Initializes the ObserverAPI with Tracker and HAR components.

        Args:
            task_name (str): Name of the task (e.g., 'latte_making').
            policy_config (dict): Configuration for intervention policies.
            allow_exceptional_transition (bool): Flag to allow exceptional transitions.
        """
        # Initialize Human Activity Recognition
        self.har_api = HumanActivityRecognitionAPI(task_name=task_name)

        # Initialize Tracker
        self.tracker_api = TrackerAPI(task_name=task_name, allow_exceptional_transition=allow_exceptional_transition)

        # Initialize Remaining Time Estimator
        self.remaining_time_estimator = RemainingTimeEstimator(self.tracker_api.graph, mc_samples=1000)

        # Initialize Intervention Policies
        self.policies = {}
        for step, config in policy_config.items():
            self.policies[step] = InterventionPolicy(
                target_step=step,
                h_threshold=config['h_threshold'],
                offset=config.get('offset', 0)
            )

    def process_data(self, data):
        """
        Processes incoming data by passing it through HAR and Tracker APIs,
        then evaluates intervention policies.

        Args:
            data (dict): Dictionary containing 'audio' and 'motion' data.

        Returns:
            dict: Contains HAR probabilities, tracking probabilities, and any interventions triggered.
        """
        # Step 1: Human Activity Recognition
        har_probs = self.har_api(data)  # List of probabilities

        # Step 2: Tracking
        tracking_probs = self.tracker_api(har_probs)  # List of probabilities

        # Step 3: Get Current Context from Tracker
        context = self.tracker_api.get_current_context()

        # Step 4: Remaining Time Estimation
        expectations, entropys = self.remaining_time_estimator.forward(context['history'])

        # Step 5: Evaluate Intervention Policies
        interventions = {}
        for step, policy in self.policies.items():
            e = expectations.get(step, 0.0)
            h = entropys.get(step, 0.0)
            status, timer_duration = policy.forward(e, h)
            if status == 'timer_start':
                interventions[step] = {'action': 'start_timer', 'duration': timer_duration}
            elif status == 'timer_stop':
                interventions[step] = {'action': 'stop_timer'}
            # 'no_action' does not require any update

        return {
            'har_probs': har_probs,
            'tracking_probs': tracking_probs,
            'interventions': interventions
        }

    def reset(self):
        """
        Resets the internal states of HAR, Tracker, and Remaining Time Estimator.
        """
        self.har_api.reset()
        self.tracker_api.reset()
        self.remaining_time_estimator.reset()
        for policy in self.policies.values():
            policy.reset()
