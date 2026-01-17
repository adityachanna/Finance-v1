
"""
GaussianHSMM Class - Required for loading the saved model.
Copy this file alongside your inference script.
"""
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

class GaussianHSMM:
    """
    Hidden Semi-Markov Model built on top of hmmlearn's GaussianHMM.
    """
    
    def __init__(self, n_components=3, covariance_type="diag",
                 duration_type="gamma", min_duration=5, max_duration=100,
                 n_iter=100, random_state=42, verbose=False):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.duration_type = duration_type
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose
        
        self.hmm = GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
            verbose=verbose
        )
        self.duration_params_ = None
    
    def fit(self, X):
        self.hmm.fit(X)
        states = self.hmm.predict(X)
        self.duration_params_ = {}
        for k in range(self.n_components):
            durations = self._extract_durations(states, k)
            if len(durations) > 0:
                mean_d = np.mean(durations)
                var_d = np.var(durations)
                scale = var_d / mean_d if mean_d > 0 else 1.0
                shape = mean_d / scale if scale > 0 else 1.0
                self.duration_params_[k] = {
                    'shape': max(shape, 0.5),
                    'scale': max(scale, 1.0),
                    'mean': mean_d,
                    'median': np.median(durations)
                }
            else:
                self.duration_params_[k] = {'shape': 2.0, 'scale': 10.0, 'mean': 20.0, 'median': 20.0}
        return self
    
    def _extract_durations(self, states, state_id):
        durations = []
        current_duration = 0
        for t in range(len(states)):
            if states[t] == state_id:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        if current_duration > 0:
            durations.append(current_duration)
        return durations
    
    def predict(self, X):
        states_raw = self.hmm.predict(X)
        return self._enforce_min_duration(states_raw)
    
    def _enforce_min_duration(self, states):
        n = len(states)
        states_smooth = states.copy()
        i = 0
        while i < n:
            current_state = states_smooth[i]
            j = i
            while j < n and states_smooth[j] == current_state:
                j += 1
            duration = j - i
            if duration < self.min_duration and duration < n:
                prev_state = states_smooth[i-1] if i > 0 else None
                next_state = states_smooth[j] if j < n else None
                if prev_state is not None and next_state is not None:
                    replace_state = prev_state if prev_state == next_state else prev_state
                elif prev_state is not None:
                    replace_state = prev_state
                elif next_state is not None:
                    replace_state = next_state
                else:
                    replace_state = current_state
                states_smooth[i:j] = replace_state
            i = j
        return states_smooth
    
    def score(self, X):
        return self.hmm.score(X)
    
    def get_duration_stats(self):
        return pd.DataFrame(self.duration_params_).T
