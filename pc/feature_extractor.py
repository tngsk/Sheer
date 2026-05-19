import numpy as np

class FeatureExtractor:
    """
    Handles buffering of sensor data and extracting features for downstream use.
    """
    def __init__(self, window_size_samples, attack_window_samples, attack_threshold=50.0):
        self.window_size_samples = window_size_samples
        self.attack_window_samples = attack_window_samples
        self.attack_threshold = attack_threshold

        self.buffer_a = []
        self.buffer_b = []

    def update(self, val_a, val_b):
        """
        Add a new sample to the buffer. Maintains buffer size.
        """
        self.buffer_a.append(val_a)
        self.buffer_b.append(val_b)

        if len(self.buffer_a) > self.window_size_samples:
            self.buffer_a.pop(0)
            self.buffer_b.pop(0)

    def is_window_full(self):
        return len(self.buffer_a) == self.window_size_samples

    def has_attack_window(self):
        return len(self.buffer_a) >= self.attack_window_samples

    def check_attack(self):
        """
        Checks for sudden, sharp movements (attacks) using the short window.
        Returns: (bool: True if attack detected, float: max peak value)
        """
        if not self.has_attack_window():
            return False, 0.0

        short_a = self.buffer_a[-self.attack_window_samples:]
        short_b = self.buffer_b[-self.attack_window_samples:]

        diff_a = np.abs(np.diff(short_a))
        diff_b = np.abs(np.diff(short_b))

        peak_a = np.max(diff_a) if len(diff_a) > 0 else 0.0
        peak_b = np.max(diff_b) if len(diff_b) > 0 else 0.0

        max_peak = max(peak_a, peak_b)
        return max_peak > self.attack_threshold, max_peak

    def extract_features(self):
        """
        Extracts features from the full sliding window.
        Returns: np.array of features.
        """
        if not self.is_window_full():
            return np.zeros(14)

        a = np.array(self.buffer_a, dtype=float)
        b = np.array(self.buffer_b, dtype=float)

        mean_a, var_a, max_a, min_a = np.mean(a), np.var(a), np.max(a), np.min(a)
        mean_b, var_b, max_b, min_b = np.mean(b), np.var(b), np.max(b), np.min(b)

        diff_a = np.abs(np.diff(a))
        mav_a = np.mean(diff_a) if len(diff_a) > 0 else 0.0

        diff_b = np.abs(np.diff(b))
        mav_b = np.mean(diff_b) if len(diff_b) > 0 else 0.0

        diff_ab = a - b
        mean_diff_ab = np.mean(diff_ab)
        var_diff_ab = np.var(diff_ab)

        eps = 1e-8
        ratio_ab = a / (a + b + eps)
        mean_ratio_ab = np.mean(ratio_ab)
        var_ratio_ab = np.var(ratio_ab)

        return np.array([
            mean_a, var_a, max_a, min_a,
            mean_b, var_b, max_b, min_b,
            mav_a, mav_b,
            mean_diff_ab, var_diff_ab,
            mean_ratio_ab, var_ratio_ab
        ])
