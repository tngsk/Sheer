from pythonosc.udp_client import SimpleUDPClient

class OscOutput:
    """
    Handles sending data to DAW/Max via OSC.
    """
    def __init__(self, ip="127.0.0.1", port=8000):
        self.ip = ip
        self.port = port
        self.client = SimpleUDPClient(self.ip, self.port)
        print(f"OSC Client initialized targeting {self.ip}:{self.port}")

    def send_attack(self, peak_val):
        self.client.send_message("/sheer/attack", peak_val)

    def send_probabilities(self, probas):
        if probas is None:
            return
        for class_idx, prob in enumerate(probas):
            self.client.send_message(f"/sheer/class/{class_idx}/prob", float(prob))

    def send_raw_features(self, features):
        """Send raw features if ML is disabled."""
        # For example, just send MAV values which represent movement intensity
        mav_a = features[8]
        mav_b = features[9]
        self.client.send_message("/sheer/raw/mav_a", float(mav_a))
        self.client.send_message("/sheer/raw/mav_b", float(mav_b))
