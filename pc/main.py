import time
import threading

from sensor_input import SensorInput
from feature_extractor import FeatureExtractor
from classifier import OptionalClassifier
from osc_output import OscOutput

# Configuration
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
OSC_IP = "127.0.0.1"
OSC_PORT = 8000

WINDOW_SIZE_MS = 500
UPDATE_INTERVAL_MS = 100
ATTACK_WINDOW_MS = 100
SAMPLE_RATE_HZ = 200
MS_PER_SAMPLE = 1000 / SAMPLE_RATE_HZ

WINDOW_SIZE_SAMPLES = int(WINDOW_SIZE_MS / MS_PER_SAMPLE)
ATTACK_WINDOW_SAMPLES = int(ATTACK_WINDOW_MS / MS_PER_SAMPLE)

def main():
    print("Project Sheer PC Client Starting...")

    # Initialize Modules (Loose Coupling)
    sensor = SensorInput(port=SERIAL_PORT, baud_rate=BAUD_RATE, sample_rate_hz=SAMPLE_RATE_HZ)
    extractor = FeatureExtractor(WINDOW_SIZE_SAMPLES, ATTACK_WINDOW_SAMPLES)
    classifier = OptionalClassifier(use_ml=True)
    osc_out = OscOutput(ip=OSC_IP, port=OSC_PORT)

    # Setup ML
    if not classifier.load_model():
        classifier.train_dummy() # Fallback to dummy for demonstration if requested

    # Connect Sensor
    sensor.connect()

    last_update_time = time.time()

    try:
        # Stream data from sensor
        for val_a, val_b in sensor.read_stream():
            current_time = time.time()

            # 1. Update Extractor Buffer
            extractor.update(val_a, val_b)

            # 2. Process at intervals (100ms)
            if (current_time - last_update_time) * 1000 >= UPDATE_INTERVAL_MS:

                # Check for attacks (fast short window)
                is_attack, peak = extractor.check_attack()
                if is_attack:
                    osc_out.send_attack(peak)
                    print(f"Attack Triggered! Peak: {peak:.2f}")

                # Process main window
                if extractor.is_window_full():
                    features = extractor.extract_features()

                    # Try ML prediction
                    probas = classifier.predict_proba(features)

                    if probas is not None:
                        # ML Mode: Send probabilities
                        osc_out.send_probabilities(probas)
                        twist_prob = probas[3] if len(probas) > 3 else 0.0
                        print(f"ML Mode | Twist Prob: {twist_prob:.3f} | MAV_A: {features[8]:.1f}, MAV_B: {features[9]:.1f}")
                    else:
                        # Raw Mode: Send basic features
                        osc_out.send_raw_features(features)
                        print(f"Raw Mode | MAV_A: {features[8]:.1f}, MAV_B: {features[9]:.1f}")

                last_update_time = current_time

    except KeyboardInterrupt:
        print("\nStopping PC Client...")
    finally:
        sensor.close()

if __name__ == "__main__":
    main()
