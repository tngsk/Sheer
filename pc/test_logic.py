import numpy as np
from feature_extractor import FeatureExtractor
from classifier import OptionalClassifier

def test_feature_extraction():
    print("Testing Feature Extraction...")
    extractor = FeatureExtractor(window_size_samples=100, attack_window_samples=20)

    # Fill buffer
    for _ in range(100):
        extractor.update(np.random.randint(1000, 2000), np.random.randint(1000, 2000))

    features = extractor.extract_features()
    assert len(features) == 14, f"Expected 14 features, got {len(features)}"
    print("Feature extraction returned correct feature array size.")
    print(f"Sample Features: {features[:4]}...")

def test_attack_trigger():
    print("\nTesting Attack Trigger...")
    extractor = FeatureExtractor(window_size_samples=100, attack_window_samples=20, attack_threshold=50.0)

    # Flat data
    for _ in range(20):
        extractor.update(1500, 1500)
    is_attack, peak = extractor.check_attack()
    assert not is_attack, "Attack should not be triggered on flat data"

    # Sharp spike
    for _ in range(19):
        extractor.update(1500, 1500)
    extractor.update(1600, 1500)

    is_attack, peak = extractor.check_attack()
    assert is_attack, "Attack should be triggered on sharp spike"
    assert peak == 100.0, f"Expected peak 100.0, got {peak}"
    print("Attack trigger logic passed.")

def test_model():
    print("\nTesting Model Training and Prediction...")
    classifier = OptionalClassifier(use_ml=True)
    if not classifier.use_ml:
        print("ML unavailable, skipping model tests.")
        return

    success = classifier.train_dummy()
    assert success, "Model failed to train"

    test_feature = np.random.rand(14) * 2000
    proba = classifier.predict_proba(test_feature)
    assert proba is not None, "Predict proba returned None"
    assert len(proba) == 4, f"Expected 4 class probabilities, got {len(proba)}"
    assert np.isclose(sum(proba), 1.0), "Probabilities should sum to 1.0"
    print("Model training and prediction passed.")

if __name__ == "__main__":
    try:
        test_feature_extraction()
        test_attack_trigger()
        test_model()
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
