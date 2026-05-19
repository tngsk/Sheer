import serial
import time
import numpy as np

class SensorInput:
    """
    Handles serial connection to the sensor device and provides a generator
    to yield incoming data.
    """
    def __init__(self, port='COM3', baud_rate=115200, sample_rate_hz=200):
        self.port = port
        self.baud_rate = baud_rate
        self.sample_rate_hz = sample_rate_hz
        self.ser = None
        self.demo_mode = False

        # for demo mode
        self.last_a = 1500
        self.last_b = 1500

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            print(f"Connected to {self.port} at {self.baud_rate} bps.")
        except serial.SerialException as e:
            print(f"Warning: Could not open serial port {self.port}. Error: {e}")
            print("Running in DEMO mode with random data for testing.")
            self.demo_mode = True

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()

    def read_stream(self):
        """
        Generator that yields (val_a, val_b) tuples as they arrive.
        """
        while True:
            if not self.demo_mode and self.ser and self.ser.is_open:
                # Read all available lines in waiting
                while self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8').strip()
                        if line:
                            parts = line.split(',')
                            if len(parts) == 3:
                                _, val_a, val_b = parts
                                yield int(val_a), int(val_b)
                    except Exception:
                        pass # Ignore malformed lines
                # Small sleep to prevent busy waiting if no data
                time.sleep(0.001)
            else:
                # Demo mode
                time.sleep(1.0 / self.sample_rate_hz)
                self.last_a = max(0, min(4095, self.last_a + np.random.randint(-50, 50)))
                self.last_b = max(0, min(4095, self.last_b + np.random.randint(-50, 50)))
                yield self.last_a, self.last_b
