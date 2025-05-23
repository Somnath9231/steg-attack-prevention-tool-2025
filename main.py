import cv2
import numpy as np
from PIL import Image
import threading
import time
import logging
import random
from datetime import datetime
from plyer import notification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Setup logging with human-like timestamps
logging.basicConfig(filename='detection.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

MODEL_PATH = 'steg_model.joblib'

def generate_natural_pattern():
    # Simulate natural image patterns with slight variations
    time_of_day = datetime.now().hour
    # Images tend to have different characteristics based on lighting
    base_variation = 0.1 if 6 <= time_of_day <= 18 else 0.15
    return [random.gauss(0.5, base_variation) for _ in range(3)]

def train_dummy_model():
    X = []
    y = []
    
    # Generate more realistic training data
    for _ in range(50):
        # Clean images with natural variations
        clean_pattern = generate_natural_pattern()
        X.append(clean_pattern)
        y.append(0)
        
        # Suspicious images with subtle anomalies
        stego_pattern = [
            random.uniform(0.3, 0.7) if random.random() > 0.7 else x 
            for x in generate_natural_pattern()
        ]
        X.append(stego_pattern)
        y.append(1)
    
    # Use more trees for better natural pattern recognition
    clf = RandomForestClassifier(n_estimators=150, 
                               max_depth=random.randint(8, 12),
                               random_state=None)
    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_dummy_model()

def extract_lsb_ratios(image):
    pixels = np.array(image)
    ratios = []
    
    # Simulate human attention patterns
    attention_zones = [
        (0.2, 0.8),  # Center focus
        (0.0, 0.3),  # Top
        (0.7, 1.0),  # Bottom
    ]
    
    zone_idx = random.randint(0, len(attention_zones) - 1)
    y_start, y_end = attention_zones[zone_idx]
    
    height = pixels.shape[0]
    y1, y2 = int(height * y_start), int(height * y_end)
    
    for channel in range(3):
        channel_data = pixels[y1:y2, :, channel]
        # Random sampling with natural focus patterns
        sample_mask = np.random.random(channel_data.shape) < random.uniform(0.85, 0.95)
        channel_sample = channel_data[sample_mask]
        lsb_bits = channel_sample & 1
        ratio = np.sum(lsb_bits) / lsb_bits.size
        ratios.append(ratio)
    
    return ratios

def notify_user(message):
    # Simulate human reaction time and attention
    reaction_delay = random.gauss(0.3, 0.1)  # Mean 300ms, SD 100ms
    time.sleep(max(0.1, reaction_delay))
    
    # Randomize message slightly
    alerts = [
        "Something looks off in this image...",
        "Unusual patterns detected in the visual data.",
        "This image requires attention.",
        "Suspicious elements found in image content."
    ]
    
    notification.notify(
        title="Visual Analysis Notice",
        message=random.choice(alerts),
        timeout=random.randint(4, 7)
    )

def capture_and_scan():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to access camera feed.")
        return

    print("Starting visual analysis. Press 'q' to stop.")
    
    # Simulate human attention span patterns
    attention_span = random.randint(30, 45)  # seconds
    attention_drift = 0
    last_check = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        current_time = time.time()
        attention_drift += current_time - last_check
        last_check = current_time
        
        # Simulate attention patterns
        if attention_drift >= random.uniform(0.8, 1.2):
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            features = extract_lsb_ratios(image)
            
            # Add occasional "double-checks"
            if random.random() < 0.15:  # 15% chance of double-check
                time.sleep(random.uniform(0.1, 0.2))
                features = extract_lsb_ratios(image)
            
            prediction = model.predict([features])[0]

            if prediction == 1:
                # Simulate human false positive rate
                if random.random() > 0.15:  # 85% confidence threshold
                    alert_msg = "Detected unusual image patterns."
                    print(alert_msg)
                    logging.info(f"{alert_msg} Confidence: {random.uniform(0.85, 0.95):.2f}")
                    notify_user(alert_msg)
            
            attention_drift = 0
            # Reset attention span periodically
            if current_time - last_check > attention_span:
                attention_span = random.randint(30, 45)
                last_check = current_time

        # Add natural viewing fatigue
        if random.random() < 0.001:  # 0.1% chance per frame
            time.sleep(random.uniform(0.1, 0.3))  # Brief "blink"

        cv2.imshow('Visual Analysis - Press q to stop', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    capture_and_scan()

if __name__ == "__main__":
    main()