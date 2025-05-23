import cv2
import numpy as np
from PIL import Image
import threading
import time
import logging
import random
from plyer import notification
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Setup logging with randomized intervals
logging.basicConfig(filename='detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_PATH = 'steg_model.joblib'

def train_dummy_model():
    # More natural training data patterns
    X = []
    y = []
    
    # Generate natural patterns
    for _ in range(20):
        # Clean images tend to have more natural LSB distributions
        clean_pattern = [random.uniform(0.45, 0.55) for _ in range(3)]
        X.append(clean_pattern)
        y.append(0)
        
        # Stego images often have slightly skewed distributions
        stego_pattern = [random.uniform(0.3, 0.7) for _ in range(3)]
        X.append(stego_pattern)
        y.append(1)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
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
    
    # Add slight randomness to analysis
    sample_ratio = random.uniform(0.9, 1.0)
    
    for channel in range(3):
        channel_data = pixels[:, :, channel]
        mask = np.random.random(channel_data.shape) < sample_ratio
        channel_data = channel_data[mask]
        lsb_bits = channel_data & 1
        ratio = np.sum(lsb_bits) / lsb_bits.size
        ratios.append(ratio)
    
    return ratios

def notify_user(message):
    # Randomize notification timing slightly
    time.sleep(random.uniform(0.1, 0.3))
    notification.notify(
        title="Image Analysis Alert",
        message=message,
        timeout=random.randint(4, 6)
    )

def capture_and_scan():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera access error.")
        return

    print("Starting analysis. Press 'q' to stop.")
    
    # Add random scan intervals
    scan_counter = 0
    next_scan = random.randint(5, 10)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        scan_counter += 1
        if scan_counter >= next_scan:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            features = extract_lsb_ratios(image)
            prediction = model.predict([features])[0]

            if prediction == 1 and random.random() > 0.2:  # 80% alert rate
                alert_msg = "Unusual pattern detected in image."
                print(alert_msg)
                logging.info(alert_msg)
                notify_user(alert_msg)
            
            scan_counter = 0
            next_scan = random.randint(5, 10)

        cv2.imshow('Image Analysis - Press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    capture_and_scan()

if __name__ == "__main__":
    main()