import os
import random
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


OUTPUT_DIR = "../outputs"
MODEL_DIR = "../models"

MODEL_PATH = os.path.join(MODEL_DIR, "model_cnn1d_equalized_date_split.keras")
X_TEST_PATH = os.path.join(OUTPUT_DIR, "X_test_equalized_date_split.npy")
Y_TEST_PATH = os.path.join(OUTPUT_DIR, "y_test_equalized_date_split.npy")

RESULT_PATH = os.path.join(OUTPUT_DIR, "mac_spoofing_demo_results.csv")

RANDOM_STATE = 42
NUM_DEMO_SAMPLES = 5000
SPOOFING_RATIO = 0.5


def generate_fake_mac(device_id):
    """
    Tạo MAC giả lập cho từng thiết bị.
    Đây là MAC dùng trong whitelist demo, không phải MAC thật trong dataset.
    """
    return f"02:00:00:00:00:{device_id + 1:02x}"


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    print("Đang load model...")
    model = load_model(MODEL_PATH)

    print("Đang load test set...")
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    num_devices = len(np.unique(y_test))

    print("Số mẫu test:", len(X_test))
    print("Số thiết bị:", num_devices)

    # Tạo whitelist giả lập
    # Device_0 -> MAC_0, Device_1 -> MAC_1, ...
    device_to_mac = {
        device_id: generate_fake_mac(device_id)
        for device_id in range(num_devices)
    }

    mac_to_device = {
        mac: device_id
        for device_id, mac in device_to_mac.items()
    }

    total_samples = min(NUM_DEMO_SAMPLES, len(X_test))

    selected_indices = np.random.choice(
        np.arange(len(X_test)),
        size=total_samples,
        replace=False
    )

    X_demo = X_test[selected_indices]
    y_true_device = y_test[selected_indices]

    print("Đang dự đoán thiết bị thật từ RF fingerprint...")
    y_pred_device = np.argmax(model.predict(X_demo), axis=1)

    results = []

    for i in range(total_samples):
        true_device = int(y_true_device[i])
        predicted_device = int(y_pred_device[i])

        is_spoofing = random.random() < SPOOFING_RATIO

        if is_spoofing:
            # Thiết bị thật cố tình khai báo MAC của thiết bị khác
            victim_candidates = [
                d for d in range(num_devices)
                if d != true_device
            ]
            claimed_device = random.choice(victim_candidates)
            claimed_mac = device_to_mac[claimed_device]
            ground_truth_status = "Spoofing"
        else:
            # Thiết bị dùng đúng MAC của chính nó
            claimed_device = true_device
            claimed_mac = device_to_mac[true_device]
            ground_truth_status = "Legitimate"

        mac_owner_device = mac_to_device[claimed_mac]

        # Logic phát hiện MAC spoofing:
        # Nếu thiết bị model dự đoán từ RF không phải chủ sở hữu MAC khai báo
        # => Spoofing
        if predicted_device == mac_owner_device:
            detected_status = "Legitimate"
        else:
            detected_status = "Spoofing"

        results.append({
            "sample_id": int(selected_indices[i]),
            "true_device": f"Device_{true_device}",
            "predicted_device": f"Device_{predicted_device}",
            "claimed_mac": claimed_mac,
            "claimed_mac_owner": f"Device_{mac_owner_device}",
            "ground_truth_status": ground_truth_status,
            "detected_status": detected_status,
            "correct_detection": ground_truth_status == detected_status
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULT_PATH, index=False)

    y_true_status = df["ground_truth_status"]
    y_detected_status = df["detected_status"]

    detection_acc = accuracy_score(y_true_status, y_detected_status)

    print("\n===== MAC SPOOFING DETECTION RESULT =====")
    print("Tổng số mẫu demo:", len(df))
    print("Tỉ lệ spoofing giả lập:", SPOOFING_RATIO)
    print(f"Detection accuracy: {detection_acc:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true_status, y_detected_status))

    print("\nConfusion matrix:")
    print(confusion_matrix(
        y_true_status,
        y_detected_status,
        labels=["Legitimate", "Spoofing"]
    ))

    print("\nMột số dòng demo:")
    print(df.head(20).to_string(index=False))

    print(f"\nĐã lưu kết quả demo tại: {RESULT_PATH}")


if __name__ == "__main__":
    main()