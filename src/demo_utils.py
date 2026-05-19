import random
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


@st.cache_resource
def load_assets(model_path: str, x_test_path: str, y_test_path: str):
    model = load_model(model_path)
    X_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    y_pred_all = np.argmax(model.predict(X_test, verbose=1), axis=1)

    return model, X_test, y_test, y_pred_all


def generate_fake_mac(device_id: int) -> str:
    return f"02:00:00:00:00:{device_id + 1:02x}"


def build_whitelist(y_test):
    device_ids = sorted(np.unique(y_test).astype(int).tolist())

    device_to_mac = {
        device_id: generate_fake_mac(device_id)
        for device_id in device_ids
    }

    mac_to_device = {
        mac.lower(): device_id
        for device_id, mac in device_to_mac.items()
    }

    return device_to_mac, mac_to_device


def get_device_stats(y_test, y_pred_all):
    rows = []

    for device_id in sorted(np.unique(y_test).astype(int).tolist()):
        mask = y_test == device_id
        total = int(np.sum(mask))
        correct = int(np.sum((y_test == device_id) & (y_pred_all == device_id)))
        accuracy = correct / total if total > 0 else 0

        rows.append(
            {
                "device_id": device_id,
                "device": f"Device_{device_id}",
                "correct": correct,
                "total": total,
                "accuracy": round(accuracy, 4),
                "mac": generate_fake_mac(device_id),
            }
        )

    rows = sorted(rows, key=lambda x: x["correct"], reverse=True)
    return rows


def choose_correct_rf_sample(y_test, y_pred_all, mode: str, target_device: int, seed: int = 42):
    random.seed(seed)

    if mode == "legitimate":
        candidate_indices = np.where(
            (y_test == target_device) &
            (y_pred_all == target_device)
        )[0]

    elif mode == "spoofing":
        candidate_indices = np.where(
            (y_test != target_device) &
            (y_test == y_pred_all)
        )[0]

    else:
        raise ValueError("mode phải là legitimate hoặc spoofing")

    if len(candidate_indices) == 0:
        raise RuntimeError(
            "Không tìm thấy sample phù hợp mà model dự đoán đúng. "
            "Hãy chọn device khác."
        )

    chosen_idx = random.choice(candidate_indices)

    return {
        "sample_index": int(chosen_idx),
        "true_device": int(y_test[chosen_idx]),
        "predicted_device": int(y_pred_all[chosen_idx]),
    }


def run_legitimate_demo(X_test, y_test, y_pred_all, device_to_mac, claimed_device: int, seed: int = 42):
    sample = choose_correct_rf_sample(
        y_test=y_test,
        y_pred_all=y_pred_all,
        mode="legitimate",
        target_device=claimed_device,
        seed=seed,
    )

    claimed_mac = device_to_mac[claimed_device]

    return {
        "scenario": "Legitimate",
        "claimed_mac": claimed_mac,
        "claimed_mac_owner": f"Device_{claimed_device}",
        "rf_true_device": f"Device_{sample['true_device']}",
        "rf_predicted_device": f"Device_{sample['predicted_device']}",
        "sample_index": sample["sample_index"],
        "status": "LEGITIMATE",
    }


def run_spoofing_demo(X_test, y_test, y_pred_all, device_to_mac, victim_device: int, seed: int = 42):
    sample = choose_correct_rf_sample(
        y_test=y_test,
        y_pred_all=y_pred_all,
        mode="spoofing",
        target_device=victim_device,
        seed=seed,
    )

    claimed_mac = device_to_mac[victim_device]

    return {
        "scenario": "MAC Spoofing",
        "claimed_mac": claimed_mac,
        "claimed_mac_owner": f"Device_{victim_device}",
        "attacker_rf_true_device": f"Device_{sample['true_device']}",
        "rf_predicted_device": f"Device_{sample['predicted_device']}",
        "sample_index": sample["sample_index"],
        "status": "SPOOFING",
    }


def run_manual_mac_demo(
    X_test,
    y_test,
    y_pred_all,
    device_to_mac,
    mac_to_device,
    claimed_mac: str,
    spoofing: bool,
    seed: int = 42,
):
    claimed_mac = claimed_mac.strip().lower()

    if claimed_mac not in mac_to_device:
        return {
            "claimed_mac": claimed_mac,
            "status": "UNKNOWN_MAC",
            "reason": "MAC không nằm trong whitelist giả lập.",
        }

    claimed_device = mac_to_device[claimed_mac]

    if spoofing:
        sample = choose_correct_rf_sample(
            y_test=y_test,
            y_pred_all=y_pred_all,
            mode="spoofing",
            target_device=claimed_device,
            seed=seed,
        )

        return {
            "scenario": "MAC Spoofing",
            "claimed_mac": claimed_mac,
            "claimed_mac_owner": f"Device_{claimed_device}",
            "attacker_rf_true_device": f"Device_{sample['true_device']}",
            "rf_predicted_device": f"Device_{sample['predicted_device']}",
            "sample_index": sample["sample_index"],
            "status": "SPOOFING",
        }

    sample = choose_correct_rf_sample(
        y_test=y_test,
        y_pred_all=y_pred_all,
        mode="legitimate",
        target_device=claimed_device,
        seed=seed,
    )

    return {
        "scenario": "Legitimate",
        "claimed_mac": claimed_mac,
        "claimed_mac_owner": f"Device_{claimed_device}",
        "rf_true_device": f"Device_{sample['true_device']}",
        "rf_predicted_device": f"Device_{sample['predicted_device']}",
        "sample_index": sample["sample_index"],
        "status": "LEGITIMATE",
    }
