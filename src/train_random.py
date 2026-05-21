import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    BatchNormalization,
    Activation,
    MaxPooling1D,
    Dropout,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


DATASET_PATH = "../data/wisig_50tx_equalized.npz"
OUTPUT_DIR = "../outputs"
MODEL_DIR = "../models"

MODEL_PATH = os.path.join(MODEL_DIR, "model_cnn1d_equalized_random_split.keras")
HISTORY_PATH = os.path.join(OUTPUT_DIR, "training_history_equalized_random_split.png")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2

EPOCHS = 80
BATCH_SIZE = 128
LEARNING_RATE = 1e-3


def normalize_iq(X_train, X_val, X_test):
    mean = np.mean(X_train)
    std = np.std(X_train) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


def build_cnn1d(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),

        Conv1D(64, kernel_size=7, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, kernel_size=5, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(256, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        Conv1D(256, kernel_size=3, padding="same"),
        BatchNormalization(),
        Activation("relu"),

        GlobalAveragePooling1D(),

        Dense(128, activation="relu"),
        Dropout(0.4),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def plot_history(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN 1D - Equalized Signal - Random Split")
    plt.legend()
    plt.savefig(HISTORY_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Đang load dataset:", DATASET_PATH)
    dataset = np.load(DATASET_PATH, allow_pickle=True)

    X = dataset["X"]
    y = dataset["y"]

    print("X:", X.shape)
    print("y:", y.shape)
    print("Số thiết bị:", len(np.unique(y)))

    # Random split:
    # 20% toàn bộ data làm test.
    # 80% còn lại gọi là train_all.
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\nChia random split:")
    print("X_train_all:", X_train_all.shape)
    print("X_test:", X_test.shape)

    # Từ train_all tách tiếp validation.
    # VAL_SIZE = 0.2 nghĩa là 20% của train_all làm validation.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all,
        y_train_all,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_all
    )

    X_train, X_val, X_test, mean, std = normalize_iq(
        X_train,
        X_val,
        X_test
    )

    print("\nSau khi chia validation:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)

    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y))

    model = build_cnn1d(input_shape, num_classes)
    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    print("\n===== TEST RESULT - RANDOM SPLIT =====")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n===== CLASSIFICATION REPORT - RANDOM SPLIT =====")
    print(classification_report(y_test, y_pred))

    plot_history(history)

    np.save(os.path.join(OUTPUT_DIR, "X_test_equalized_random_split.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test_equalized_random_split.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "y_pred_equalized_random_split.npy"), y_pred)

    with open(os.path.join(MODEL_DIR, "normalization_equalized_random_split.pkl"), "wb") as f:
        pickle.dump(
            {
                "mean": float(mean),
                "std": float(std),
                "split_type": "random_split",
                "test_size": TEST_SIZE,
                "val_size": VAL_SIZE
            },
            f
        )

    print("\nĐã lưu:")
    print(MODEL_PATH)
    print(HISTORY_PATH)
    print("../outputs/X_test_equalized_random_split.npy")
    print("../outputs/y_test_equalized_random_split.npy")
    print("../outputs/y_pred_equalized_random_split.npy")
    print("../models/normalization_equalized_random_split.pkl")


if __name__ == "__main__":
    main()