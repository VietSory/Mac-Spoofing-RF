import os
import pickle
import numpy as np
from collections import Counter

INPUT_PATH = "data/ManyTx.pkl"
OUTPUT_DIR = "data"

NUM_TX = 50  # số thiết bị muốn lấy


def build_dataset(original_data, tx_list, rx_list, capture_date_list, equalized_index):
    """
    Cấu trúc đúng của data là:

        data[tx_id][rx_id][date_id][equalized_index]

    Trong đó:
        tx_id            = thiết bị phát
        rx_id            = receiver
        date_id          = ngày capture
        equalized_index  = 0 raw, 1 equalized

    Mỗi samples hợp lệ có shape:
        (số_mẫu, 256, 2)
    """

    X_parts = []
    y_parts = []

    tx_meta = []
    rx_meta = []
    date_meta = []

    for tx_id in range(NUM_TX):
        for rx_id in range(len(rx_list)):
            for date_id in range(len(capture_date_list)):

                try:
                    samples = original_data[tx_id][rx_id][date_id][equalized_index]
                except Exception as e:
                    print(
                        f"Lỗi truy cập tx={tx_id}, rx={rx_id}, "
                        f"date={date_id}, equalized={equalized_index}: {e}"
                    )
                    continue

                if samples is None:
                    continue

                samples = np.asarray(samples)

                # Bỏ qua mảng rỗng: shape (0, 256, 2)
                if samples.shape[0] == 0:
                    continue

                # Chỉ giữ dữ liệu RF đúng dạng
                if samples.ndim != 3 or samples.shape[1:] != (256, 2):
                    print(
                        f"Bỏ qua tx={tx_id}, rx={rx_id}, date={date_id}, "
                        f"equalized={equalized_index}, shape={samples.shape}"
                    )
                    continue

                X_parts.append(samples.astype(np.float32))

                labels = np.full(samples.shape[0], tx_id, dtype=np.int64)
                y_parts.append(labels)

                tx_meta.extend([tx_list[tx_id]] * samples.shape[0])
                rx_meta.extend([rx_list[rx_id]] * samples.shape[0])
                date_meta.extend([capture_date_list[date_id]] * samples.shape[0])

    if len(X_parts) == 0:
        raise ValueError(f"Không có dữ liệu hợp lệ cho equalized_index={equalized_index}")

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0).astype(np.int64)

    tx_meta = np.array(tx_meta)
    rx_meta = np.array(rx_meta)
    date_meta = np.array(date_meta)

    return X, y, tx_meta, rx_meta, date_meta


def save_dataset(output_path, X, y, tx_meta, rx_meta, date_meta, equalized_index):
    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        tx_meta=tx_meta,
        rx_meta=rx_meta,
        date_meta=date_meta,
        num_tx=NUM_TX,
        equalized_index=equalized_index
    )

    print("\n" + "=" * 80)
    print(f"Đã lưu: {output_path}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Số thiết bị thực tế:", len(np.unique(y)))
    print("Danh sách nhãn:", np.unique(y))
    print("Số mẫu mỗi thiết bị:")
    print(Counter(y.tolist()))
    print("=" * 80)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Đang đọc file gốc:", INPUT_PATH)

    with open(INPUT_PATH, "rb") as f:
        dataset = pickle.load(f)

    tx_list = dataset["tx_list"]
    rx_list = dataset["rx_list"]
    capture_date_list = dataset["capture_date_list"]
    original_data = dataset["data"]

    print("Số tx trong file gốc:", len(tx_list))
    print("Số rx trong file gốc:", len(rx_list))
    print("Số ngày capture:", len(capture_date_list))
    print("capture_date_list:", capture_date_list)
    print("Số tx sẽ lấy:", NUM_TX)

    if NUM_TX > len(tx_list):
        raise ValueError(f"NUM_TX={NUM_TX} lớn hơn số tx thật trong file: {len(tx_list)}")

    # RAW: equalized_index = 0
    print("\nĐang tạo dataset RAW, equalized_index = 0")

    X_raw, y_raw, tx_meta_raw, rx_meta_raw, date_meta_raw = build_dataset(
        original_data=original_data,
        tx_list=tx_list,
        rx_list=rx_list,
        capture_date_list=capture_date_list,
        equalized_index=0
    )

    save_dataset(
        output_path=os.path.join(OUTPUT_DIR, "wisig_50tx_raw.npz"),
        X=X_raw,
        y=y_raw,
        tx_meta=tx_meta_raw,
        rx_meta=rx_meta_raw,
        date_meta=date_meta_raw,
        equalized_index=0
    )

    # EQUALIZED: equalized_index = 1
    print("\nĐang tạo dataset EQUALIZED, equalized_index = 1")

    X_eq, y_eq, tx_meta_eq, rx_meta_eq, date_meta_eq = build_dataset(
        original_data=original_data,
        tx_list=tx_list,
        rx_list=rx_list,
        capture_date_list=capture_date_list,
        equalized_index=1
    )

    save_dataset(
        output_path=os.path.join(OUTPUT_DIR, "wisig_50tx_equalized.npz"),
        X=X_eq,
        y=y_eq,
        tx_meta=tx_meta_eq,
        rx_meta=rx_meta_eq,
        date_meta=date_meta_eq,
        equalized_index=1
    )

    print("\nXong. Đã tạo 2 file:")
    print("outputs/wisig_50tx_raw.npz")
    print("outputs/wisig_50tx_equalized.npz")


if __name__ == "__main__":
    main()