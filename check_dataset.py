import numpy as np
from collections import Counter

DATASET_PATH = "data/wisig_50tx_equalized.npz"
# Hoặc đổi sang:
# DATASET_PATH = "outputs/wisig_50tx_raw.npz"


def print_counter(title, values, max_items=20):
    print(f"\n{title}")
    counter = Counter(values.tolist() if hasattr(values, "tolist") else values)
    for key, count in counter.most_common(max_items):
        print(f"{key}: {count}")
    print(f"Tổng số nhóm khác nhau: {len(counter)}")


def main():
    print("Đang kiểm tra file:", DATASET_PATH)

    data = np.load(DATASET_PATH, allow_pickle=True)

    print("\n===== 1. Các key trong file =====")
    print(list(data.keys()))

    print("\n===== 2. Shape dữ liệu =====")
    for key in data.keys():
        try:
            print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
        except Exception:
            print(f"{key}: {data[key]}")

    if "X" not in data or "y" not in data:
        raise ValueError("File thiếu X hoặc y. Chưa thể train.")

    X = data["X"]
    y = data["y"]

    print("\n===== 3. Kiểm tra X/y =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    if X.ndim != 3:
        print("LỖI: X không phải dữ liệu 3 chiều.")
    elif X.shape[1:] != (256, 2):
        print("CẢNH BÁO: X không có shape chuẩn (số_mẫu, 256, 2).")
    else:
        print("OK: X có shape đúng dạng (số_mẫu, 256, 2).")

    if len(X) != len(y):
        print("LỖI: số mẫu X khác số nhãn y.")
    else:
        print("OK: số mẫu X khớp với y.")

    print("\nSố lượng mẫu:", len(X))
    print("Số thiết bị/class:", len(np.unique(y)))

    print("\n===== 4. Kiểm tra NaN / Inf =====")
    print("Có NaN không:", np.isnan(X).any())
    print("Có Inf không:", np.isinf(X).any())

    print("\nGiá trị min/max/mean/std của X:")
    print("min:", np.min(X))
    print("max:", np.max(X))
    print("mean:", np.mean(X))
    print("std:", np.std(X))

    print_counter("===== 5. Số mẫu theo thiết bị y =====", y)

    if "rx_meta" in data:
        print_counter("===== 6. Số mẫu theo receiver =====", data["rx_meta"])

    if "date_meta" in data:
        print_counter("===== 7. Số mẫu theo date =====", data["date_meta"])
    else:
        print("\nCẢNH BÁO: File chưa có date_meta.")
        print("Nên tách lại file để lưu date_meta, vì cần chia train/test theo ngày.")

    if "equalized_index" in data:
        print("\n===== 8. Equalized index =====")
        print(data["equalized_index"])


if __name__ == "__main__":
    main()