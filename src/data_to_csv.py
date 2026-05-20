import pickle
import csv

PKL_PATH = "../data/ManyTx.pkl"
CSV_PATH = "../outputs/manytx_data_preview.csv"

def safe_value(value, max_len=500):
    text = repr(value)
    if len(text) > max_len:
        text = text[:max_len] + " ...[truncated]"
    return text

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

data = obj["data"]

rows = []

for i, item in enumerate(data):
    row = {
        "tx_index": i,
        "item_type": str(type(item)),
        "item_length": len(item) if hasattr(item, "__len__") else "",
        "item_shape": str(item.shape) if hasattr(item, "shape") else "",
        "preview": safe_value(item)
    }

    if isinstance(item, dict):
        row["dict_keys"] = list(item.keys())
    else:
        row["dict_keys"] = ""

    rows.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "tx_index",
            "item_type",
            "item_length",
            "item_shape",
            "dict_keys",
            "preview"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Đã xuất CSV phần data: {CSV_PATH}")
print(f"Số dòng: {len(rows)}")