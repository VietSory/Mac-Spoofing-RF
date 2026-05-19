import pickle
import csv
import json

PKL_PATH = "data/ManyTx.pkl"
CSV_PATH = "outputs/manytx_preview.csv"

def safe_value(value, max_len=300):
    """
    Chuyển dữ liệu phức tạp sang chuỗi ngắn để CSV không bị quá nặng.
    """
    text = repr(value)
    if len(text) > max_len:
        text = text[:max_len] + " ...[truncated]"
    return text

with open(PKL_PATH, "rb") as f:
    obj = pickle.load(f)

print("Loaded type:", type(obj))
print("Keys:", obj.keys())

rows = []

for key, value in obj.items():
    if isinstance(value, list):
        for i, item in enumerate(value):
            row = {
                "main_key": key,
                "index": i,
                "item_type": str(type(item)),
                "item_length": len(item) if hasattr(item, "__len__") else "",
                "item_shape": str(item.shape) if hasattr(item, "shape") else "",
                "preview": safe_value(item)
            }
            rows.append(row)
    else:
        row = {
            "main_key": key,
            "index": "",
            "item_type": str(type(value)),
            "item_length": len(value) if hasattr(value, "__len__") else "",
            "item_shape": str(value.shape) if hasattr(value, "shape") else "",
            "preview": safe_value(value)
        }
        rows.append(row)

with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "main_key",
            "index",
            "item_type",
            "item_length",
            "item_shape",
            "preview"
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Đã xuất CSV: {CSV_PATH}")
print(f"Số dòng: {len(rows)}")