# 🛡️ MAC Spoofing Detection - Hệ thống phát hiện giả mạo địa chỉ MAC bằng RF Fingerprinting

Python TensorFlow Keras Streamlit WiSig CNN1D Status

Hệ thống phát hiện tấn công giả mạo địa chỉ MAC trong mạng Wi-Fi bằng học máy dựa trên đặc trưng lớp vật lý. Project sử dụng dataset WiSig ManyTx để huấn luyện mô hình CNN 1D nhận diện thiết bị thật từ tín hiệu RF/IQ, sau đó so sánh với địa chỉ MAC khai báo để phát hiện MAC Spoofing.

---

## 📑 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng nổi bật](#-tính-năng-nổi-bật)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Cài đặt & Khởi chạy](#-cài-đặt--khởi-chạy)
- [Quy trình thực nghiệm](#-quy-trình-thực-nghiệm)
- [Hướng dẫn sử dụng website demo](#-hướng-dẫn-sử-dụng-website-demo)
- [Kết quả thực nghiệm](#-kết-quả-thực-nghiệm)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Giới hạn & Hướng phát triển](#-giới-hạn--hướng-phát-triển)

---

## 📖 Giới thiệu

Trong mạng Wi-Fi, địa chỉ MAC thường được sử dụng để định danh thiết bị hoặc áp dụng cơ chế kiểm soát truy cập như MAC filtering và MAC whitelist. Tuy nhiên, địa chỉ MAC không phải là một yếu tố xác thực an toàn tuyệt đối vì nó có thể bị thay đổi bằng phần mềm.

Nếu hệ thống chỉ kiểm tra MAC address, attacker có thể giả mạo địa chỉ MAC của một thiết bị hợp lệ để vượt qua cơ chế whitelist. Vì vậy, project này xây dựng một hướng phát hiện khác:

> Không chỉ tin vào địa chỉ MAC khai báo, mà xác thực thiết bị bằng RF fingerprint ở lớp vật lý.

RF fingerprint là dấu vân tay tín hiệu vô tuyến của thiết bị. Do sai lệch phần cứng trong quá trình phát sóng, mỗi thiết bị Wi-Fi có đặc điểm tín hiệu RF riêng. Project sử dụng dữ liệu I/Q từ dataset WiSig ManyTx để huấn luyện mô hình CNN 1D nhận diện thiết bị thật.

Nguyên lý phát hiện:

```text
MAC khai báo cho biết thiết bị tự nhận là ai.
RF fingerprint cho biết thiết bị thật có khả năng là ai.

Nếu predicted_device == claimed_mac_owner
=> Legitimate

Nếu predicted_device != claimed_mac_owner
=> MAC Spoofing Detected
```

---

## 🚀 Tính năng nổi bật

### 📡 Phát hiện dựa trên RF Fingerprinting

Hệ thống không chỉ dựa vào MAC address. Thay vào đó, hệ thống dùng tín hiệu RF/IQ để xác thực thiết bị ở lớp vật lý.

### 🧠 CNN 1D Device Identification

Sử dụng mô hình CNN 1D để nhận diện thiết bị phát Wi-Fi từ tín hiệu I/Q.

```text
Input  : RF/IQ sample shape (256, 2)
Output : Device_0 ... Device_49
```

Mô hình học đặc trưng tín hiệu của từng thiết bị, từ đó dự đoán thiết bị thật phát ra tín hiệu.

### 📂 Tiền xử lý dataset WiSig ManyTx

Project xử lý file gốc `ManyTx.pkl` có cấu trúc lồng nhau:

```python
data[tx_id][rx_id][date_id][equalized_index]
```

Sau đó chuyển thành dataset sạch dạng:

```text
X.shape = (167174, 256, 2)
y.shape = (167174,)
```

Dataset sau xử lý có:

```text
50 thiết bị phát
18 receiver
4 ngày capture
167,174 mẫu tín hiệu
equalized_index = 1
```

### 🧪 Hỗ trợ hai chiến lược đánh giá

Project hỗ trợ hai kiểu train/test:

#### Date Split

```text
Train/Validation: 2021_03_01, 2021_03_08, 2021_03_15
Test           : 2021_03_23
```

Dùng để đánh giá khả năng tổng quát hóa của mô hình sang ngày capture mới.

#### Random Split Baseline

```text
Train/Validation/Test được chia ngẫu nhiên
```

Dùng làm baseline để kiểm tra mô hình có học được RF fingerprint khi train/test cùng phân phối hay không.

### 🛡️ Mô phỏng MAC Spoofing

Hệ thống tạo whitelist giả lập:

```text
Device_0  -> 02:00:00:00:00:01
Device_1  -> 02:00:00:00:00:02
...
Device_27 -> 02:00:00:00:00:1c
```

Sau đó mô phỏng hai tình huống:

```text
Legitimate: thiết bị dùng đúng MAC của nó
Spoofing  : thiết bị dùng MAC của thiết bị khác
```

### 🌐 Website demo bằng Streamlit

Website demo trực quan, gồm các chức năng:

- Xem tổng quan hệ thống.
- Xem whitelist Device-MAC.
- Demo thiết bị hợp lệ.
- Demo thiết bị giả mạo MAC.
- Hiển thị kết luận `LEGITIMATE` hoặc `MAC SPOOFING DETECTED`.

---

## 📂 Cấu trúc dự án

```text
mac-spoofing-rf/
│
├── data/                                   # Dataset gốc và dataset đã xử lý
│   ├── ManyTx.pkl                          # File gốc WiSig ManyTx
│   └── wisig_50tx_equalized.npz            # Dataset sạch dùng để train
│
├── models/                                 # Model đã train và thông tin chuẩn hóa
│   ├── model_cnn1d_equalized_date_split.keras
│   ├── model_cnn1d_equalized_random_split.keras
│   ├── normalization_equalized_date_split.pkl
│   └── normalization_equalized_random_split.pkl
│
├── outputs/                                # Kết quả train, test và demo
│   ├── training_history_equalized_date_split.png
│   ├── training_history_equalized_random_split.png
│   ├── X_test_equalized_date_split.npy
│   ├── y_test_equalized_date_split.npy
│   ├── y_pred_equalized_date_split.npy
│   ├── X_test_equalized_random_split.npy
│   ├── y_test_equalized_random_split.npy
│   ├── y_pred_equalized_random_split.npy
│   └── mac_spoofing_demo_results.csv
│
├── src/                                    # Source code chính
│   ├── 01_split_wisig.py                   # Tách ManyTx.pkl thành dataset sạch
│   ├── check_dataset.py                    # Kiểm tra shape, NaN/Inf, class, date, receiver
│   ├── 02_train_cnn1d_date_split.py        # Train CNN 1D bằng date split
│   ├── 02_train_cnn1d_random_split.py      # Train CNN 1D bằng random split baseline
│   ├── 03_mac_spoofing_demo.py             # Demo MAC spoofing bằng terminal
│   ├── app.py                              # Entry point website Streamlit
│   └── demo_utils.py                       # Logic xử lý demo, load model, chọn sample
│
├── requirements.txt                        # Thư viện cho xử lý dữ liệu và train model
├── requirements-web.txt                    # Thư viện cho website Streamlit
├── .gitignore                              # Bỏ qua dataset, model, output nặng
└── README.md                               # Tài liệu hướng dẫn project
```

---

## 🛠 Cài đặt & Khởi chạy

### 1. Yêu cầu môi trường

Khuyến nghị:

```text
Python 3.10 hoặc 3.11
```

Các thư viện chính:

```text
tensorflow
numpy
pandas
scikit-learn
matplotlib
streamlit
```

---

### 2. Tạo môi trường ảo

Tại thư mục gốc project:

```bash
python -m venv .venv
```

Kích hoạt trên Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script, chạy:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Sau đó kích hoạt lại:

```bash
.venv\Scripts\Activate.ps1
```

---

### 3. Cài thư viện

Cài thư viện cho train model:

```bash
pip install -r requirements.txt
```

Cài thư viện cho website demo:

```bash
pip install -r requirements-web.txt
```

Nếu muốn cài nhanh toàn bộ:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib streamlit
```

---

## 🧪 Quy trình thực nghiệm

### Step 1: Chuẩn bị dataset

Đặt file gốc WiSig vào:

```text
data/ManyTx.pkl
```

File gốc có cấu trúc:

```python
data[tx_id][rx_id][date_id][equalized_index]
```

Trong đó:

| Thành phần | Ý nghĩa |
|---|---|
| `tx_id` | Thiết bị phát Wi-Fi |
| `rx_id` | Receiver thu tín hiệu |
| `date_id` | Ngày capture |
| `equalized_index = 0` | Tín hiệu raw |
| `equalized_index = 1` | Tín hiệu equalized |

---

### Step 2: Tách dataset

Đứng trong thư mục `src`:

```bash
cd src
python 01_split_wisig.py
```

Kết quả tạo ra dataset sạch:

```text
data/wisig_50tx_equalized.npz
```

Dataset này chứa:

```text
X
y
tx_meta
rx_meta
date_meta
num_tx
equalized_index
```

---

### Step 3: Kiểm tra dataset

```bash
python check_dataset.py
```

Kết quả mong muốn:

```text
X shape: (167174, 256, 2)
y shape: (167174,)
Số thiết bị/class: 50
Có NaN không: False
Có Inf không: False
```

Thống kê dataset đã xử lý:

| Thành phần | Giá trị |
|---|---:|
| Số mẫu | 167,174 |
| Số thiết bị | 50 |
| Số receiver | 18 |
| Số ngày capture | 4 |
| Equalized index | 1 |
| Input shape | `(256, 2)` |

---

### Step 4: Train mô hình bằng Date Split

```bash
python 02_train_cnn1d_date_split.py
```

Date split dùng:

```text
Train/Validation: 2021_03_01, 2021_03_08, 2021_03_15
Test           : 2021_03_23
```

Kết quả đã đạt:

```text
Test accuracy      : 68.71%
Test loss          : 1.6765
Macro F1-score     : 66%
Weighted F1-score  : 69%
```

Các file được tạo:

```text
models/model_cnn1d_equalized_date_split.keras
models/normalization_equalized_date_split.pkl
outputs/training_history_equalized_date_split.png
outputs/X_test_equalized_date_split.npy
outputs/y_test_equalized_date_split.npy
outputs/y_pred_equalized_date_split.npy
```

---

### Step 5: Train mô hình bằng Random Split Baseline

```bash
python 02_train_cnn1d_random_split.py
```

Random split dùng:

```text
Train      : khoảng 64%
Validation : khoảng 16%
Test       : khoảng 20%
```

Các file được tạo:

```text
models/model_cnn1d_equalized_random_split.keras
models/normalization_equalized_random_split.pkl
outputs/training_history_equalized_random_split.png
outputs/X_test_equalized_random_split.npy
outputs/y_test_equalized_random_split.npy
outputs/y_pred_equalized_random_split.npy
```

Random split được dùng làm baseline để chứng minh mô hình có khả năng học RF fingerprint trong điều kiện train/test cùng phân phối.

---

### Step 6: Chạy demo MAC Spoofing bằng terminal

```bash
python 03_mac_spoofing_demo.py
```

Script này tạo các case:

```text
Legitimate
Spoofing
```

Kết quả được lưu tại:

```text
outputs/mac_spoofing_demo_results.csv
```

---

## 🌐 Hướng dẫn sử dụng website demo

### 1. Khởi chạy website

Từ thư mục gốc project:

```bash
streamlit run src/app.py
```

Website sẽ mở tại:

```text
http://localhost:8501
```

---

### 2. Chọn thí nghiệm

Trong sidebar, chọn một trong hai chế độ:

```text
Date split
Random split
```

Date split dùng:

```text
models/model_cnn1d_equalized_date_split.keras
outputs/X_test_equalized_date_split.npy
outputs/y_test_equalized_date_split.npy
```

Random split dùng:

```text
models/model_cnn1d_equalized_random_split.keras
outputs/X_test_equalized_random_split.npy
outputs/y_test_equalized_random_split.npy
```

---

### 3. Xem tổng quan

Mục `Tổng quan` hiển thị:

```text
Số thiết bị
Số mẫu test
Test accuracy
Device gợi ý để demo
```

Website sẽ dự đoán toàn bộ test set để tính accuracy và chọn các sample mà model dự đoán đúng.

---

### 4. Xem whitelist Device-MAC

Mục `Whitelist` hiển thị ánh xạ:

```text
Device_0  -> 02:00:00:00:00:01
Device_1  -> 02:00:00:00:00:02
...
Device_27 -> 02:00:00:00:00:1c
```

Whitelist này mô phỏng cơ chế MAC filtering trong mạng Wi-Fi.

---

### 5. Demo thiết bị hợp lệ

Mục `Legitimate demo` mô phỏng trường hợp thiết bị dùng đúng MAC của nó.

Ví dụ:

```text
Claimed MAC owner  : Device_27
RF true device     : Device_27
RF predicted device: Device_27
Kết luận           : LEGITIMATE
```

---

### 6. Demo thiết bị giả mạo MAC

Mục `Spoofing demo` mô phỏng trường hợp một thiết bị dùng MAC của thiết bị khác.

Ví dụ:

```text
Claimed MAC owner  : Device_27
RF true device     : Device_32
RF predicted device: Device_32
Kết luận           : MAC SPOOFING DETECTED
```

---

## 📊 Kết quả thực nghiệm

### Date Split Experiment

| Chỉ số | Giá trị |
|---|---:|
| Dataset | WiSig ManyTx Equalized |
| Số thiết bị | 50 |
| Input shape | `(256, 2)` |
| Train dates | 2021_03_01, 2021_03_08, 2021_03_15 |
| Test date | 2021_03_23 |
| Test accuracy | 68.71% |
| Test loss | 1.6765 |
| Macro F1-score | 66% |
| Weighted F1-score | 69% |

### Random Split Baseline

| Chỉ số | Giá trị |
|---|---:|
| Dataset | WiSig ManyTx Equalized |
| Số thiết bị | 50 |
| Input shape | `(256, 2)` |
| Split | Random train/validation/test |
| Test accuracy | Điền sau khi train |
| Test loss | Điền sau khi train |
| Macro F1-score | Điền sau khi train |
| Weighted F1-score | Điền sau khi train |

---

## 🧠 Ý nghĩa kết quả

Date split là thí nghiệm nghiêm túc hơn vì mô hình được test trên ngày capture chưa từng thấy. Accuracy 68.71% cho thấy mô hình đã học được RF fingerprint ở mức nhất định, nhưng vẫn chịu ảnh hưởng bởi thay đổi theo thời gian và điều kiện kênh truyền.

Random split thường cho kết quả cao hơn vì train và test có cùng phân phối dữ liệu. Kết quả random split dùng làm baseline để chứng minh CNN 1D có khả năng học đặc trưng RF của thiết bị.

---

## 💻 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python |
| Deep Learning | TensorFlow / Keras |
| Mô hình | CNN 1D |
| Dataset | WiSig ManyTx |
| Xử lý dữ liệu | NumPy, Pandas |
| Đánh giá mô hình | Scikit-learn |
| Biểu đồ | Matplotlib |
| Website demo | Streamlit |

---

## ⚠️ Giới hạn & Hướng phát triển

### Giới hạn hiện tại

- Website demo chưa capture RF live trực tiếp từ thiết bị thật.
- RF fingerprint trong website được mô phỏng bằng sample từ WiSig test set.
- Muốn live end-to-end cần thiết bị SDR hoặc USRP để thu tín hiệu I/Q thật.
- Accuracy date split hiện tại chưa cao, cho thấy RF fingerprint chịu ảnh hưởng bởi thời gian capture và điều kiện kênh truyền.
- Hệ thống hiện tại là mô hình thực nghiệm, chưa phải sản phẩm triển khai thực tế.

### Hướng phát triển

- Thu tín hiệu RF live bằng SDR/USRP.
- So sánh raw signal `equalized_index = 0` và equalized signal `equalized_index = 1`.
- Thử các mô hình mạnh hơn như ResNet 1D, CNN sâu hơn hoặc attention.
- Bổ sung open-set recognition để phát hiện thiết bị lạ chưa từng train.
- Thêm confidence threshold để giảm false positive và false negative.
- Bổ sung confusion matrix, log lịch sử và dashboard trực quan hơn.
- Triển khai thử nghiệm trong môi trường Wi-Fi thực tế.

---

## ✅ Tóm tắt pipeline

```text
WiSig ManyTx
→ Tách dữ liệu RF/IQ
→ Kiểm tra dataset
→ Train CNN 1D nhận diện 50 thiết bị
→ Tạo whitelist Device-MAC
→ So sánh predicted_device với claimed_mac_owner
→ Phát hiện MAC Spoofing
→ Demo bằng Streamlit
```

---

## Câu chốt của project

> MAC address có thể bị giả mạo bằng phần mềm, nhưng RF fingerprint phản ánh đặc trưng vật lý của thiết bị phát sóng. Vì vậy, việc kết hợp MAC whitelist với RF fingerprint có thể hỗ trợ phát hiện các trường hợp MAC Spoofing trong mạng Wi-Fi.
