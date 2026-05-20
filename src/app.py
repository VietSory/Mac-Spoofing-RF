import streamlit as st
import pandas as pd

from demo_utils import (
    load_assets,
    build_whitelist,
    get_device_stats,
    run_legitimate_demo,
    run_spoofing_demo,
    run_manual_mac_demo,
)

st.set_page_config(
    page_title="MAC Spoofing Detection Demo",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ MAC Spoofing Detection using RF Fingerprinting")
st.caption(
    "Demo mô phỏng hệ thống phát hiện thiết bị giả mạo MAC bằng CNN 1D và RF fingerprint từ WiSig."
)

with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_path = st.text_input(
        "Model path",
        value="../models/model_cnn1d_equalized_date_split.keras"
    )
    x_test_path = st.text_input(
        "X test path",
        value="../outputs/X_test_equalized_date_split.npy"
    )
    y_test_path = st.text_input(
        "y test path",
        value="../outputs/y_test_equalized_date_split.npy"
    )
    random_seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1
    )

    st.divider()
    st.markdown("### Demo mode")
    mode = st.radio(
        "Chọn chế độ",
        [
            "Tổng quan",
            "Whitelist",
            "Legitimate demo",
            "Spoofing demo",
            "Nhập MAC từ Kali"
        ]
    )

try:
    with st.spinner("Đang load model và test data..."):
        model, X_test, y_test, y_pred_all = load_assets(
            model_path=model_path,
            x_test_path=x_test_path,
            y_test_path=y_test_path
        )
except Exception as e:
    st.error("Không load được model hoặc test data.")
    st.exception(e)
    st.stop()

device_to_mac, mac_to_device = build_whitelist(y_test)
device_stats = get_device_stats(y_test, y_pred_all)
num_devices = len(device_to_mac)

total_samples = len(y_test)
correct_samples = int((y_test == y_pred_all).sum())
test_acc = correct_samples / total_samples if total_samples else 0

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Số thiết bị", num_devices)
metric_col2.metric("Số mẫu test", f"{total_samples:,}")
metric_col3.metric("Test accuracy", f"{test_acc:.4f}")

st.divider()

if mode == "Tổng quan":
    st.subheader("🎯 Ý tưởng demo")

    st.markdown(
        """
        Hệ thống không tin tuyệt đối vào địa chỉ MAC, vì MAC có thể bị đổi bằng Kali hoặc công cụ tương tự.

        Quy trình demo:

        1. Mỗi thiết bị hợp lệ được gán một MAC trong whitelist.
        2. CNN 1D nhận diện thiết bị thật dựa trên RF fingerprint.
        3. Nếu thiết bị dự đoán từ RF không khớp với chủ sở hữu MAC khai báo, hệ thống báo **MAC Spoofing**.
        """
    )

    st.info(
        "Lưu ý: Website này không capture RF live từ Kali. Kali dùng để chứng minh MAC có thể bị đổi thật. "
        "RF fingerprint được mô phỏng bằng sample từ WiSig test set."
    )

    st.subheader("📊 Device gợi ý để demo")
    st.write("Nên chọn các device có nhiều sample được dự đoán đúng để demo mượt hơn.")

    df_stats = pd.DataFrame(device_stats)
    st.dataframe(df_stats, use_container_width=True)

elif mode == "Whitelist":
    st.subheader("📋 Whitelist giả lập")

    whitelist_df = pd.DataFrame(
        [
            {"device": f"Device_{device_id}", "mac": mac}
            for device_id, mac in device_to_mac.items()
        ]
    )

    st.dataframe(whitelist_df, use_container_width=True)

    st.markdown(
        """
        Ví dụ nếu Kali đổi MAC thành MAC của `Device_27`, thì hệ thống sẽ hiểu thiết bị đang khai báo danh tính là `Device_27`.

        Nhưng hệ thống vẫn kiểm tra thêm RF fingerprint để xác minh thiết bị thật.
        """
    )

elif mode == "Legitimate demo":
    st.subheader("✅ Legitimate demo")
    st.write("Kịch bản: thiết bị dùng đúng MAC của chính nó.")

    good_devices = [row["device_id"] for row in device_stats if row["correct"] > 0]

    selected_device = st.selectbox(
        "Chọn device hợp lệ",
        options=good_devices,
        format_func=lambda x: f"Device_{x} — {device_to_mac[x]}"
    )

    if st.button("Run legitimate demo", type="primary"):
        result = run_legitimate_demo(
            X_test=X_test,
            y_test=y_test,
            y_pred_all=y_pred_all,
            device_to_mac=device_to_mac,
            claimed_device=selected_device,
            seed=int(random_seed)
        )

        st.success("Kết luận: LEGITIMATE")
        st.json(result)

        st.markdown(
            f"""
            **Giải thích:** MAC khai báo thuộc `{result["claimed_mac_owner"]}` và RF fingerprint cũng được CNN nhận diện là `{result["rf_predicted_device"]}`.  
            Vì hai thông tin khớp nhau nên hệ thống kết luận thiết bị hợp lệ.
            """
        )

elif mode == "Spoofing demo":
    st.subheader("🚨 Spoofing demo")
    st.write("Kịch bản: một thiết bị thật dùng MAC của thiết bị khác.")

    victim_device = st.selectbox(
        "Chọn device bị giả mạo MAC",
        options=list(device_to_mac.keys()),
        format_func=lambda x: f"Device_{x} — {device_to_mac[x]}"
    )

    if st.button("Run spoofing demo", type="primary"):
        result = run_spoofing_demo(
            X_test=X_test,
            y_test=y_test,
            y_pred_all=y_pred_all,
            device_to_mac=device_to_mac,
            victim_device=victim_device,
            seed=int(random_seed)
        )

        st.error("Kết luận: MAC SPOOFING DETECTED")
        st.json(result)

        st.markdown(
            f"""
            **Giải thích:** Thiết bị khai báo MAC của `{result["claimed_mac_owner"]}`, nhưng RF fingerprint được CNN nhận diện là `{result["rf_predicted_device"]}`.  
            Vì hai danh tính không khớp nên hệ thống cảnh báo **MAC Spoofing**.
            """
        )

elif mode == "Nhập MAC từ Kali":
    st.subheader("💻 Nhập MAC từ Kali")
    st.write(
        "Sau khi đổi MAC trên Kali, nhập MAC đó vào đây để mô phỏng hệ thống kiểm tra whitelist + RF fingerprint."
    )

    claimed_mac = st.text_input(
        "Claimed MAC từ Kali",
        value="02:00:00:00:00:1c"
    ).strip().lower()

    rf_mode = st.radio(
        "Chọn RF sample mô phỏng",
        [
            "RF của đúng chủ MAC — legitimate",
            "RF của thiết bị khác — spoofing"
        ]
    )

    if st.button("Check MAC + RF fingerprint", type="primary"):
        result = run_manual_mac_demo(
            X_test=X_test,
            y_test=y_test,
            y_pred_all=y_pred_all,
            device_to_mac=device_to_mac,
            mac_to_device=mac_to_device,
            claimed_mac=claimed_mac,
            spoofing=("spoofing" in rf_mode.lower()),
            seed=int(random_seed)
        )

        if result["status"] == "UNKNOWN_MAC":
            st.warning("Kết luận: UNKNOWN MAC / NOT AUTHORIZED")
            st.json(result)

        elif result["status"] == "LEGITIMATE":
            st.success("Kết luận: LEGITIMATE")
            st.json(result)

        elif result["status"] == "SPOOFING":
            st.error("Kết luận: MAC SPOOFING DETECTED")
            st.json(result)

        st.markdown(
            """
            Khi thuyết trình, bạn nói rõ: Kali chứng minh MAC có thể bị đổi thật; còn phần RF fingerprint được mô phỏng bằng WiSig vì muốn capture live RF/IQ cần USRP hoặc SDR.
            """
        )
