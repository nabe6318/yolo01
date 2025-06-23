import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
from PIL import Image
import os

# モデルのロード
model = YOLO("yolov8n.pt")  # or your own trained model

st.title("YOLOv8 画像・動画検出アプリ")

# ファイルアップロード
uploaded_file = st.file_uploader("画像または動画ファイルをアップロードしてください", type=["jpg", "jpeg", "png", "mp4", "mov"])

if uploaded_file is not None:
    # 一時ファイルとして保存
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # 画像 or 動画処理の分岐
    if suffix.lower() in [".jpg", ".jpeg", ".png"]:
        st.image(temp_path, caption="元画像", use_column_width=True)
        results = model.predict(source=temp_path, save=False, conf=0.5)

        # 検出画像をPILで描画
        for r in results:
            img = r.plot()  # numpy array
            st.image(img, caption="検出結果", channels="BGR", use_column_width=True)

    elif suffix.lower() in [".mp4", ".mov"]:
        st.video(temp_path)
        st.write("動画を処理中...")
        # 検出後の動画を保存するパス
        output_path = temp_path.replace(suffix, f"_detected{suffix}")
        results = model.predict(source=temp_path, save=True, conf=0.5, project="runs", name="streamlit", exist_ok=True)

        # 結果保存先の動画を取得
        output_video_path = os.path.join("runs", "detect", "streamlit", os.path.basename(temp_path))
        if os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.error("検出結果の動画が見つかりませんでした。")
