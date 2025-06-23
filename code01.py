import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

st.title("YOLOv8 画像検出アプリ（OpenCVなし）")

# モデルの読み込み（yolov8n.ptは最軽量）
model = YOLO("yolov8n.pt")  # または "best.pt"

# ファイルアップロード
uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # アップロード画像の一時保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        temp_path = temp.name

    # 元画像表示
    st.image(temp_path, caption="元画像", use_column_width=True)

    # YOLOで推論（save=Falseでメモリ上処理）
    results = model.predict(source=temp_path, conf=0.5, save=False)

    # 検出後の画像（PIL.Image）を取得・表示
    for r in results:
        img_pil = Image.fromarray(r.plot())  # numpy → PIL
        st.image(img_pil, caption="検出結果", use_column_width=True)

    # 一時ファイル削除
    os.remove(temp_path)
