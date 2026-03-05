import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# パソコン（GitHubのリポジトリ内）にあるモデルを直接読み込む設定
# 名前が 'meat_quality_model.h5' で一致していることを前提としています
model = tf.keras.models.load_model('meat_quality_model.keras')

st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の写真をアップロードすると、AIが「合格」か「不合格」かを判定します。")

uploaded_file = st.file_uploader("お肉の画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を表示
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # AIが読める形に加工
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 判定実行
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    st.divider()
    
    # 結果表示
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")