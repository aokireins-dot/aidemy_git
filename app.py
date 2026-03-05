import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 最新の.keras形式を読み込む最もシンプルな方法
@st.cache_resource # モデルを一度読み込んだらメモリに保持して高速化します
def load_my_model():
    return tf.keras.models.load_model('meat_quality_model.keras')

model = load_my_model()

st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の画像をアップロードして、AI判定を開始しましょう！")

uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # 前処理
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 判定
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")

