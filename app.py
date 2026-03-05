import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

# 1. AIの構造（形）を定義する
def create_model():
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights=None)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# 2. 形を作ってから「記憶（重み）」を流し込む
model = create_model()
model.load_weights('meat_weights.weights.h5')

st.title("🥩 焼肉・肉質判定アプリ")

uploaded_file = st.file_uploader("お肉の画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")
