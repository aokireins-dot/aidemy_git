import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import numpy as np

# 【重要】Colabでの学習時と全く同じ構造を定義します
def create_model():
    # ベースモデルの読み込み
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights=None  # 重みは後でロードするのでNone
    )
    base_model.trainable = False # 学習済み部分を固定
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid') # 2値分類（合格/不合格）
    ])
    return model

# モデルを生成して重みを読み込む
try:
    model = create_model()
    # ファイル名が 'meat_weights.weights.h5' であることを確認してください
    model.load_weights('meat_weights.weights.h5')
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")

st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の写真をアップロードして判定を開始してください。")

uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # 前処理（Colabでの学習時と同じ設定にします）
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.info(f"スコア: {score:.4f}")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.info(f"スコア: {score:.4f}")