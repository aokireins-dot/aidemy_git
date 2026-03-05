import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. モデルの読み込み（キャッシュを利用して高速化）
@st.cache_resource
def load_my_model():
    # ここに書く名前は、GitHubにあるファイル名と「完全に」一致させる必要があります
    return tf.keras.models.load_model('final_meat_model.keras')

model = load_my_model()

# 2. アプリの画面構成
st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の画像をアップロードして判定を開始してください。")

uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # 3. 前処理（学習時と同じ 224x224, 0-1に正規化）
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 4. 判定実行
    prediction = model.predict(img_array)
    score = prediction[0][0]
    
    st.divider()
    # 0.5 をしきい値として判定
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")