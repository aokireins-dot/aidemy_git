import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# モデルの読み込み
@st.cache_resource
def load_my_model():
    try:
        # Colabと同じ2.19.0環境で読み込みます
        return tf.keras.models.load_model('final_meat_model.keras')
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        return None

model = load_my_model()

st.title("🥩 焼肉・肉質判定アプリ")

# 安全装置：モデルがない場合はここで止める
if model is None:
    st.warning("現在、AIの準備ができていません。デプロイ完了までお待ちください。")
    st.stop()

uploaded_file = st.file_uploader("画像を選択...", type=["jpg", "jpeg", "png"])

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