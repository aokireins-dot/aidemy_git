import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# モデルの読み込み
@st.cache_resource
def load_my_model():
    # 最新の TensorFlow (Keras 3) でそのままロードします
    return tf.keras.models.load_model('final_meat_model.keras')

model = load_my_model()

st.title("🥩 焼肉・肉質判定アプリ")
st.write("画像をアップロードすると、AIが「合格」か「不合格」かを判定します。")

uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner('AIが解析しています...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")
