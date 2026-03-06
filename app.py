import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. モデルの読み込み（最新のKeras 3エンジンを使用）
@st.cache_resource
def load_my_model():
    # Google Colabで保存した最新の .keras ファイルを指定します
    return tf.keras.models.load_model('final_meat_model.keras')

# モデルをロード
try:
    model = load_my_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。ファイル名を確認してください: {e}")

# 2. アプリの画面構成
st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の画像をアップロードすると、AIが「合格」か「不合格」かを判定します。")

# 画像アップローダーの設置
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込みと表示
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # 3. AIが読める形に画像を加工（前処理）
    # 学習時と同じ224x224にリサイズし、0-1の範囲に正規化します
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # 1枚のデータとして認識させる
    
    # 4. 判定実行
    with st.spinner('AIが解析しています...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    # 5. 結果表示
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")