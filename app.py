import streamlit as st
import tensorflow as tf
import tf_keras # 互換性エラーを解決するためのライブラリ
from PIL import Image
import numpy as np

# モデルの読み込みを高速化するためのキャッシュ設定
@st.cache_resource
def load_my_model():
    # Keras 3の互換性問題を回避するため tf_keras を使用してロードします
    # ファイル名が GitHub 上のものと完全に一致しているか確認してください
    return tf_keras.models.load_model('final_meat_model.keras')

# モデルの準備
try:
    model = load_my_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。ファイル名を確認してください: {e}")

st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の画像をアップロードすると、AIが「合格」か「不合格」かを判定します。")

# 画像アップローダーの設置
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の読み込みと表示
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # AIが学習した時と同じ前処理（224x224にリサイズし、0-1に正規化）
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 判定実行
    with st.spinner('AIが解析しています...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    st.divider()
    
    # 結果の表示（0.5を基準に判定）
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")