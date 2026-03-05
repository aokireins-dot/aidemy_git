Renderへの移行、素晴らしい決断です！Renderで動かす場合、環境をこちらで制御できるため、最も標準的かつ安定した読み込み方法を使用するのがベストです。

最新の .keras 形式のモデルを読み込み、Renderのポート設定に対応させた app.py の決定版がこちらです。

📄 app.py の最新コード
この内容をすべてコピーして、現在の app.py に上書き保存してください。

Python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# モデルの読み込みを高速化するためのキャッシュ設定
@st.cache_resource
def load_my_model():
    # Renderにアップロードした .keras ファイルの名前と一致させてください
    return tf.keras.models.load_model('meat_quality_model.keras')

# モデルの準備
try:
    model = load_my_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")

st.title("🥩 焼肉・肉質判定アプリ")
st.write("お肉の画像をアップロードすると、AIが「合格」か「不合格」かを判定します。")

uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像の表示
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='判定中...', use_container_width=True)
    
    # AIが学習した時と同じ前処理
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 判定実行
    with st.spinner('AIが解析しています...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    st.divider()
    
    # 結果の表示
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")