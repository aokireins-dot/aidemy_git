import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. AIモデルの読み込み設定
@st.cache_resource
def load_my_model():
    try:
        # Colab環境（2.19.0）と同じ状態で読み込みます
        return tf.keras.models.load_model('final_meat_model.keras')
    except Exception as e:
        st.error(f"AIモデルの読み込みに失敗しました。原因: {e}")
        return None

# モデルを変数に代入
model = load_my_model()

st.title("🥩 焼肉・肉質判定アプリ")

# 2. 安全装置：モデルが正常に読み込めていない場合はここで処理を中断する
if model is None:
    st.warning("AIモデルがまだ準備できていません。デプロイ完了までお待ちください。")
    st.stop() # これにより、読み込み失敗時の NameError を防ぎます

# 3. 画像のアップロード
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像を開いてRGB形式に変換
    image = Image.open(uploaded_file).convert('RGB')
    
    # 2026年最新仕様: 従来の use_container_width の代わりに width='stretch' を使用
    st.image(image, caption='判定中...', width='stretch')
    
    # 4. AIが判定できる形に画像を加工（前処理）
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. 判定実行
    with st.spinner('AIが解析しています...'):
        prediction = model.predict(img_array)
        score = prediction[0][0]
    
    # 6. 結果の表示
    st.divider()
    if score > 0.5:
        st.success(f"【判定結果】 合格（良質な肉質）")
        st.write(f"信頼度: {score * 100:.2f}%")
    else:
        st.error(f"【判定結果】 不合格（基準外）")
        st.write(f"信頼度: {(1 - score) * 100:.2f}%")