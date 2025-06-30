import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Konfigurasi halaman
st.set_page_config(page_title="Footwear Classifier", page_icon="👟", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model.save("model.keras")
    return model

model = load_model()

# Daftar kelas sesuai dataset
class_names = ['Boot', 'Sandal', 'Shoe']

# Sidebar navigasi
st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman", ["🏠 Home", "🧠 Klasifikasi"])

# Halaman Home
if page == "🏠 Home":
    st.title("👟 Footwear Classifier App")
    st.markdown("---")
    st.markdown("""
    Selamat datang di aplikasi klasifikasi gambar alas kaki berbasis AI!  
    Aplikasi ini dapat mengenali gambar **Boot**, **Sandal**, atau **Shoe** hanya dari foto 📸.

    **🔍 Cara Menggunakan:**
    - Pergi ke halaman **Klasifikasi**
    - Unggah gambar alas kaki
    - Dapatkan hasil prediksi dan probabilitasnya!

    ---
    """)

# Halaman Klasifikasi
elif page == "🧠 Klasifikasi":
    st.title("🔎 Klasifikasi Gambar Alas Kaki")
    
    st.markdown("---")
    
    st.write("Unggah gambar alas kaki (Boot, Sandal, atau Shoe), dan model akan memprediksi jenisnya.")

    uploaded_file = st.file_uploader("📤 Upload gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="🖼️ Gambar yang Diupload", use_container_width=True)

        # Preprocessing
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediksi
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_names[predicted_class[0]]
        probability = predictions[0][predicted_class[0]]

        # Kotak hasil prediksi
        with st.container():
            st.markdown(
    f"""
    <div style="background-color:#f0f2f6; padding: 25px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align:center;">
        <h2 style="color:#2E86C1; margin:0;">✅ Hasil Prediksi: {predicted_label}</h2>
        <p style="font-size: 20px; margin-top: 8px;">Probabilitas: {probability:.2%}</p>
    </div>
    """,
    unsafe_allow_html=True
)

    # Distribusi probabilitas (judul)
        st.markdown("<h4>📊 Distribusi Probabilitas</h4>", unsafe_allow_html=True)

    # Progress bar tiap kelas
        for i, label in enumerate(class_names):
            prob = predictions[0][i]
            st.markdown(f"**{label}:** {prob:.2%}")
            st.progress(min(int(prob * 100), 100))

    # Penutup div
        st.markdown("</div>", unsafe_allow_html=True)


