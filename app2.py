# app2.py
import streamlit as st
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(__file__))
from auto_pipeline import auto_remove_chinese_text
from PIL import Image


st.title("Delete China Text 🤖 ")

uploaded_file = st.file_uploader("Import Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tạo thư mục tạm trên Windows
    os.makedirs("temp", exist_ok=True)
    
    image_path = os.path.join("temp", uploaded_file.name)

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(image_path, caption="Ảnh gốc")

    # if st.button("Delete Text...🚀"):
    #     with st.spinner("In Processing...⚙️"):
    #         result_img = auto_remove_chinese_text(image_path)
            
    #         result_pil = Image.fromarray(result_img)
    #         st.image(result_pil, caption="Ảnh đã xử lý")


    #         # Tùy chọn tải ảnh về
    #         #result_img.save("output.jpg")

    #         result_pil.save("output.jpg")

    if st.button("Delete Text...🚀"):
    # 🔄 Tạo vùng tạm để hiển thị GIF động và thông báo
        loading_placeholder = st.empty()

        with loading_placeholder.container():
            loading_placeholder.markdown(
                """<div style="text-align: center;">
                    <img src="https://images.emojiterra.com/google/noto-emoji/animated-emoji/2699.gif" width="60">
                    <p style="font-weight: bold;">Đang xử lý, vui lòng chờ...</p>
                </div>""",
                unsafe_allow_html=True
            )

        # 🧠 Xử lý ảnh
        result_img = auto_remove_chinese_text(image_path)
        result_pil = Image.fromarray(result_img)
        result_pil.save("output.jpg")

        # ❌ Xoá GIF động sau khi xử lý xong
        loading_placeholder.empty()

        # ✅ Hiển thị kết quả
        st.image(result_pil, caption="Ảnh đã xử lý")

        # 📥 Nút tải ảnh
        with open("output.jpg", "rb") as f:
            st.download_button(
                label="📥 Tải ảnh đã xử lý",
                data=f,
                file_name="output.jpg",
                mime="image/jpeg"
            )
