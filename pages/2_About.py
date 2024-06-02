import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from function import EfficientNetClassification, classification

#header
st.set_page_config('Skripsi',layout='wide')
st.title("Klasifikasi Kualitas Benih Jagung Menggunakan EfficientNet dan DeepSMOTE")
st.write("_"*100)
bio = """
<p style='text-align:justify;'>
Penelitian ini dibuat bertujuan untuk melakukan klasifikasi kualitas benih jagung
menggunakan EfficientNet dan DeepSMOTE. Sejauh mana performa dari model EfficientNet dalam melakukan
klasifikasi. Dan sejauh mana performa dari DeepSMOTE dalam menyeimbangkan data
</p>

<p style='text-align:justify;'>
Saya, Muhamad Fahmi Ammar, adalah seseorang Mahasiswa Universitas Trunojoyo Madura sebagai penulis penelitian ini.
Saya menyukai bidang kreatif, seperti designer dan video editor. Saya juga menyukai bidang
informatika, seperti data science dan machine learning. Saya bisa
mengoperasikan aplikasi Photoshop dan Inkscape. Saya juga menguasai
pemrograman Python. Saya akan terus belajar, menggali potensi diri saya
lebih dalam, agar berguna bagi kalangan luas. </p>
"""
st.write("### Biodata Penulis")
kolom = st.columns(2)
# kolom = st.columns(2)
kolom[0] = kolom[0].image(Image.open('media/penulis.jpg'), width=300)
kolom[1] = kolom[1].write(bio,unsafe_allow_html=True)
st.write("<footer>",unsafe_allow_html=True)

