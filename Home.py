import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

#header
st.set_page_config('Skripsi',layout='wide')
st.title("Klasifikasi Kualitas Benih Jagung Menggunakan EfficientNet dan DeepSMOTE")
abstrak="\tBenih jagung merupakan salah satu faktor yang sangat mempengaruhi produktivitas dan produksi jagung. Klasifikasi kualitas benih jagung juga termasuk menjaga kualitas benih jagung. Secara konvensional, klasifikasi kualitas benih masih melalui pengamatan visual. Mengenai hal itu, CNN menjadi jawaban untuk mengatasi klasifikasi kualitas benih jagung. Dalam meningkatkan akurasi CNN ada berbagai cara yang dilakukan. Di beberapa penelitian meningkatkan CNN berdasarkan kedalamannya, lebarnya, ataupun resolusinya. EfficientNet melakukan peningkatan akurasi dengan menyeimbangkan semua dimensi, yaitu lebar, kedalaman, dan resolusi jaringan. Pada dataset kualitas benih jagung terdiri dari kelas discolored, pure, broken, dan silkcut dengan jumlah data yang tidak seimbang. Untuk menyeimbangkannya, penulis menggunakan oversampling berbasis DeepSMOTE. Pada tahapan penelitian ini, dimulai dengan memasukkan dataset, melakukan image processing seperti segmentasi dan resize, melakukan oversampling berbasis DeepSMOTE, setelah itu melakukan pelatihan dengan EfficientNet yang dioptimasi dengan RMSProp, lalu melakukan k-fold cross validation, dari model terbaik, akan diuji performanya dengan confusion matrix, accuracy, precision, recall, dan f1-score. Skenario pengujian yang akan dilakukan adalah membandingkan perbesaran arsitektur EfficientNet B0 hingga B2 antara menggunakan DeepSMOTE dan tidak menggunakan DeepSMOTE. Hasil dari penelitian ini didapatkan dari eksperimen dengan performa terbaik adalah klasifikasi EfficientNet B0 dan penyeimbangan data DeepSMOTE. Performa akurasi, precision, recall, dan f1-score dari eksperimen tersebut berturut-urut 79.2%, 79.9%, 79.2%, dan 78.8%."
intro="<p style='text-align:justify;'>"+abstrak+"</p>"
st.write("_"*100)
img_class = Image.open('media/class.png')
st.image(img_class, width=600)

st.write("_"*100)
alur_pelatihan = ""
performa_validasi = "Performa"
st.markdown("### Pendahuluan",unsafe_allow_html=True)
st.markdown(intro,unsafe_allow_html=True)
st.markdown("### Alur Penelitian",unsafe_allow_html=True)
# st.markdown(intro,unsafe_allow_html=True)
img_alur = Image.open('media/alur.png')
st.image(img_alur)

