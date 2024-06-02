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

st.markdown("### Interface EfficientNet",unsafe_allow_html=True)
files=st.file_uploader("Masukan File Benih Jagung", type=["jpg", "png", "jpeg"])
if files is not None:
    image = Image.open(files)
    image = image.resize((224,224))
    # pil to tensor
    tensor_img = transforms.ToTensor()(image)
    model = EfficientNetClassification('b0')
    predict = classification(model, tensor_img, 'cuda')
    kolom=st.columns(2)
    kolom[0] = kolom[0].image(image, caption=f"Kelas {predict}",width=300)
    kolom[1]=kolom[1].write(f"#### Citra tersebut termasuk jenis : {predict}")

# st.markdown(intro,unsafe_allow_html=True)
# run class efficientNet

