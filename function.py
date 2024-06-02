import torch
import torch.nn as nn
from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

args={}
args['class']=['Discolored', 'Pure', 'Broken', 'Silkcut']
args['model']="model/best_lib_eff_b0_0fold_.pth"

def segmentasi(batchImg):
    hasil_segmentasi=[]
    for img in batchImg:
        citra_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        mean = citra_gray.mean()
        mask = torch.where(citra_gray <= mean.item(), 1, 0)
        hasil = mask.unsqueeze(2) * img.to(torch.float32)
        hasil_segmentasi.append(hasil.to(torch.float32))

    return torch.stack(hasil_segmentasi, dim=0)
def preprocess(data):
    return segmentasi(data)

class EfficientNetClassification(nn.Module):
    def __init__(self, version):
        super(EfficientNetClassification, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-'+version)
        in_features = self.efficientnet._fc.in_features
        num_class = len(args['class'])
        self.efficientnet._fc = nn.Linear(in_features, num_class)
        self.version = version

    def forward(self, x):
        return self.efficientnet(x)

def classification(model,data,device):
    with torch.no_grad():
        model.load_state_dict(torch.load(args['model'], map_location=device) ,strict=False)
        model.eval()
        data = data.unsqueeze(0)
        data = preprocess(data)
        outputs = model(data)
        _, predict = torch.max(outputs.data,1)
    return args['class'][predict]
