# -*- coding: utf-8 -*-
"""ResultExtractor.ipynb

This script is only used to convert the results into a usable LaTeX format

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ByG4NpKORuypWmQCpoYrq6yw7GTLx1UI
"""

models = {"efficient_b0":"Efficient b0",\
"efficient_b4":"Efficient b4",\
"swin_tiny":"Swin Tiny",\
"swin_base":"Swin Base",\
"swin_large":"Swin Large",\
"vgg16":"VGG16",\
"res50":"Resnet50",\
"resnext":"ResNext",\
"coatnet":"CoAtNet",\
"bit": "BiT",\
"vit_base": "ViT Base",\
"vit_large": "ViT Large",\
"deit_base": "DeiT Base"}

import pandas as pd
import os

pd.DataFrame(columns=["Accuracy", "AUC", "Avg. precision", "Precision"])

# dir = "results/MODELS2/"

test = ["FFpp1", "FFpp2", "FFpp3", "FFpp4", "StyleGAN", "VQGAN", "PNDM", "DDPM", "LDM", "DDIM", "ProGAN"]

text = dict.fromkeys(test, '')

# for filename in os.listdir(dir):

filename = "/home/abdallah/Deepfake-Detection/models_trained/swin_tiny_Forensics/swin_tiny_0106_FFpp2/0406_swin_tiny-FFpp2.csv"
# tmp = pd.read_csv(os.path.join(dir, filename),skiprows=1).set_index("testset")
tmp = pd.read_csv(filename,skiprows=1).set_index("testset")

tmp = tmp[["accuracy", "roc score", "avg precision", "precision"]]
tmp.rename(columns={'accuracy': 'Accuracy', 'roc score': 'AUC', 'avg precision': 'Avg. precision', "precision": "Precision"}, inplace=True)
for key in models.keys():
  if key in filename:
    for model_test in test:
      if "FFpp" not in model_test:
        name = model_test
      if model_test == "FFpp1":
        name = "Deepfakes"
      if model_test == "FFpp2":
        name = "Face2Face"
      if model_test == "FFpp3":
        name = "FaceSwap"
      if model_test == "FFpp4":
        name = "NeuralTextures"
      # text[model_test] += f"{name}         & {tmp['Accuracy'].loc[model_test]:.5f}          & {tmp['AUC'].loc[model_test]:.5f}         & {tmp['Avg. precision'].loc[model_test]:.5f}              & {tmp['Precision'].loc[model_test]:.5f}      \\\\ \hline\n"
      text[model_test] += f"{tmp['Avg. precision'].loc[model_test]:.5f} / {tmp['AUC'].loc[model_test]:.5f}"
      # AP / AUC & AP / AUC & AP / AUC

GANS = text["VQGAN"] + " & " + text["StyleGAN"] + " & " + text["VQGAN"]
DMs = text["DDIM"] + " & " + text["DDPM"] + " & " + text["PNDM"] + " & " +  text["LDM"]
FFpps = text["FFpp1"] + " & " + text["FFpp2"] + " & " + text["FFpp3"] + " & " +  text["FFpp4"]

print(GANS)
print(DMs)
print(FFpps)
# print(text["FFpp1"])
# print(text["FFpp2"])
# print(text["FFpp3"])
# print(text["FFpp4"])

# print(text["ProGAN"])
# # print("StyleGAN")
# print(text["StyleGAN"])
# # print("VQGAN")
# print(text["VQGAN"])
# # print("LDM")
# print(text["LDM"])
# # print("DDPM")
# print(text["DDPM"])

# print(text["DDIM"])
# # print("PNDM")
# print(text["PNDM"])