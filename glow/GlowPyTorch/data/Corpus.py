import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
import time
import random
import sys
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import PIL.ImageOps
from sklearn.model_selection import train_test_split
import torchaudio
import librosa

class DatasetGenerator(Dataset):
    def __init__(self, path_to_vc_dir):
        """
        Args:
            path_to_vc_dir: path to voice directory.
            transform: optional transform to be applied on a sample.
        """
        wav_list = []
        lbl_list = []
        for root, dirs, files in os.walk(path_to_vc_dir):
            for file in files:
                if 'wav' in file:
                    wav_path = os.path.join(root + '/' + file)
                    wav_list.append(wav_path)
                    lbl = root.split('/')[-1][1:]
                    lbl_list.append([int(lbl)])

        self.wav_list = wav_list
        self.lbl_list = lbl_list
        self.to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        self.max_mel_length = 192

    def preprocess(self, wave):
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        mel_length = mel_tensor.size(2)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, :, random_start:random_start + self.max_mel_length]
        return mel_tensor

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        wav_path = self.wav_list[index] 
        audio, source_sr = librosa.load(wav_path, sr=24000)
        audio = audio / np.max(np.abs(audio))
        audio.dtype = np.float32
        audio = self.preprocess(audio)

        lbl = self.lbl_list[index]
        label = torch.as_tensor(lbl, dtype=torch.long)
  
        return audio, label

    def __len__(self):
        return len(self.wav_list)

path_to_vc_dir = '/data/tmpexec/opencode/StarGANv2-VC/Data/Data/'

def get_train_dataset_corpus():
    dataset_corpus = DatasetGenerator(path_to_vc_dir)
    return dataset_corpus

if __name__ == "__main__":

    datast = get_train_dataset_corpus()
    data_loader = DataLoader(datast,batch_size=8,shuffle=False,num_workers=0,drop_last=True,)
    for batch_idx,  (voc, lbl) in enumerate(data_loader):
        print(voc.shape)
        print(lbl.shape)