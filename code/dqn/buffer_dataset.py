import os
import glob
import csv
import random

from PIL import Image
from numpy import float32
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class BufferDataset(Dataset):
    def __init__(self, buffer):
        super(BufferDataset, self).__init__()
        self.data = buffer

    def __len__(self):
        return len(self.data)



    def __getitem__(self, idx):
        # idx-[0->len(images)]
        images = torch.tensor(self.data._encode_sample([idx], True)[0], dtype=torch.float32)[0]
        return images





