import pandas as pd
import torch
import os
from skimage import io
from PIL import Image
from torch.utils.data import Dataset

class ImageNet(Dataset):

  def __init__(self, data_file, root_dir, transform = None):
    self.annotations = pd.read_csv(data_file, delim_whitespace=True)
    self.root_dir = root_dir
    self.transform = transform

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    # image = io.imread(img_path)
    image = Image.open(img_path)
    if image.mode != "RGB":
      image = image.convert("RGB")
    y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

    if self.transform:
      image = self.transform(image)

    return (image, y_label)
