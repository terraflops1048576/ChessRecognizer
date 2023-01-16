import torchvision
import torch
import torchvision.transforms as T
import os
from typing import Dict, List, Tuple, Any
from PIL import Image

torch.manual_seed(0)
class ChessImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: Dict[str, Any], prefix_path: str, max_images: int=100000):
        self.data_dict = data_dict
        self.labels = []
        self.images = []
        transform = T.Compose([T.PILToTensor(), T.Resize(224), T.CenterCrop(224)])
        for img, img_idx in zip(self.data_dict['images'], range(max_images)):
            image_path = os.path.join(prefix_path, img['imagePath'])
            if img_idx + 1 % 50 == 0:
                print(f"Loaded {img_idx} GIFs!")
            with Image.open(image_path) as gif:
                for i, label in zip(range(gif.n_frames), img['boardFens']):
                    gif.seek(i)
                    self.images.append(transform(gif.convert("RGB")))
                    self.labels.append(label)
                    
            
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.labels)
