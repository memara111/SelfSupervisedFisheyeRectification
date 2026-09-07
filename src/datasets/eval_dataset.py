import os
import torch
from PIL import Image
from . import BaseDataset


class EvalDataset(BaseDataset):
    def __init__(self, data_path, transform=None, width=None, height=None):
        super().__init__(transform=transform)

        # BUG FIX: the original __getitem__ fed images straight into
        # `transform` (ToTensor + Normalize only, no resize) at their
        # native resolution. The model is trained exclusively at
        # (width, height) == the training config's resolution (e.g.
        # 256x144), via FisheyeEffector's internal resize in
        # DistortDataset. Feeding a real photo at, say, 1920x1080 puts
        # every feature map wildly out of the distribution the network's
        # filters were calibrated for, producing collapsed, weakly
        # content-sensitive predictions clustered around a near-constant
        # value regardless of the image's actual distortion.
        #
        # width/height must match cfg['DATASET']['WIDTH'/'HEIGHT'] for the
        # checkpoint being loaded.
        self.width = width
        self.height = height

        for file in os.listdir(data_path):
            if os.path.isfile(os.path.join(data_path, file)):
                self.img_list.append(os.path.join(data_path, file))

    def __getitem__(self, idx):
        image_file = self.img_list[idx]

        image = Image.open(image_file).convert("RGB")

        if self.width is not None and self.height is not None:
            image = image.resize((self.width, self.height))

        if self.transform:
            image = self.transform(image)

        return image, image_file
