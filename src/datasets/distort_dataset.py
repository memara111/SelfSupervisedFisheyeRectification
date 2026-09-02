import os
import random

import torch
from . import BaseDataset
from . import FisheyeEffector


class DistortDataset(BaseDataset):
    '''
    Expected Dataset Format is:
    DATASET_ROOT/
        train.lst
        val.lst
        test.lst
        hoge.jpg
        fuga.jpg
        piyo.jpg
        ...

    Labels are the self-supervision signal, not a class index: the three key points the
    effector projected into the fisheye image, plus the magnification the crop applied
    to it (the loss needs it to get back to the frame the model parameter is defined
    in). ``distortion`` is carried along for reporting only.
    '''

    def __init__(
        self,
        list_path,
        height=720,
        width=1280,
        transform=None,
        distortions=None,
        return_distortion=False,
        output_dir=None,
        deterministic=False,
        crop=True,
        num_effectors=10,
    ):
        super().__init__(list_path=None, transform=transform)
        self.return_distortion = return_distortion
        self.deterministic = deterministic
        self.height, self.width = height, width
        self.output_dir = output_dir
        self.num_effectors = num_effectors
        self.crop = crop

        self.list_path = list_path
        self.img_list = []
        if self.list_path is not None:
            self.img_list = self.readList(self.list_path)

        self.updateEffector(distortions=distortions)

    @staticmethod
    def _makeDistortions(distortions, num_effectors):
        # a mutable default argument used to leak between datasets; None means
        # "random per-dataset effectors"
        if distortions is not None and len(distortions) > 0:
            return [float(d) for d in distortions]
        return [random.uniform(-1.0, 1.0) for _ in range(num_effectors)]

    def updateEffector(self, distortions=None):
        self.distortions = self._makeDistortions(distortions, self.num_effectors)
        self.effectors = [
            FisheyeEffector(height=self.height, width=self.width, distortion=d, crop=self.crop)
            for d in self.distortions
        ]
        return self.distortions

    def __getitem__(self, idx):
        image_file = self.img_list[idx]
        # val/test pass deterministic=True so that the same image is measured under the
        # same effector every epoch (otherwise "validation loss" is noise)
        effector = (
            self.effectors[idx % len(self.effectors)]
            if self.deterministic
            else random.choice(self.effectors)
        )

        image = self.loadImage(image_file)
        image = image.resize((self.width, self.height))
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            image.save(os.path.join(self.output_dir, "{}_org.jpg".format(idx)))

        image = effector(image)
        if self.output_dir:
            image.save(os.path.join(self.output_dir, "{}_dis.jpg".format(idx)))

        if self.transform:
            image = self.transform(image)

        label = {
            "coords": torch.tensor(effector.getKeyCoordinates(), dtype=torch.float32),
            "expansion": torch.tensor(effector.getExpansion(), dtype=torch.float32),
            "distortion": torch.tensor(effector.getDistortion(), dtype=torch.float32),
            "path": image_file,
        }
        return image, label
