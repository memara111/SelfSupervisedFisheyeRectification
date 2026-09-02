import os

from . import BaseDataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


class EvalDataset(BaseDataset):
    """Every image directly inside ``data_path``; no list file and no labels.

    ``height``/``width`` resize the *network input* to the scale the model was trained
    at -- feeding it arbitrary resolutions used to be a silent accuracy cliff.
    """

    def __init__(self, data_path, transform=None, height=None, width=None):
        super().__init__(list_path=None, transform=transform)
        self.data_path = data_path
        self.height, self.width = height, width

        if not os.path.isdir(data_path):
            raise NotADirectoryError("No such directory: {}".format(data_path))

        for file in sorted(os.listdir(data_path)):
            full = os.path.join(data_path, file)
            if os.path.isfile(full) and file.lower().endswith(IMAGE_EXTENSIONS):
                self.img_list.append(full)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_file = self.img_list[idx]

        image = self.loadImage(image_file)
        if self.height and self.width:
            image = image.resize((self.width, self.height))
        if self.transform:
            image = self.transform(image)

        return image, {"path": image_file}
