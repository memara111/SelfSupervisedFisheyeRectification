from PIL import Image
import torch.utils as utils


class BaseDataset(utils.data.Dataset):
    """List-file backed dataset.

    Each line of ``list_path`` is ``<image path>`` or ``<image path> <label file>``.
    Blank lines are ignored, so a trailing newline in a generated ``.lst`` does not
    produce a phantom sample.
    """

    def __init__(self, list_path=None, transform=None):
        self.list_path = list_path
        self.transform = transform

        self.img_list = []
        if self.list_path is not None:
            self.img_list = self.readList(self.list_path)

    @staticmethod
    def readList(list_path):
        """One entry per non-empty line; subclasses that need fields split themselves."""
        with open(list_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.img_list)

    def loadImage(self, image_file):
        # the effector and the network both require H x W x 3, so any palette /
        # grayscale / RGBA file has to be normalised here or it blows up downstream
        return Image.open(image_file).convert("RGB")

    def __getitem__(self, idx):
        image_file, label_file = self.img_list[idx].split()

        image = self.loadImage(image_file)
        if self.transform:
            image = self.transform(image)

        with open(label_file, "r", encoding="utf-8") as f:
            label = int(f.readline().strip())

        # used to return int(label_file) -- i.e. the *filename* cast to int
        return image, label
